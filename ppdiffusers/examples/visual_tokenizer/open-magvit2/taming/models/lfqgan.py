from dis import dis
import sys
import paddle
from main import instantiate_from_config
from contextlib import contextmanager
from collections import OrderedDict
from taming.modules.diffusionmodules.improved_model import Encoder, Decoder
from taming.modules.vqvae.lookup_free_quantize import LFQ
from taming.modules.scheduler.lr_scheduler import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
from taming.modules.ema import LitEma
from paddlenlp.utils.log import logger


class VQModel(paddle.nn.Layer):

    def __init__(self, ddconfig, lossconfig, n_embed, embed_dim,
        sample_minimization_weight, batch_maximization_weight, ckpt_path=
        None, ignore_keys=[], image_key='image', colorize_nlabels=None,
        monitor=None, learning_rate=None, resume_lr=None, warmup_epochs=1.0,
        scheduler_type='linear-warmup_cosine-decay', min_learning_rate=0,
        use_ema=False, token_factorization=False, stage=None, lr_drop_epoch
        =None, lr_drop_rate=0.1, factorized_bits=[9, 9]):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        
        self.quantize = LFQ(dim=embed_dim, codebook_size=n_embed,
            sample_minimization_weight=sample_minimization_weight,
            batch_maximization_weight=batch_maximization_weight,
            token_factorization=token_factorization, factorized_bits=
            factorized_bits)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer(name='colorize', tensor=paddle.randn(shape
                =[3, colorize_nlabels, 1, 1]))
        if monitor is not None:
            self.monitor = monitor
        self.use_ema = use_ema
        if self.use_ema and stage is None:
            self.model_ema = LitEma(self)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, stage=stage
                )
        self.resume_lr = resume_lr
        self.learning_rate = learning_rate
        self.lr_drop_epoch = lr_drop_epoch
        self.lr_drop_rate = lr_drop_rate
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.automatic_optimization = False
        self.strict_loading = False
        
        self.opt_gen = None
        self.opt_disc = None
        self.scheduler_ae = None
        self.scheduler_disc = None
        
    
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f'{context}: Switched to EMA weights')
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f'{context}: Restored training weights')

    def load_state_dict(self, *args, strict=False):
        """
        Resume not strict loading
        """
        return super().set_state_dict(*args)

    # def state_dict(self):
    #     """
    #     filter out the non-used keys
    #     """
    #     return {k: v for k, v in super().state_dict().items() if 'model_ema' not in k}

    def init_from_ckpt(self, path, ignore_keys=list(), stage='transformer'):
        sd = paddle.load(path=str(path))['state_dict']
        ema_mapping = {}
        new_params = OrderedDict()
        if stage == 'transformer':
            if self.use_ema:
                for k, v in sd.items():
                    if 'encoder' in k:
                        if 'model_ema' in k:
                            k = k.replace('model_ema.', '')
                            new_k = ema_mapping[k]
                            new_params[new_k] = v
                        s_name = k.replace('.', '')
                        ema_mapping.update({s_name: k})
                        continue
                    if 'decoder' in k:
                        if 'model_ema' in k:
                            k = k.replace('model_ema.', '')
                            new_k = ema_mapping[k]
                            new_params[new_k] = v
                        s_name = k.replace('.', '')
                        ema_mapping.update({s_name: k})
                        continue
            else:
                for k, v in sd.items():
                    if 'encoder' in k:
                        new_params[k] = v
                    elif 'decoder' in k:
                        new_params[k] = v
        missing_keys, unexpected_keys = self.set_state_dict(state_dict=
            new_params)
        print(f'Restored from {path}')

    def encode(self, x):
        h = self.encoder(x)
        (quant, emb_loss, info), loss_breakdown = self.quantize(h,
            return_loss_breakdown=True)
        return quant, emb_loss, info, loss_breakdown

    def decode(self, quant):
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _, loss_break = self.encode(input)
        dec = self.decode(quant)
        return dec, diff, loss_break

    def get_input(self, batch, k):
        x = batch[k]
        if len(tuple(x.shape)) == 3:
            x = x[..., None]
        x = x.transpose(perm=[0, 3, 1, 2]).contiguous()
        return x.astype(dtype='float32')

    def on_train_start(self):
        """
        change lr after resuming
        """
        if self.resume_lr is not None:
            self.resume_lr = float(self.resume_lr)
            for opt_gen_param_group, opt_disc_param_group in zip(self.opt_gen.
                param_groups, self.opt_disc.param_groups):
                opt_gen_param_group['lr'] = self.resume_lr
                opt_disc_param_group['lr'] = self.resume_lr

    def training_step(self, batch,global_step):
        self.train()
        x = self.get_input(batch, self.image_key)
        
        xrec, eloss, loss_break = self(x)

        aeloss, log_dict_ae = self.loss(eloss, loss_break, x, xrec, 0, global_step, last_layer=self.get_last_layer(), split='train')

        self.opt_gen.clear_grad(set_to_zero=False)
        aeloss.backward()
        self.opt_gen.step()
        
        discloss, log_dict_disc = self.loss(eloss, loss_break, x, xrec, 1,global_step, last_layer=self.get_last_layer(), split='train')
 
        self.opt_disc.clear_grad(set_to_zero=False)
        discloss.backward()
        self.opt_disc.step()
        
        return log_dict_ae, log_dict_disc
        

    def on_train_batch_end(self):
        if self.use_ema:
            self.model_ema(self)

    def on_train_epoch_end(self,current_epoch):
        self.lr_annealing(current_epoch)

    def lr_annealing(self,current_epoch):
        """
        Perform Lr decay
        """
        if self.lr_drop_epoch is not None:
            if current_epoch + 1 in self.lr_drop_epoch:

                for opt_gen_param_group, opt_disc_param_group in zip(self.opt_gen
                    .param_groups, self.opt_disc.param_groups):
                    opt_gen_param_group['lr'] = opt_gen_param_group['lr'
                        ] * self.lr_drop_rate
                    opt_disc_param_group['lr'] = opt_disc_param_group['lr'
                        ] * self.lr_drop_rate

    @paddle.no_grad()
    def validation_step(self, batch,global_step):
        self.eval()
        if self.use_ema:
            with self.ema_scope():
                log_dict_ae_ema, log_dict_disc_ema = self._validation_step(batch,global_step,
                    suffix='_ema')
                return log_dict_ae_ema, log_dict_disc_ema
        else:
            log_ae_dict,log_disc_dict= self._validation_step(batch,global_step)
            return log_ae_dict,log_disc_dict
            

    def _validation_step(self, batch, global_step,suffix=''):
        x = self.get_input(batch, self.image_key)
        quant, eloss, indices, loss_break = self.encode(x)
        x_rec = self.decode(quant).clip(min=-1, max=1)
        
        aeloss, log_dict_ae = self.loss(eloss, loss_break, x, x_rec, 0,
            global_step, last_layer=self.get_last_layer(), split='val' +
            suffix)
        aeloss.detach()
        
        discloss, log_dict_disc = self.loss(eloss, loss_break, x, x_rec, 1,
            global_step, last_layer=self.get_last_layer(), split='val' +
            suffix)
        discloss.detach()
        
        return log_dict_ae, log_dict_disc

    def configure_optimizers(self,train_dataloader_len, world_size,max_epochs):
        lr = float(self.learning_rate)
    
       
        opt_gen = paddle.optimizer.Adam(parameters=list(self.encoder.parameters())+
                                                   list(self.decoder.conv_out.parameters())+
                                                   list(self.quantize.parameters()),
                                                   learning_rate=lr, beta1=0.5,
                                                   beta2=0.9)
        
        opt_disc = paddle.optimizer.Adam(parameters=self.loss.discriminator.parameters(), 
                                         learning_rate=lr,
                                         beta1=0.5, 
                                         beta2=0.9, weight_decay=0.0)
        
        if paddle.distributed.get_rank() == 0:
            print('step_per_epoch: {}'.format(train_dataloader_len // world_size))
            
            
        step_per_epoch = train_dataloader_len // world_size
        warmup_steps = step_per_epoch * self.warmup_epochs
        training_steps = step_per_epoch * max_epochs
        
        if self.scheduler_type == 'None':
            self.opt_gen = opt_gen
            self.opt_disc = opt_disc
            return
        
        if self.scheduler_type == 'linear-warmup':
            tmp_lr = paddle.optimizer.lr.LambdaDecay(lr_lambda=Scheduler_LinearWarmup(warmup_steps), learning_rate=opt_gen.get_lr())
            opt_gen.set_lr_scheduler(tmp_lr)
            scheduler_ae = tmp_lr
            
            tmp_lr = paddle.optimizer.lr.LambdaDecay(lr_lambda=Scheduler_LinearWarmup(warmup_steps), learning_rate=opt_disc.get_lr())
            opt_disc.set_lr_scheduler(tmp_lr)
            scheduler_disc = tmp_lr
            
        elif self.scheduler_type == 'linear-warmup_cosine-decay':
            multipler_min = self.min_learning_rate / self.learning_rate
            tmp_lr = paddle.optimizer.lr.LambdaDecay(lr_lambda=Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min), learning_rate=opt_gen.get_lr())
            opt_gen.set_lr_scheduler(tmp_lr)
            scheduler_ae = tmp_lr
            
            tmp_lr = paddle.optimizer.lr.LambdaDecay(lr_lambda=Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min), learning_rate=opt_disc.get_lr())
            opt_disc.set_lr_scheduler(tmp_lr)
            scheduler_disc = tmp_lr
        else:
            raise NotImplementedError()
        
        self.opt_gen = opt_gen
        self.opt_disc = opt_disc
        self.scheduler_ae = scheduler_ae
        self.scheduler_disc = scheduler_disc
        return

    def get_last_layer(self):
        if paddle.distributed.get_world_size() > 1:
            return self.decoder._layers.conv_out.weight
        else:
            return self.decoder.conv_out.weight
        

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if tuple(x.shape)[1] > 3:
            assert tuple(xrec.shape)[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log['inputs'] = x
        log['reconstructions'] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == 'segmentation'
        if not hasattr(self, 'colorize'):
            self.register_buffer(name='colorize', tensor=paddle.randn(shape
                =[3, tuple(x.shape)[1], 1, 1]).to(x))
        x = paddle.nn.functional.conv2d(x=x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x
    
    def on_log(self, log_dict):
        logger.info(", ".join(f"{k}: {v.item()}" for k, v in log_dict.items()))
        
        
