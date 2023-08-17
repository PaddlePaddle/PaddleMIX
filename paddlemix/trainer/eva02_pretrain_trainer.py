# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.trainer.trainer import Trainer
from paddle.io import DataLoader
from tensorboardX import SummaryWriter
#from torch.utils.tensorboard import SummaryWriter
from IPython import embed


def compute_loss(output, label):
    loss_func = paddle.nn.CosineSimilarity(axis=-1)
    loss = loss_func(
        output.astype(dtype='float32'), label.astype(dtype='float32'))
    return -loss.mean()


class EVA02PretrainTrainer(Trainer):
    def __init__(self, **kwargs):
        """
        Implementation of an `Trainer` suitable for EVA-CLIP
        1、selfdefine optimizer for sharding which can't create by passing by args
        2、support for accum_freq
        
        Args:
            kwargs (dict): any arugments to pass to `Trainer`
        
        Returns:
            None
        """
        super().__init__(**kwargs)
        self.beit_like = True
        if self.args.accum_freq > 1:
            self.accum_images = []
            self.accum_labels = []
            self.accu_step = 0
        
        self.iter = 0 # real iter

        self.rank = paddle.distributed.get_rank()
        if self.rank == 0 and self.args.tensorboard:
            self.writer = SummaryWriter(
                self.args.output_dir
            )
            self.logstep = 0

    def training_step(self, model, inputs) -> paddle.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Layer`):
                The model to train.
            inputs (`Dict[str, Union[paddle.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `paddle.Tensor`: The tensor with training loss on this batch.
        """
        
        it = self.iter // self.args.accum_freq
        if self.lr_schedule_values is not None or self.wd_schedule_values is not None:
            for i, param_group in enumerate(self.optimizer._param_groups):
                if self.lr_schedule_values is not None:
                    param_group["learning_rate"] = self.lr_schedule_values[it] * param_group["lr_scale"]
                    for param in param_group['params']:
                        param.optimize_attr['learning_rate'] = self.lr_schedule_values[it] * param_group["lr_scale"]
                if self.wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = self.wd_schedule_values[it]

        if 1:
            samples, images, bool_masked_pos = inputs[0] # [128, 3, 224, 224] [128, 3, 224, 224] [128, 16, 16]   [128]
            # print('sum samples, images, bool_masked_pos: ', self.logstep, samples.sum().item(), images.sum().item(), bool_masked_pos.sum().item())
            inputs = {'samples': samples, 'images': images, 'bool_masked_pos': bool_masked_pos}
        else:
            inputs = {}
            inputs['samples'] = paddle.to_tensor(np.load('../samples.npy').astype(np.float32))[:4, :, :, :]
            inputs['images'] = paddle.to_tensor(np.load('../images.npy').astype(np.int64))[:4, :, :, :]
            inputs['bool_masked_pos'] = paddle.to_tensor(np.load('../bool_masked_pos.npy').astype(np.int64))[:4, :, :]


        if self.args.pipeline_parallel_degree > 1:
            return self.training_pipeline_step(model, inputs)
        elif self.args.accum_freq > 1:
            return self.training_step_accumfreq(model, inputs)


        with self.autocast_smart_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        loss_value = loss.item()

        # for name, param in model.named_parameters():
        #     if 'head' in name or 'patch_embed' in name or 'logit_scale' in name or 'cls_token' in name or 'pos_embed' in name:
        #         if param.grad is not None:
        #             print(name, paddle.abs(param.grad).mean().item())
        #             np.save("pd/{}.npy".format(name.replace('_layers', 'module')), param.grad.numpy())
        #         # else:
        #         #     print('None ', name)
        # print('. loss. ', loss_value)
        # exit()

        # f = open('eva02_B_m38m_model.txt', 'a')
        # for k, v in model.state_dict().items():
        #     f.write('{} {} {}\n'.format(k, v.shape, v.cpu().sum().numpy()))
        # f.close()

        if self.do_grad_scaling:
            self.scaler.unscale_(self.optimizer)
        
        grad_norms = get_grad_norm_and_clip(model, self.args.max_grad_norm)
            
        min_lr = 10.
        max_lr = 0.
        for group in self.optimizer._param_groups:
            min_lr = min(min_lr, group["learning_rate"])
            max_lr = max(max_lr, group["learning_rate"])
        self.curr_lr = max_lr

        if self.rank == 0 and self.args.tensorboard:
            self.writer.add_scalar("loss/loss", loss_value, self.logstep)
            self.writer.add_scalar("opt/grad_norm", grad_norms.item(), self.logstep)
            self.writer.add_scalar("opt/lr", max_lr, self.logstep)
            self.writer.add_scalar("opt/min_lr", min_lr, self.logstep)
            self.writer.add_scalar("opt/lr0", self.optimizer._param_groups[0]["learning_rate"], self.logstep)

            # for name, param in model.named_parameters():
            #     if 'head' in name or 'patch_embed' in name or 'logit_scale' in name or 'cls_token' in name or 'pos_embed' in name:
            #         if param.grad is not None:
            #             print(name, paddle.abs(param.grad).mean().item())
            #             np.save("pd/{}.npy".format(name.replace('_layers', 'module')), param.grad.numpy())
            #         # else:
            #         #     print('None ', name)
            # print('. loss. ', loss.item())
            # exit()

            self.logstep += 1
        
        self.iter += 1
        return loss.detach()

    def _get_learning_rate(self):
        return self.curr_lr

    def training_step_accumfreq(self, model, inputs) -> paddle.Tensor:
        model.train()

        self.accum_images.append(inputs['x'])
        self.accum_labels.append(inputs['labels'])
        self.accu_step += 1

        # If (cnt + 1) % accum_freq is not zero, move on to the next batch.
        if (self.accu_step % self.args.accum_freq) > 0:
            # FIXME this makes data time logging unreliable when accumulating
            return paddle.full([1], 0, dtype="float32")


        # Now, ready to take gradients for the last accum_freq batches.
        # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
        # Call backwards each time, but only step optimizer at the end.
        self.optimizer.clear_grad()
        for j in range(self.args.accum_freq):
            with self.autocast_smart_context_manager():
                inputs_j = {'x': self.accum_images[j], 'labels': self.accum_labels[j]}
                loss = self.compute_loss(model, inputs_j)

            if self.do_grad_scaling:
                self.scaler.scale(loss / self.args.accum_freq).backward()
            else:
                (loss / self.args.accum_freq).backward()

        # clear for next accu batches
        self.accum_images.clear()
        self.accum_labels.clear()
        self.accu_step = 0

        loss_value = loss.item()

        if self.do_grad_scaling:
            self.scaler.unscale_(self.optimizer)

        grad_norms = get_grad_norm_and_clip(model, self.args.max_grad_norm)

        min_lr = 10.
        max_lr = 0.
        for group in self.optimizer._param_groups:
            min_lr = min(min_lr, group["learning_rate"])
            max_lr = max(max_lr, group["learning_rate"])
        self.curr_lr = max_lr

        if self.rank == 0 and self.args.tensorboard:
            self.writer.add_scalar("loss/loss", loss_value, self.logstep)
            self.writer.add_scalar("opt/grad_norm", grad_norms.item(), self.logstep)
            self.writer.add_scalar("opt/lr", max_lr, self.logstep)
            self.writer.add_scalar("opt/min_lr", min_lr, self.logstep)
            self.writer.add_scalar("opt/lr0", self.optimizer._param_groups[0]["learning_rate"], self.logstep)
            self.logstep += 1
            # Note: logstep is not same as iter when accum_freq > 1

        self.iter += self.args.accum_freq
        return loss.detach()

    def get_train_dataloader(self):
        sampler_train = paddle.io.DistributedBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            num_replicas=self.args.data_world_size,
            rank=self.args.data_world_rank,
            shuffle=True,
            drop_last=True)

        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler_train,
            num_workers=self.args.dataloader_num_workers,
            use_shared_memory=True)


class CosineSimilarityLoss(paddle.nn.Layer):

    def __init__(self, data_type=paddle.float32):
        super(CosineSimilarityLoss, self).__init__()
        self.data_type = data_type
        self.loss_func = paddle.nn.CosineSimilarity(axis=-1)

    def forward(self, output, label):
        loss = self.loss_func(output.cast(self.data_type), label.cast(self.data_type))
        return -loss.mean()


def get_grad_norm_and_clip(model, max_norm, norm_type=2.0, error_if_nonfinite=False):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    parameters = model.parameters()
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm) if max_norm else 0.0
    norm_type = float(norm_type)
    if len(grads) == 0:
        return paddle.to_tensor([0.])
    if norm_type == float("inf"):
        norms = [g.detach().abs().max() for g in grads]
        total_norm = norms[0] if len(norms) == 1 else paddle.max(paddle.stack(norms))
    else:
        total_norm = paddle.norm(paddle.stack([paddle.norm(g.detach(), norm_type) for g in grads]), norm_type)

    if max_norm is None or max_norm <= 0.:
        return total_norm

    if error_if_nonfinite and paddle.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = paddle.clip(clip_coef, max=1.0)
    for g in grads:
        clipg = paddle.multiply(g, clip_coef_clamped)
        g.set_value(clipg)

    return total_norm
