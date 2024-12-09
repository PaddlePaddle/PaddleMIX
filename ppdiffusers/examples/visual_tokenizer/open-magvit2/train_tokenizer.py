import argparse
import os
import random
import importlib

import numpy as np
import paddle

import yaml
from omegaconf import OmegaConf

from taming.models.lfqgan import VQModel
from paddlenlp.utils.log import logger



custom_collate = paddle.io.dataloader.collate.default_collate_fn


def load_vqgan_new(config, ckpt_path=None, is_gumbel=False):
    model = VQModel(**config['model']['init_args'])
    if ckpt_path is not None:
        sd = paddle.load(path=str(ckpt_path))
        missing, unexpected = model.set_state_dict(state_dict=sd)
    return model

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not 'target' in config:
        raise KeyError('Expected key `target` to instantiate.')
    return get_obj_from_str(config['target'])(**config.get('params', dict()))


class WrappedDataset(paddle.io.Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig():

    def __init__(self, batch_size, train=None, validation=None, test=None,
        wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = (num_workers if num_workers is not None else 
            batch_size * 2)
        if train is not None:
            self.dataset_configs['train'] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs['validation'] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs['test'] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap
        self.setup()


    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict((k, instantiate_from_config(self.
            dataset_configs[k])) for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset=self.datasets['train'],
            shuffle=True,
            batch_size=self.batch_size
        )
        return paddle.io.DataLoader(dataset=self.datasets['train'],
            num_workers=self.num_workers,batch_sampler=batch_sampler,collate_fn=custom_collate)

    def _val_dataloader(self):
        return paddle.io.DataLoader(dataset=self.datasets['validation'],
            batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=custom_collate, shuffle=False)

    def _test_dataloader(self):
        return paddle.io.DataLoader(dataset=self.datasets['test'],
            batch_size=self.batch_size, num_workers=self.num_workers,
            collate_fn=custom_collate, shuffle=False)




def parse_args():
    """paraser args"""
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
   
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    return args


def main(main_args, args, config):
    random.seed(0)
    np.random.seed(0)
    paddle.seed(0)

    world_size = paddle.distributed.get_world_size()
    max_epochs = config["trainer"]["max_epochs"]
    max_steps = config["trainer"]["max_steps"]
    log_every_n_steps = config["trainer"]["log_every_n_steps"]
    num_sanity_val_steps = config["trainer"]["num_sanity_val_steps"]
    
    if world_size>1:
        paddle.distributed.init_parallel_env()

    
    #save_checkpint
    save_path = config["trainer"]["save_path"]
    if save_path is not None:
        os.makedirs(save_path,exist_ok=True)
    else:
        os.makedirs("checkpoints",exist_ok=True)
        save_path = "checkpoints"
    
    save_checkpoint_epochs = config["trainer"]["save_checkpoint_epochs"]
    save_checkpoint_steps = config["trainer"]["save_checkpoint_steps"]

    data_config = config["data"]["init_args"]
    data_module = DataModuleFromConfig(**data_config)
    train_dataloader = data_module._train_dataloader()
    
    model = load_vqgan_new(config, config['ckpt_path'])
    model.configure_optimizers(len(train_dataloader),world_size,max_epochs)
    
    if paddle.distributed.get_world_size() > 1:
        model.encoder = paddle.DataParallel(model.encoder)
        model.decoder = paddle.DataParallel(model.decoder)
        model.loss.discriminator = paddle.DataParallel(model.loss.discriminator)
    
    
    model.on_train_start()
    global_step = 1
    
    
    
    with paddle.amp.auto_cast(enable=True,level='O1',dtype=config["trainer"]['precision']):
        for epoch in range(0, max_epochs):
            for step,inputs in enumerate(train_dataloader):
                log_dict_ae,log_dict_disc = model.training_step(inputs,global_step)
                
                if global_step % log_every_n_steps == 0:
                    logger.info(f"Epoch {epoch}, Step {global_step}: ")
                    logger.info("AE loss:")
                    model.on_log(log_dict_ae)
                    logger.info("Discriminator loss:")
                    model.on_log(log_dict_disc)
                
                
                if global_step % num_sanity_val_steps == 0:
                    log_dict_ae,log_dict_disc = model.validation_step(inputs,global_step)
                    logger.info("AE loss:")
                    model.on_log(log_dict_ae)
                    logger.info("Discriminator loss:")
                    model.on_log(log_dict_disc)
                
                if save_checkpoint_steps is not None and global_step % save_checkpoint_steps == 0 :
                    if paddle.distributed.get_rank() == 0:
                        paddle.save(model.state_dict(),os.path.join(save_path,f"checkpoint_step_{global_step}.pdparams"))

                
                global_step += 1
                model.on_train_batch_end()
                
                if max_steps is not None and global_step > max_steps:
                    break
        
            model.on_train_epoch_end(epoch)
        
            if save_checkpoint_epochs is not None and epoch % save_checkpoint_epochs == 0:
                if paddle.distributed.get_rank() == 0:
                    paddle.save(model.state_dict(),os.path.join(save_path,f"checkpoint_epoch_{epoch}.pdparams"))
                        
            
    


def convert_to_namespace(config_dict):
    """convert args"""
    parser = argparse.ArgumentParser()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            setattr(parser, key, convert_to_namespace(value))
        else:
            parser.add_argument(f"--{key}", type=type(value), default=value)
    return parser.parse_args([])


def namespace_to_dict(namespace):
    """args to dict"""
    result = {}
    for key, value in namespace.__dict__.items():
        if hasattr(value, "__dict__"):
            result[key] = namespace_to_dict(value)
        else:
            result[key] = value
    return result


if __name__ == "__main__":
    main_args = parse_args()
    config = OmegaConf.load(main_args.config)
    train_args = convert_to_namespace(config)
    with open(main_args.config, "r") as f:
        config_dict = yaml.load(f.read(), Loader=yaml.Loader)
    main(main_args, train_args, config_dict)
