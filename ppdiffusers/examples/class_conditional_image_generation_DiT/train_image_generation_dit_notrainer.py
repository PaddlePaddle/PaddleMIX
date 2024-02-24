# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
import os
import json
import argparse
import logging
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import paddle
import paddle.distributed as dist

from diffusion import create_diffusion
from diffusion_trainer.dit import DiT


def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@paddle.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    one_minus_decay = 1.0 - decay
    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        #ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
        ema_params[name].scale_(decay).add_(param.data * one_minus_decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.stop_gradient = not flag


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


class CustomDataset(paddle.io.Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]
        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        #return paddle.to_tensor(features), paddle.to_tensor(labels)
        return features, labels
    

def main(args, local_rank):
    # Setup an experiment folder:
    model_name = args.dit_config_file.split("/")[-1].replace(".json", "")
    if local_rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = model_name.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    model = DiT(**read_json(args.dit_config_file))
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    print(f"DiT Parameters: {sum(p.numpy().size for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=1e-4, weight_decay=0.)

    # Setup data:
    features_dir = f"{args.feature_path}/imagenet256_features"
    labels_dir = f"{args.feature_path}/imagenet256_labels"
    dataset = CustomDataset(features_dir, labels_dir)

    #print("dist.get_world_size()", dist.get_world_size())
    train_sampler = paddle.io.DistributedBatchSampler(
                dataset, 
                int(args.global_batch_size // dist.get_world_size()),
                num_replicas=None,
                rank=None,
                shuffle=False,
                drop_last=True)
    loader = paddle.io.DataLoader(
                dataset,
                batch_sampler=train_sampler,
                num_workers=args.num_workers,
                use_shared_memory=True,
            ) 
    print(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    #model = DistributedDataParallel(model)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.squeeze(axis=1)
            y = y.squeeze(axis=1)
            t = paddle.randint(0, diffusion.num_timesteps, (x.shape[0],))
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            #opt.zero_grad()
            #opt.clear_grad()
            #accelerator.backward(loss)

            # loss_sum = loss.detach().mean()
            # dist.all_reduce(loss_sum)
            #loss_mean = loss / dist.get_world_size()
            loss.backward()


            opt.step()
            opt.clear_grad()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                #torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = paddle.to_tensor(running_loss / log_steps)
                avg_loss = avg_loss.item() / dist.get_world_size()
                
                print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                #if accelerator.is_main_process:
                if local_rank == 0:
                    checkpoint = {
                        "model": model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pdparams"
                    paddle.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, default="data/fastdit_imagenet256")
    parser.add_argument("--results_dir", type=str, default="output_notrainer")
    parser.add_argument("--dit_config_file", type=str, default="config/DiT_XL_patch2.json")
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--ckpt_every", type=int, default=500)
    args = parser.parse_args()
    print(args)

    if 1:
        local_rank = 0
    else:
        local_rank = os.environ['LOCAL_RANK']
        local_rank = eval(local_rank)
    main(args, local_rank)
