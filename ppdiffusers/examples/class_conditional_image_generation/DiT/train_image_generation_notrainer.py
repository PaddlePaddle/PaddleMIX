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
import argparse
import json
import os
import random
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time

import numpy as np
import paddle
import paddle.distributed as dist
from diffusion import create_diffusion
from diffusion.dist_env import set_hyrbid_parallel_seed
from diffusion.dit import DiT
from diffusion.dit_llama import DiT_Llama
from paddle.distributed import fleet
from transport import create_transport
from transport.sit import SiT
from transport.utils import parse_transport_args


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
        name = name.replace("_layers.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].scale_(decay).add_(param.data * one_minus_decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.stop_gradient = not flag


class FeatureDataset(paddle.io.Dataset):
    def __init__(self, features_dir, labels_dir):
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.features_files = sorted(os.listdir(features_dir))
        self.labels_files = sorted(os.listdir(labels_dir))

    def __len__(self):
        assert len(self.features_files) == len(
            self.labels_files
        ), "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]
        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        return features, labels


def main(args):
    # Setup DDP:
    dist.init_parallel_env()
    assert args.global_batch_size % dist.get_world_size() == 0, "Batch size must be divisible by world size."
    rank = dist.get_rank()
    seed = args.global_seed * dist.get_world_size() + rank
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Setup an experiment folder:
    model_config_name = args.config_file.split("/")[-1].replace(".json", "")
    model_name = model_config_name.split("_")[0]
    assert model_name in ["DiT", "LargeDiT", "SiT"], f"Model {model_name} not supported."
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = model_config_name.replace("/", "-")
        experiment_dir = (
            f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        )
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Experiment directory created at {experiment_dir}")

    # Create model:
    if model_name == "DiT":
        model = DiT(**read_json(args.config_file))
    elif model_name == "LargeDiT":
        model = DiT_Llama(**read_json(args.config_file))
    elif model_name == "SiT":
        model = SiT(**read_json(args.config_file))
    else:
        raise NotImplementedError(f"Model {model_name} not supported.")
    assert model.sample_size == args.image_size // 8
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    if dist.get_world_size() > 1:
        model = paddle.DataParallel(model)

    if model_name in ["DiT", "LargeDiT"]:
        diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    elif model_name == "SiT":
        transport = create_transport(
            args.path_type,  # "Linear"
            args.prediction,  # "velocity"
            args.loss_weight,  # None
            args.train_eps,  # 0
            args.sample_eps,  # 0
        )
    else:
        raise NotImplementedError(f"Model {model_name} not supported.")
    print(f"Transformer model Parameters: {sum(p.numpy().size for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=1e-4, weight_decay=0.0)

    # Setup data:
    features_dir = f"{args.feature_path}/imagenet{args.image_size}_features"
    labels_dir = f"{args.feature_path}/imagenet{args.image_size}_labels"
    dataset = FeatureDataset(features_dir, labels_dir)
    train_sampler = paddle.io.DistributedBatchSampler(
        dataset,
        int(args.global_batch_size // dist.get_world_size()),
        num_replicas=None,
        rank=None,
        shuffle=False,
        drop_last=True,
    )
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

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    print(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.squeeze(axis=1)
            y = y.squeeze(axis=1)
            if model_name in ["DiT", "LargeDiT"]:
                t = paddle.randint(0, diffusion.num_timesteps, (x.shape[0],))
            model_kwargs = dict(y=y)

            if model_name in ["DiT", "LargeDiT"]:
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            else:
                loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.clear_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                paddle.device.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = paddle.to_tensor(running_loss / log_steps)
                if dist.get_world_size() > 1:
                    dist.all_reduce(avg_loss)
                avg_loss = avg_loss.item() / dist.get_world_size()
                print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT/SiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pdparams"
                    paddle.save(model.state_dict(), checkpoint_path)
                    print(f"Saved model checkpoint to {checkpoint_path}")

                    ema_checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}_ema.pdparams"
                    paddle.save(ema.state_dict(), ema_checkpoint_path)
                    print(f"Saved ema checkpoint to {ema_checkpoint_path}")

                    paddle.save({"args": args}, f"{checkpoint_dir}/{train_steps:07d}_args.json")
                    paddle.save(opt.state_dict(), f"{checkpoint_dir}/{train_steps:07d}.pdopt")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, default="data/fastdit_imagenet256")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--results_dir", type=str, default="output_notrainer")
    parser.add_argument("--config_file", type=str, default="config/DiT_XL_patch2.json")
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--ckpt_every", type=int, default=5000)
    parse_transport_args(parser)
    args = parser.parse_args()
    print(args)

    strategy = fleet.DistributedStrategy()
    fleet.init(is_collective=True, strategy=strategy)

    sharding_parallel_degree = 1
    hcg = fleet.get_hybrid_communicate_group()
    mp_rank = hcg.get_model_parallel_rank()
    dp_rank = hcg.get_data_parallel_rank()
    sharding_rank = hcg.get_sharding_parallel_rank()
    data_world_rank = dp_rank * sharding_parallel_degree + sharding_rank

    # seed control in hybrid parallel
    set_hyrbid_parallel_seed(args.global_seed, data_world_rank, mp_rank)
    main(args)
