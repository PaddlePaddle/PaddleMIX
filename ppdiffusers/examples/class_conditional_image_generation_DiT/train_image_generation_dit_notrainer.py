# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
"""
import paddle
import paddle.distributed as dist
#from paddle.nn.parallel import DistributedDataParallel
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
#from accelerate import Accelerator

from ldm.models import DiT_models
from diffusion import create_diffusion


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@paddle.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    one_minus_decay = 1.0 - decay
    for name, param in model_params.items():
        # name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        #ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
        ema_params[name].scale_(decay).add_(param.data * one_minus_decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        #p.requires_grad = flag
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
        print('len features_files ', len(self.features_files))
        print('len labels_files ', len(self.labels_files))

    def __len__(self):
        assert len(self.features_files) == len(self.labels_files), \
            "Number of feature files and label files should be same"
        return len(self.features_files)

    def __getitem__(self, idx):
        feature_file = self.features_files[idx]
        label_file = self.labels_files[idx]
        features = np.load(os.path.join(self.features_dir, feature_file))
        labels = np.load(os.path.join(self.labels_dir, label_file))
        #return torch.from_numpy(features), torch.from_numpy(labels)
        return paddle.to_tensor(features), paddle.to_tensor(labels)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args, local_rank):
    """
    Trains a new DiT model.
    """

    # Setup an experiment folder:
    #if accelerator.is_main_process:
    if local_rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    # Note that parameter initialization is done within the DiT constructor
    #model = model.to(device)
    ema = deepcopy(model) #.to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    #print(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = paddle.optimizer.AdamW(parameters=model.parameters(), learning_rate=1e-4, weight_decay=0.)

    # Setup data:
    features_dir = f"{args.feature_path}/imagenet256_features"
    labels_dir = f"{args.feature_path}/imagenet256_labels"
    dataset = CustomDataset(features_dir, labels_dir)

    print("dist.get_world_size()", dist.get_world_size())
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

    # loader = paddle.io.DataLoader(
    #     dataset,
    #     #batch_size=int(args.global_batch_size // accelerator.num_processes),
    #     batch_size=int(args.global_batch_size // dist.get_world_size()),
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     use_shared_memory=True,
    #     drop_last=True
    # )
    print(f"Dataset contains {len(dataset):,} images ({args.feature_path})")

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    #model, opt, loader = accelerator.prepare(model, opt, loader)
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
                        "model": model.state_dict(), # 
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
    parser.add_argument("--feature-path", type=str, default="features")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=500)
    args = parser.parse_args()
    print(args)

    if 1:
        local_rank = 0
    else:
        local_rank = os.environ['LOCAL_RANK']
        local_rank = eval(local_rank)
    main(args, local_rank)
