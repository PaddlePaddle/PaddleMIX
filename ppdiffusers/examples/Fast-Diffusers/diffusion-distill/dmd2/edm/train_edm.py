# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# code is heavily based on https://github.com/tianweiy/DMD2

import matplotlib

matplotlib.use("Agg")

import argparse
import logging
import os
import shutil
import time

import paddle
import wandb
from data.lmdb_dataset import LMDBDataset
from edm.edm_unified_model import EDMUniModel
from utils import (
    cycle,
    draw_probability_histogram,
    draw_valued_array,
    prepare_images_for_saving,
)

from ppdiffusers.accelerate import Accelerator
from ppdiffusers.accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from ppdiffusers.optimization import get_scheduler


class Trainer:
    def __init__(self, args):

        self.args = args

        accelerator_project_config = ProjectConfiguration(logging_dir=args.output_path)
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=1,  # no accumulation
            mixed_precision="no",
            log_with="wandb",
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )
        set_seed(args.seed + accelerator.process_index)
        self._generator = paddle.Generator().manual_seed(args.seed + accelerator.process_index)
        print(accelerator.state)

        if accelerator.is_main_process:
            output_path = os.path.join(args.output_path, f"time_{int(time.time())}_seed{args.seed}")
            os.makedirs(output_path, exist_ok=False)
            self.output_path = output_path

            if args.cache_dir != "":
                self.cache_dir = os.path.join(args.cache_dir, f"time_{int(time.time())}_seed{args.seed}")
                os.makedirs(self.cache_dir, exist_ok=False)

        self.model = EDMUniModel(args, accelerator)
        self.dataset_name = args.dataset_name
        self.real_image_path = args.real_image_path

        self.dfake_gen_update_ratio = args.dfake_gen_update_ratio
        self.num_train_timesteps = args.num_train_timesteps

        self.cls_loss_weight = args.cls_loss_weight

        self.gan_classifier = args.gan_classifier
        self.gen_cls_loss_weight = args.gen_cls_loss_weight
        self.no_save = args.no_save
        self.previous_time = None
        self.step = 0
        self.cache_checkpoints = args.cache_dir != ""
        self.max_checkpoint = args.max_checkpoint

        self.logger = logging.getLogger("dmd2")

        formatter = logging.Formatter(
            "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(formatter)

        self.logger.addHandler(consoleHandler)
        self.logger.setLevel(logging.DEBUG)

        if args.ckpt_only_path is not None:
            if accelerator.is_main_process:
                print(f"loading checkpoints without optimizer states from {args.ckpt_only_path}")
            from safetensors.paddle import load_file

            generator_path = os.path.join(args.ckpt_only_path, "model.safetensors")
            guidance_path = os.path.join(args.ckpt_only_path, "model_1.safetensors")

            generator_state_dict = load_file(generator_path)
            guidance_state_dict = load_file(guidance_path)

            print(self.model.feedforward_model.load_state_dict(generator_state_dict, strict=False))
            print(self.model.guidance_model.load_state_dict(guidance_state_dict, strict=False))

            self.step = int(args.ckpt_only_path.replace("/", "").split("_")[-1])

        if args.generator_ckpt_path is not None:
            if accelerator.is_main_process:
                print(f"loading generator checkpoints from {args.generator_ckpt_path}")
            from safetensors.paddle import load_file

            generator_path = os.path.join(args.generator_ckpt_path, "model.safetensors")
            print(self.model.feedforward_model.load_state_dict(load_file(generator_path)))

        # also load the training dataset images, this will be useful for GAN loss
        real_dataset = LMDBDataset(args.real_image_path)

        real_image_dataloader = paddle.io.DataLoader(
            real_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers
        )

        real_image_dataloader = accelerator.prepare(real_image_dataloader)
        self.real_image_dataloader = cycle(real_image_dataloader)

        # actually this scheduler is not very useful (it warms up from 0 to max_lr in 500 / num_gpu steps), but we keep it here for consistency
        self.scheduler_guidance = get_scheduler(
            "constant_with_warmup",
            learning_rate=args.guidance_lr,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters,
        )

        self.scheduler_generator = get_scheduler(
            "constant_with_warmup",
            learning_rate=args.generator_lr,
            num_warmup_steps=args.warmup_step,
            num_training_steps=args.train_iters,
        )

        self.optimizer_guidance = paddle.optimizer.AdamW(
            parameters=[param for param in self.model.guidance_model.parameters() if param.requires_grad],
            learning_rate=self.scheduler_guidance,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.01,
        )
        self.optimizer_generator = paddle.optimizer.AdamW(
            parameters=[param for param in self.model.feedforward_model.parameters() if param.requires_grad],
            learning_rate=self.scheduler_generator,
            beta1=0.9,
            beta2=0.999,
            weight_decay=0.01,
        )

        # the self.model is not wrapped in ddp, only its two subnetworks are wrapped
        (
            self.model.feedforward_model,
            self.model.guidance_model,
            self.optimizer_guidance,
            self.optimizer_generator,
            self.scheduler_guidance,
            self.scheduler_generator,
        ) = accelerator.prepare(
            self.model.feedforward_model,
            self.model.guidance_model,
            self.optimizer_guidance,
            self.optimizer_generator,
            self.scheduler_guidance,
            self.scheduler_generator,
        )

        self.accelerator = accelerator
        self.train_iters = args.train_iters
        self.batch_size = args.batch_size
        self.resolution = args.resolution
        self.log_iters = args.log_iters
        self.wandb_iters = args.wandb_iters
        self.conditioning_sigma = args.conditioning_sigma

        self.label_dim = args.label_dim
        self.eye_matrix = paddle.eye(self.label_dim)
        self.delete_ckpts = args.delete_ckpts
        self.max_grad_norm = args.max_grad_norm

        if args.checkpoint_path is not None:
            self.load(args.checkpoint_path)

        if self.accelerator.is_main_process:
            run = wandb.init(
                config=args,
                dir=self.output_path,
                **{"mode": "offline", "entity": args.wandb_entity, "project": args.wandb_project},
            )
            wandb.run.log_code(".")
            wandb.run.name = args.wandb_name
            print(f"run dir: {run.dir}")
            self.wandb_folder = run.dir
            os.makedirs(self.wandb_folder, exist_ok=True)

    def load(self, checkpoint_path):
        # Please note that, after loading the checkpoints, all random seed, learning rate, etc.. will be reset to align with the checkpoint.
        self.step = int(checkpoint_path.replace("/", "").split("_")[-1])
        print("loading a previous checkpoints including optimizer and random seed")
        print(self.accelerator.load_state(checkpoint_path, strict=False))
        self.accelerator.print(f"Loaded checkpoint from {checkpoint_path}")

    def save(self):
        # training states
        output_path = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
        print(f"start saving checkpoint to {output_path}")

        self.accelerator.save_state(output_path)

        # remove previous checkpoints
        if self.delete_ckpts:
            for folder in os.listdir(self.output_path):
                if folder.startswith("checkpoint_model") and folder != f"checkpoint_model_{self.step:06d}":
                    shutil.rmtree(os.path.join(self.output_path, folder))

        if self.cache_checkpoints:
            # copy checkpoints to cache
            # overwrite the cache
            if os.path.exists(os.path.join(self.cache_dir, f"checkpoint_model_{self.step:06d}")):
                shutil.rmtree(os.path.join(self.cache_dir, f"checkpoint_model_{self.step:06d}"))

            shutil.copytree(
                os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}"),
                os.path.join(self.cache_dir, f"checkpoint_model_{self.step:06d}"),
            )

            checkpoints = sorted(
                [folder for folder in os.listdir(self.cache_dir) if folder.startswith("checkpoint_model")]
            )

            if len(checkpoints) > self.max_checkpoint:
                for folder in checkpoints[: -self.max_checkpoint]:
                    shutil.rmtree(os.path.join(self.cache_dir, folder))

        print("done saving")

    def train_one_step(self):
        self.model.train()

        accelerator = self.accelerator

        # Retrieve a batch of real images from the dataloader.
        real_dict = next(self.real_image_dataloader)

        # Extract the images from the dictionary and normalize them.
        # scaled from [0,1] to [-1,1].
        real_image = real_dict["images"] * 2.0 - 1.0
        real_label = self.eye_matrix[real_dict["class_labels"].astype(paddle.int64).squeeze(axis=1)]

        real_train_dict = {"real_image": real_image, "real_label": real_label}

        # Generate scaled noise based on the maximum sigma value.
        scaled_noise = (
            paddle.randn(
                (
                    self.batch_size,
                    3,
                    self.resolution,
                    self.resolution,
                )
                # device=accelerator.device
            )
            * self.conditioning_sigma
        )

        # Set timestep sigma to a preset value for all images in the batch.
        # timestep_sigma = paddle.ones(self.batch_size, device=accelerator.device) * self.conditioning_sigma
        timestep_sigma = paddle.ones(self.batch_size) * self.conditioning_sigma

        # For conditional generation, randomly generate labels.
        labels = paddle.randint(
            low=0,
            high=self.label_dim,
            shape=(self.batch_size,),
            # device=accelerator.device,
            dtype=paddle.int64,
        )  # .astype(paddle.float64)
        # Convert these labels to one-hot encoding.
        labels = self.eye_matrix[labels]

        COMPUTE_GENERATOR_GRADIENT = self.step % self.dfake_gen_update_ratio == 0

        # generate images and optionaly compute the generator gradient
        generator_loss_dict, generator_log_dict = self.model(
            scaled_noise,
            timestep_sigma,
            labels,
            real_train_dict=real_train_dict,
            compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
            generator_turn=True,
            guidance_turn=False,
        )

        # first update the generator if the current step is a multiple of dfake_gen_update_ratio
        generator_loss = 0.0

        if COMPUTE_GENERATOR_GRADIENT:
            generator_loss += generator_loss_dict["loss_dm"]

            if self.gan_classifier:
                generator_loss += generator_loss_dict["gen_cls_loss"] * self.gen_cls_loss_weight

            self.accelerator.backward(generator_loss)
            generator_grad_norm = accelerator.clip_grad_norm_(
                self.model.feedforward_model.parameters(), self.max_grad_norm
            )
            self.optimizer_generator.step()

            # if we also compute gan loss, the classifier also received gradient
            # zero out guidance model's gradient avoids undesired gradient accumulation
            self.optimizer_generator.zero_grad()
            self.optimizer_guidance.zero_grad()

        self.scheduler_generator.step()

        guidance_loss_dict, guidance_log_dict = self.model(
            scaled_noise,
            timestep_sigma,
            labels,
            real_train_dict=real_train_dict,
            compute_generator_gradient=False,
            generator_turn=False,
            guidance_turn=True,
            guidance_data_dict=generator_log_dict["guidance_data_dict"],
        )

        guidance_loss = 0

        guidance_loss += guidance_loss_dict["loss_fake_mean"]

        if self.gan_classifier:
            guidance_loss += guidance_loss_dict["guidance_cls_loss"] * self.cls_loss_weight

        self.accelerator.backward(guidance_loss)
        guidance_grad_norm = accelerator.clip_grad_norm_(self.model.guidance_model.parameters(), self.max_grad_norm)
        self.optimizer_guidance.step()
        self.optimizer_guidance.zero_grad()
        self.scheduler_guidance.step()
        self.optimizer_generator.zero_grad()

        # combine the two dictionaries
        loss_dict = {**generator_loss_dict, **guidance_loss_dict}
        log_dict = {**generator_log_dict, **guidance_log_dict}

        if self.step % self.wandb_iters == 0:
            log_dict["generated_image"] = accelerator.gather(log_dict["generated_image"])
            log_dict["dmtrain_grad"] = accelerator.gather(log_dict["dmtrain_grad"])
            log_dict["dmtrain_timesteps"] = accelerator.gather(log_dict["dmtrain_timesteps"])
            log_dict["dmtrain_pred_real_image"] = accelerator.gather(log_dict["dmtrain_pred_real_image"])
            log_dict["dmtrain_pred_fake_image"] = accelerator.gather(log_dict["dmtrain_pred_fake_image"])

        if accelerator.is_main_process and self.step % self.wandb_iters == 0:
            # TODO: Need more refactoring here
            with paddle.no_grad():
                generated_image = log_dict["generated_image"]
                generated_image_brightness = (generated_image * 0.5 + 0.5).clip(0, 1).mean()
                generated_image_std = (generated_image * 0.5 + 0.5).clip(0, 1).std()

                generated_image_grid = prepare_images_for_saving(generated_image, resolution=self.resolution)

                data_dict = {
                    "generated_image": wandb.Image(generated_image_grid),
                    "generated_image_brightness": generated_image_brightness.item(),
                    "generated_image_std": generated_image_std.item(),
                    "generator_grad_norm": generator_grad_norm.item(),
                    "guidance_grad_norm": guidance_grad_norm.item(),
                }

                (
                    dmtrain_noisy_latents,
                    dmtrain_pred_real_image,
                    dmtrain_pred_fake_image,
                    dmtrain_grad,
                    dmtrain_gradient_norm,
                ) = (
                    log_dict["dmtrain_noisy_latents"],
                    log_dict["dmtrain_pred_real_image"],
                    log_dict["dmtrain_pred_fake_image"],
                    log_dict["dmtrain_grad"],
                    log_dict["dmtrain_gradient_norm"],
                )

                gradient_brightness = dmtrain_grad.mean()
                gradient_std = dmtrain_grad.std(axis=[1, 2, 3]).mean()

                dmtrain_pred_real_image_mean = (dmtrain_pred_real_image * 0.5 + 0.5).clip(0, 1).mean()
                dmtrain_pred_fake_image_mean = (dmtrain_pred_fake_image * 0.5 + 0.5).clip(0, 1).mean()

                dmtrain_pred_read_image_std = (dmtrain_pred_real_image * 0.5 + 0.5).clip(0, 1).std()
                dmtrain_pred_fake_image_std = (dmtrain_pred_fake_image * 0.5 + 0.5).clip(0, 1).std()

                dmtrain_noisy_latents_grid = prepare_images_for_saving(
                    dmtrain_noisy_latents, resolution=self.resolution
                )
                dmtrain_pred_real_image_grid = prepare_images_for_saving(
                    dmtrain_pred_real_image, resolution=self.resolution
                )
                dmtrain_pred_fake_image_grid = prepare_images_for_saving(
                    dmtrain_pred_fake_image, resolution=self.resolution
                )

                gradient = dmtrain_grad
                gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
                gradient = (gradient - 0.5) / 0.5
                gradient = prepare_images_for_saving(gradient, resolution=self.resolution)

                gradient_scale_grid = draw_valued_array(
                    dmtrain_grad.abs().mean(axis=[1, 2, 3]).cpu().numpy(), output_dir=self.wandb_folder
                )

                difference_scale_grid = draw_valued_array(
                    (dmtrain_pred_real_image - dmtrain_pred_fake_image).abs().mean(axis=[1, 2, 3]).cpu().numpy(),
                    output_dir=self.wandb_folder,
                )

                difference = dmtrain_pred_fake_image - dmtrain_pred_real_image

                difference_brightness = difference.mean()

                difference = (difference - difference.min()) / (difference.max() - difference.min())
                difference = (difference - 0.5) / 0.5
                difference = prepare_images_for_saving(difference, resolution=self.resolution)

                dmtrain_timesteps_grid = draw_valued_array(
                    log_dict["dmtrain_timesteps"].squeeze().cpu().numpy(), output_dir=self.wandb_folder
                )

                data_dict.update(
                    {
                        "dmtrain_noisy_latents_grid": wandb.Image(dmtrain_noisy_latents_grid),
                        "dmtrain_pred_real_image_grid": wandb.Image(dmtrain_pred_real_image_grid),
                        "dmtrain_pred_fake_image_grid": wandb.Image(dmtrain_pred_fake_image_grid),
                        "loss_dm": loss_dict["loss_dm"].item(),
                        "loss_fake_mean": loss_dict["loss_fake_mean"].item(),
                        "dmtrain_gradient_norm": dmtrain_gradient_norm,
                        "gradient": wandb.Image(gradient),
                        "difference": wandb.Image(difference),
                        "gradient_scale_grid": wandb.Image(gradient_scale_grid),
                        "difference_norm_grid": wandb.Image(difference_scale_grid),
                        "dmtrain_timesteps_grid": wandb.Image(dmtrain_timesteps_grid),
                        "gradient_brightness": gradient_brightness.item(),
                        "difference_brightness": difference_brightness.item(),
                        "gradient_std": gradient_std.item(),
                        "dmtrain_pred_real_image_mean": dmtrain_pred_real_image_mean.item(),
                        "dmtrain_pred_fake_image_mean": dmtrain_pred_fake_image_mean.item(),
                        "dmtrain_pred_read_image_std": dmtrain_pred_read_image_std.item(),
                        "dmtrain_pred_fake_image_std": dmtrain_pred_fake_image_std.item(),
                    }
                )

                (faketrain_latents, faketrain_noisy_latents, faketrain_x0_pred) = (
                    log_dict["faketrain_latents"],
                    log_dict["faketrain_noisy_latents"],
                    log_dict["faketrain_x0_pred"],
                )

                faketrain_latents_grid = prepare_images_for_saving(faketrain_latents, resolution=self.resolution)
                faketrain_noisy_latents_grid = prepare_images_for_saving(
                    faketrain_noisy_latents, resolution=self.resolution
                )
                faketrain_x0_pred_grid = prepare_images_for_saving(faketrain_x0_pred, resolution=self.resolution)

                data_dict.update(
                    {
                        "faketrain_latents": wandb.Image(faketrain_latents_grid),
                        "faketrain_noisy_latents": wandb.Image(faketrain_noisy_latents_grid),
                        "faketrain_x0_pred": wandb.Image(faketrain_x0_pred_grid),
                    }
                )

                if self.gan_classifier:
                    data_dict["guidance_cls_loss"] = loss_dict["guidance_cls_loss"].item()
                    data_dict["gen_cls_loss"] = loss_dict["gen_cls_loss"].item()

                    pred_realism_on_fake = log_dict["pred_realism_on_fake"]
                    pred_realism_on_real = log_dict["pred_realism_on_real"]

                    hist_pred_realism_on_fake = draw_probability_histogram(pred_realism_on_fake.cpu().numpy())
                    hist_pred_realism_on_real = draw_probability_histogram(pred_realism_on_real.cpu().numpy())

                    data_dict.update(
                        {
                            "hist_pred_realism_on_fake": wandb.Image(hist_pred_realism_on_fake),
                            "hist_pred_realism_on_real": wandb.Image(hist_pred_realism_on_real),
                        }
                    )

                wandb.log(data_dict, step=self.step)

                log_str = "Step %s generator lr: %s guidance lr %s. " % (
                    self.step,
                    self.scheduler_generator.get_lr(),
                    self.scheduler_guidance.get_lr(),
                )

                for k, v in data_dict.items():
                    if "loss" in k:
                        log_str += k + ": {:.4}".format(v) + " "

                self.logger.info(log_str)

        self.accelerator.wait_for_everyone()

    def train(self):
        for index in range(self.step, self.train_iters):
            self.train_one_step()

            if self.accelerator.is_main_process:
                if (not self.no_save) and self.step % self.log_iters == 0:
                    self.save()

                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            self.accelerator.wait_for_everyone()
            self.step += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--train_iters", type=int, default=1000000)
    parser.add_argument("--log_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--initialie_generator", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_iters", type=int, default=100)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--label_dim", type=int, default=10)
    parser.add_argument("--warmup_step", type=int, default=500, help="warmup step for network")
    parser.add_argument("--min_step_percent", type=float, default=0.02, help="minimum step percent for training")
    parser.add_argument("--max_step_percent", type=float, default=0.98, help="maximum step percent for training")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--conditioning_sigma", type=float, default=80.0)

    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--dataset_name", type=str, default="imagenet")
    parser.add_argument("--ckpt_only_path", type=str, default=None, help="checkpoint (no optimizer state) only path")
    parser.add_argument("--delete_ckpts", action="store_true")
    parser.add_argument("--max_checkpoint", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--max_grad_norm", type=int, default=10)
    parser.add_argument("--real_image_path", type=str)
    parser.add_argument("--generator_lr", type=float)
    parser.add_argument("--guidance_lr", type=float)
    parser.add_argument("--dfake_gen_update_ratio", type=int, default=1)

    parser.add_argument("--cls_loss_weight", type=float, default=1.0)
    parser.add_argument("--gan_classifier", action="store_true")
    parser.add_argument("--gen_cls_loss_weight", type=float, default=0)
    parser.add_argument("--diffusion_gan", action="store_true")
    parser.add_argument("--diffusion_gan_max_timestep", type=int, default=0)

    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--generator_ckpt_path", type=str)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    assert (
        args.wandb_iters % args.dfake_gen_update_ratio == 0
    ), "wandb_iters should be a multiple of dfake_gen_update_ratio"

    return args


if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)
    trainer.train()
