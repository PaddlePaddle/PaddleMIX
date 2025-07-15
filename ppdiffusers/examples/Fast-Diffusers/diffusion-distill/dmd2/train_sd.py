# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# code is heavily based on https://github.com/tianweiy/DMD2

import gc

import matplotlib

matplotlib.use("Agg")
import argparse
import os
import shutil
import time

import paddle
import wandb
from paddle.distributed.sharding import group_sharded_parallel
from sd_image_dataset import SDImageDatasetLMDB
from sd_unified_model import SDUniModel
from utils import (
    SDTextDataset,
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
from ppdiffusers.transformers import AutoTokenizer, CLIPTokenizer


class Trainer:
    def __init__(self, args):
        self.args = args

        accelerator_project_config = ProjectConfiguration(logging_dir=args.log_path)
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="no",
            log_with="wandb",
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
            dispatch_batches=False,
        )
        set_seed(args.seed + accelerator.process_index)

        print(accelerator.state)

        if accelerator.is_main_process:
            output_path = os.path.join(args.output_path, f"time_{int(time.time())}_seed{args.seed}")
            os.makedirs(output_path, exist_ok=False)

            self.cache_dir = os.path.join(args.cache_dir, f"time_{int(time.time())}_seed{args.seed}")
            os.makedirs(self.cache_dir, exist_ok=False)

            self.output_path = output_path

            os.makedirs(args.log_path, exist_ok=True)

            run = wandb.init(
                config=args,
                dir=args.log_path,
                **{"mode": "offline", "entity": args.wandb_entity, "project": args.wandb_project},
            )
            wandb.run.log_code(".")
            wandb.run.name = args.wandb_name
            print(f"run dir: {run.dir}")
            self.wandb_folder = run.dir
            os.makedirs(self.wandb_folder, exist_ok=True)

        self.model = SDUniModel(args, accelerator)

        self.max_grad_norm = args.max_grad_norm
        self.denoising = args.denoising
        self.step = 0

        if self.denoising:
            assert args.sdxl, "denoising only supported for sdxl."

        if args.ckpt_only_path is not None:
            if accelerator.is_main_process:
                print(f"loading ckpt only from {args.ckpt_only_path}")
            generator_path = os.path.join(args.ckpt_only_path, "paddle_model.bin")
            guidance_path = os.path.join(args.ckpt_only_path, "paddle_model_1.bin")
            print(
                self.model.feedforward_model.load_state_dict(
                    paddle.load(generator_path, map_location="cpu"), strict=False
                )
            )
            print(
                self.model.guidance_model.load_state_dict(paddle.load(guidance_path, map_location="cpu"), strict=False)
            )

            self.step = int(args.ckpt_only_path.replace("/", "").split("_")[-1])

        if args.generator_ckpt_path is not None:
            if accelerator.is_main_process:
                print(f"loading generator ckpt from {args.generator_ckpt_path}")
            print(
                self.model.feedforward_model.load_state_dict(
                    paddle.load(args.generator_ckpt_path, map_location="cpu"), strict=True
                )
            )

        self.sdxl = args.sdxl

        if self.sdxl:
            tokenizer_one = AutoTokenizer.from_pretrained(
                args.model_id, subfolder="tokenizer", revision=args.revision, use_fast=False
            )

            tokenizer_two = AutoTokenizer.from_pretrained(
                args.model_id, subfolder="tokenizer_2", revision=args.revision, use_fast=False
            )

            dataset = SDTextDataset(
                args.train_prompt_path, is_sdxl=True, tokenizer_one=tokenizer_one, tokenizer_two=tokenizer_two
            )

            # also load the training dataset images, this will be useful for GAN loss
            real_dataset = SDImageDatasetLMDB(
                args.real_image_path, is_sdxl=True, tokenizer_one=tokenizer_one, tokenizer_two=tokenizer_two
            )
        else:
            tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
            uncond_input_ids = tokenizer(
                [""], max_length=tokenizer.model_max_length, return_tensors="pt", padding="max_length", truncation=True
            ).input_ids.to(accelerator.device)

            dataset = SDTextDataset(args.train_prompt_path, tokenizer, is_sdxl=False)
            self.uncond_embedding = self.model.text_encoder(uncond_input_ids)[0]

            # also load the training dataset images, this will be useful for GAN loss
            real_dataset = SDImageDatasetLMDB(args.real_image_path, is_sdxl=False, tokenizer_one=tokenizer)

        dataloader = paddle.io.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        dataloader = accelerator.prepare(dataloader)
        self.dataloader = cycle(dataloader)

        real_dataloader = paddle.io.DataLoader(
            real_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True
        )
        real_dataloader = accelerator.prepare(real_dataloader)
        self.real_dataloader = cycle(real_dataloader)

        # use two dataloader
        # as the generator and guidance model are trained at different paces
        guidance_dataloader = paddle.io.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        guidance_dataloader = accelerator.prepare(guidance_dataloader)
        self.guidance_dataloader = cycle(guidance_dataloader)

        self.guidance_cls_loss_weight = args.guidance_cls_loss_weight

        self.cls_on_clean_image = args.cls_on_clean_image
        self.gen_cls_loss = args.gen_cls_loss
        self.gen_cls_loss_weight = args.gen_cls_loss_weight
        self.previous_time = None

        if self.denoising:
            denoising_dataloader = paddle.io.DataLoader(
                real_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True, drop_last=True
            )
            denoising_dataloader = accelerator.prepare(denoising_dataloader)
            self.denoising_dataloader = cycle(denoising_dataloader)

        self.gsp = args.gsp

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

        if self.gsp:
            (self.scheduler_generator, self.scheduler_guidance) = accelerator.prepare(
                self.scheduler_generator, self.scheduler_guidance
            )

            self.model.feedforward_model, self.optimizer_generator, _ = group_sharded_parallel(
                self.model.feedforward_model,
                self.optimizer_generator,
                "os_g",
                scaler=None,
                exclude_layer=["GroupNorm"],
            )
            self.model.guidance_model, self.optimizer_guidance, _ = group_sharded_parallel(
                self.model.guidance_model, self.optimizer_guidance, "os_g", scaler=None, exclude_layer=["GroupNorm"]
            )
            self.model.feedforward_model._auto_refresh_trainable = False
            self.model.guidance_model._auto_refresh_trainable = False
        else:
            # the self.model is not wrapped in ddp, only its two subnetworks are wrapped
            (
                self.model.feedforward_model,
                self.model.guidance_model,
                self.optimizer_generator,
                self.optimizer_guidance,
                self.scheduler_generator,
                self.scheduler_guidance,
            ) = accelerator.prepare(
                self.model.feedforward_model,
                self.model.guidance_model,
                self.optimizer_generator,
                self.optimizer_guidance,
                self.scheduler_generator,
                self.scheduler_guidance,
            )

        self.accelerator = accelerator
        self.train_iters = args.train_iters
        self.batch_size = args.batch_size
        self.resolution = args.resolution
        self.log_iters = args.log_iters
        self.wandb_iters = args.wandb_iters
        self.latent_resolution = args.latent_resolution
        self.grid_size = args.grid_size
        self.log_loss = args.log_loss
        self.latent_channel = args.latent_channel

        self.no_save = args.no_save
        self.max_checkpoint = args.max_checkpoint

        self.dfake_gen_update_ratio = args.dfake_gen_update_ratio

        if args.checkpoint_path is not None:
            self.load(args.checkpoint_path)

    def load(self, checkpoint_path):
        # this is used for non-gsp models.
        self.step = int(checkpoint_path.replace("/", "").split("_")[-1])
        print(self.accelerator.load_state(checkpoint_path, strict=False))
        self.accelerator.print(f"Loaded checkpoint from {checkpoint_path}")

    def save(self):
        # NOTE: we save the checkpoints to two places
        # 1. output_path: save the latest one, this is assumed to be a permanent storage
        # 2. cache_dir: save all checkpoints, this is assumed to be a temporary storage
        if self.accelerator.is_main_process:
            output_path = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(output_path, exist_ok=True)
            print(f"start saving checkpoint to {output_path}")

            if self.gsp:
                paddle.distributed.sharding.save_group_sharded_model(self.model.feedforward_model, output_path)
            else:
                self.accelerator.save_state(output_path)

            # remove previous checkpoints
            for folder in os.listdir(self.output_path):
                if folder.startswith("checkpoint_model") and folder != f"checkpoint_model_{self.step:06d}":
                    shutil.rmtree(os.path.join(self.output_path, folder))

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
        paddle.device.cuda.empty_cache()
        gc.collect()

    def train_one_step(self):
        self.model.train()

        accelerator = self.accelerator

        # 4 channel for SD-VAE, please adapt for other autoencoders
        noise = paddle.randn(
            [self.batch_size, self.latent_channel, self.latent_resolution, self.latent_resolution],
        )
        visual = self.step % self.wandb_iters == 0

        COMPUTE_GENERATOR_GRADIENT = self.step % self.dfake_gen_update_ratio == 0

        if COMPUTE_GENERATOR_GRADIENT:
            text_embedding = next(self.dataloader)
        else:
            text_embedding = next(self.guidance_dataloader)

        if self.sdxl:
            # SDXL uses zero as the uncond_embedding
            uncond_embedding = None
        else:
            text_embedding = text_embedding["text_input_ids_one"].squeeze(1)  # actually it is tokenized text prompts
            uncond_embedding = self.uncond_embedding.repeat(len(text_embedding), 1, 1)

        if self.denoising:
            denoising_dict = next(self.denoising_dataloader)
        else:
            denoising_dict = None

        if self.cls_on_clean_image:
            real_train_dict = next(self.real_dataloader)
            if real_train_dict["images"].shape[-1] != args.latent_resolution:
                real_train_dict["images"] = paddle.nn.functional.interpolate(
                    real_train_dict["images"], (args.latent_resolution, args.latent_resolution), mode="bilinear"
                )
        else:
            real_train_dict = None

        # generate images and optionaly compute the generator gradient
        generator_loss_dict, generator_log_dict = self.model(
            noise,
            text_embedding,
            uncond_embedding,
            visual=visual,
            denoising_dict=denoising_dict,
            compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
            real_train_dict=real_train_dict,
            generator_turn=True,
            guidance_turn=False,
        )

        # first update the generator if the current step is a multiple of dfake_gen_update_ratio
        generator_loss = 0.0

        if COMPUTE_GENERATOR_GRADIENT:
            if not self.args.gan_alone:
                generator_loss += generator_loss_dict["loss_dm"] * self.args.dm_loss_weight

            if self.cls_on_clean_image and self.gen_cls_loss:
                generator_loss += generator_loss_dict["gen_cls_loss"] * self.gen_cls_loss_weight

            self.accelerator.backward(generator_loss)
            generator_grad_norm = accelerator.clip_grad_norm_(
                self.model.feedforward_model.parameters(), self.max_grad_norm
            )
            # generator_grad_norm = paddle.zeros([1])
            self.optimizer_generator.step()
            # if we also compute gan loss, the classifier may also receive gradient
            # zero out guidance model's gradient avoids undesired gradient accumulation
            self.optimizer_generator.clear_grad()
            self.optimizer_guidance.clear_grad()

        self.scheduler_generator.step()

        # update the guidance model (dfake and classifier)
        # with self.model.guidance_model.no_sync():
        guidance_loss_dict, guidance_log_dict = self.model(
            noise,
            text_embedding,
            uncond_embedding,
            visual=visual,
            denoising_dict=denoising_dict,
            real_train_dict=real_train_dict,
            compute_generator_gradient=COMPUTE_GENERATOR_GRADIENT,
            generator_turn=False,
            guidance_turn=True,
            guidance_data_dict=generator_log_dict["guidance_data_dict"],
        )

        guidance_loss = 0

        guidance_loss += guidance_loss_dict["loss_fake_mean"]

        if self.cls_on_clean_image:
            guidance_loss += guidance_loss_dict["guidance_cls_loss"] * self.guidance_cls_loss_weight

        self.accelerator.backward(guidance_loss)
        # fused_allreduce_gradients(list(self.model.guidance_model.parameters()), None)
        guidance_grad_norm = accelerator.clip_grad_norm_(self.model.guidance_model.parameters(), self.max_grad_norm)
        # guidance_grad_norm = paddle.zeros([1])
        self.optimizer_guidance.step()
        self.optimizer_guidance.clear_grad()
        # zero out the generator's gradient as well
        self.optimizer_generator.clear_grad()

        self.scheduler_guidance.step()

        # combine the two dictionaries
        loss_dict = {**generator_loss_dict, **guidance_loss_dict}
        log_dict = {**generator_log_dict, **guidance_log_dict}

        generated_image_mean = log_dict["guidance_data_dict"]["image"].mean()
        generated_image_std = log_dict["guidance_data_dict"]["image"].std()

        generated_image_mean = accelerator.gather(generated_image_mean).mean()
        generated_image_std = accelerator.gather(generated_image_std).mean()

        if COMPUTE_GENERATOR_GRADIENT:
            if not self.args.gan_alone:
                dmtrain_pred_real_image_mean = log_dict["dmtrain_pred_real_image"].mean()
                dmtrain_pred_real_iamge_std = log_dict["dmtrain_pred_real_image"].std()

                dmtrain_pred_real_image_mean = accelerator.gather(dmtrain_pred_real_image_mean).mean()
                dmtrain_pred_real_iamge_std = accelerator.gather(dmtrain_pred_real_iamge_std).mean()

                dmtrain_pred_fake_image_mean = log_dict["dmtrain_pred_fake_image"].mean()
                dmtrain_pred_fake_image_std = log_dict["dmtrain_pred_fake_image"].std()

                dmtrain_pred_fake_image_mean = accelerator.gather(dmtrain_pred_fake_image_mean).mean()
                dmtrain_pred_fake_image_std = accelerator.gather(dmtrain_pred_fake_image_std).mean()

        if self.denoising:
            original_image_mean = denoising_dict["images"].mean()
            original_image_std = denoising_dict["images"].std()

            original_image_mean = accelerator.gather(original_image_mean).mean()
            original_image_std = accelerator.gather(original_image_std).mean()

        if accelerator.is_main_process and self.log_loss and (not visual):
            wandb_loss_dict = {
                "loss_fake_mean": guidance_loss_dict["loss_fake_mean"].item(),
                "guidance_grad_norm": guidance_grad_norm.item(),
                "generated_image_mean": generated_image_mean.item(),
                "generated_image_std": generated_image_std.item(),
                "batch_size": len(noise),
            }

            if COMPUTE_GENERATOR_GRADIENT and (not self.args.gan_alone):
                wandb_loss_dict.update(
                    {
                        "dmtrain_pred_real_image_mean": dmtrain_pred_real_image_mean.item(),
                        "dmtrain_pred_real_image_std": dmtrain_pred_real_iamge_std.item(),
                        "dmtrain_pred_fake_image_mean": dmtrain_pred_fake_image_mean.item(),
                        "dmtrain_pred_fake_image_std": dmtrain_pred_fake_image_std.item(),
                    }
                )

            if self.denoising:
                wandb_loss_dict.update(
                    {
                        "original_image_mean": original_image_mean.item(),
                        "original_image_std": original_image_std.item(),
                    }
                )

            if COMPUTE_GENERATOR_GRADIENT:
                wandb_loss_dict["generator_grad_norm"] = generator_grad_norm.item()

                if not self.args.gan_alone:
                    wandb_loss_dict["loss_dm"] = loss_dict["loss_dm"].item()
                    wandb_loss_dict["dmtrain_gradient_norm"] = log_dict["dmtrain_gradient_norm"]

                if self.gen_cls_loss:
                    wandb_loss_dict.update({"gen_cls_loss": loss_dict["gen_cls_loss"].item()})

            if self.cls_on_clean_image:
                wandb_loss_dict.update({"guidance_cls_loss": loss_dict["guidance_cls_loss"].item()})

            wandb.log(wandb_loss_dict, step=self.step)

        if visual:
            if not self.args.gan_alone:
                log_dict["dmtrain_pred_real_image_decoded"] = accelerator.gather(
                    log_dict["dmtrain_pred_real_image_decoded"]
                )
                log_dict["dmtrain_pred_fake_image_decoded"] = accelerator.gather(
                    log_dict["dmtrain_pred_fake_image_decoded"]
                )

            log_dict["generated_image"] = accelerator.gather(log_dict["generated_image"])

            if self.denoising:
                log_dict["original_clean_image"] = accelerator.gather(log_dict["original_clean_image"])
                log_dict["denoising_timestep"] = accelerator.gather(log_dict["denoising_timestep"])

        if accelerator.is_main_process and visual:
            with paddle.no_grad():
                if not self.args.gan_alone:
                    (dmtrain_pred_real_image, dmtrain_pred_fake_image, dmtrain_gradient_norm) = (
                        log_dict["dmtrain_pred_real_image_decoded"],
                        log_dict["dmtrain_pred_fake_image_decoded"],
                        log_dict["dmtrain_gradient_norm"],
                    )

                    dmtrain_pred_real_image_grid = prepare_images_for_saving(
                        dmtrain_pred_real_image, resolution=self.resolution, grid_size=self.grid_size
                    )
                    dmtrain_pred_fake_image_grid = prepare_images_for_saving(
                        dmtrain_pred_fake_image, resolution=self.resolution, grid_size=self.grid_size
                    )

                    difference_scale_grid = draw_valued_array(
                        (dmtrain_pred_real_image - dmtrain_pred_fake_image).abs().mean(axis=[1, 2, 3]).cpu().numpy(),
                        output_dir=self.wandb_folder,
                        grid_size=self.grid_size,
                    )

                    difference = dmtrain_pred_real_image - dmtrain_pred_fake_image
                    difference = (difference - difference.min()) / (difference.max() - difference.min())
                    difference = (difference - 0.5) / 0.5
                    difference = prepare_images_for_saving(
                        difference, resolution=self.resolution, grid_size=self.grid_size
                    )

                    data_dict = {
                        "dmtrain_pred_real_image": wandb.Image(dmtrain_pred_real_image_grid),
                        "dmtrain_pred_fake_image": wandb.Image(dmtrain_pred_fake_image_grid),
                        "loss_dm": loss_dict["loss_dm"].item(),
                        "dmtrain_gradient_norm": dmtrain_gradient_norm,
                        "difference": wandb.Image(difference),
                        "difference_norm_grid": wandb.Image(difference_scale_grid),
                    }
                else:
                    data_dict = {}

                generated_image = log_dict["generated_image"]
                generated_image_grid = prepare_images_for_saving(
                    generated_image, resolution=self.resolution, grid_size=self.grid_size
                )

                generated_image_mean = generated_image.mean()
                generated_image_std = generated_image.std()

                data_dict.update(
                    {
                        "generated_image": wandb.Image(generated_image_grid),
                        "loss_fake_mean": loss_dict["loss_fake_mean"].item(),
                        "generator_grad_norm": generator_grad_norm.item(),
                        "guidance_grad_norm": guidance_grad_norm.item(),
                    }
                )

                if self.denoising:
                    origianl_clean_image = log_dict["original_clean_image"]
                    origianl_clean_image_grid = prepare_images_for_saving(
                        origianl_clean_image, resolution=self.resolution, grid_size=self.grid_size
                    )

                    denoising_timestep = log_dict["denoising_timestep"]
                    denoising_timestep_grid = draw_valued_array(
                        denoising_timestep.cpu().numpy(), output_dir=self.wandb_folder, grid_size=self.grid_size
                    )

                    data_dict.update(
                        {
                            "original_clean_image": wandb.Image(origianl_clean_image_grid),
                            "original_image_mean": original_image_mean.item(),
                            "original_image_std": original_image_std.item(),
                            "denoising_timestep": wandb.Image(denoising_timestep_grid),
                        }
                    )

                if self.cls_on_clean_image:
                    data_dict["guidance_cls_loss"] = loss_dict["guidance_cls_loss"].item()

                    if self.gen_cls_loss:
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

                print(log_str)

        self.accelerator.wait_for_everyone()

    def train(self):
        for index in range(self.step, self.train_iters):
            self.train_one_step()
            if (not self.no_save) and self.step % self.log_iters == 0:
                self.save()

            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            self.step += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--output_path", type=str, default="/mnt/localssd/test_stable_diffusion_coco")
    parser.add_argument("--log_path", type=str, default="/mnt/localssd/log_stable_diffusion_coco")
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
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="max grad norm for network")
    parser.add_argument("--warmup_step", type=int, default=500, help="warmup step for network")
    parser.add_argument("--min_step_percent", type=float, default=0.02, help="minimum step percent for training")
    parser.add_argument("--max_step_percent", type=float, default=0.98, help="maximum step percent for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--ckpt_only_path", type=str, default=None, help="checkpoint (no optimizer state) only path")
    parser.add_argument("--train_prompt_path", type=str)
    parser.add_argument("--latent_resolution", type=int, default=64)
    parser.add_argument("--real_guidance_scale", type=float, default=6.0)
    parser.add_argument("--fake_guidance_scale", type=float, default=1.0)
    parser.add_argument("--grid_size", type=int, default=2)
    parser.add_argument("--no_save", action="store_true", help="don't save ckpt for debugging only")
    parser.add_argument("--cache_dir", type=str, default="/mnt/localssd/cache")
    parser.add_argument("--log_loss", action="store_true", help="log loss at every iteration")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--latent_channel", type=int, default=4)
    parser.add_argument("--max_checkpoint", type=int, default=150)
    parser.add_argument("--dfake_gen_update_ratio", type=int, default=1)
    parser.add_argument("--generator_lr", type=float)
    parser.add_argument("--guidance_lr", type=float)
    parser.add_argument("--cls_on_clean_image", action="store_true")
    parser.add_argument("--gen_cls_loss", action="store_true")
    parser.add_argument("--gen_cls_loss_weight", type=float, default=0)
    parser.add_argument("--guidance_cls_loss_weight", type=float, default=0)
    parser.add_argument("--sdxl", action="store_true")
    parser.add_argument("--gsp", action="store_true")
    parser.add_argument("--generator_ckpt_path", type=str)
    parser.add_argument("--conditioning_timestep", type=int, default=999)
    parser.add_argument("--tiny_vae", action="store_true")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="apply gradient checkpointing for dfake and generator. this might be a better option than gsp",
    )
    parser.add_argument("--dm_loss_weight", type=float, default=1.0)

    parser.add_argument("--denoising", action="store_true", help="train the generator for denoising")
    parser.add_argument("--denoising_timestep", type=int, default=1000)
    parser.add_argument("--num_denoising_step", type=int, default=1)
    parser.add_argument("--denoising_loss_weight", type=float, default=1.0)

    parser.add_argument("--diffusion_gan", action="store_true")
    parser.add_argument("--diffusion_gan_max_timestep", type=int, default=0)
    parser.add_argument("--revision", type=str)

    parser.add_argument("--real_image_path", type=str)
    parser.add_argument("--gan_alone", action="store_true", help="only use the gan loss without dmd")
    parser.add_argument("--backward_simulation", action="store_true")

    parser.add_argument("--generator_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    assert args.gradient_accumulation_steps == 1, "grad accumulation not supported yet"

    assert (
        args.wandb_iters % args.dfake_gen_update_ratio == 0
    ), "wandb_iters should be a multiple of dfake_gen_update_ratio"

    return args


if __name__ == "__main__":
    args = parse_args()
    if (
        paddle.distributed.get_world_size() > 1
        and not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized()
    ):
        from paddle.distributed import fleet

        strategy = fleet.DistributedStrategy()
        order = ["dp", "sharding", "pp", "sep", "mp"]
        hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": paddle.distributed.get_world_size(),
        }

        strategy.hybrid_configs = hybrid_configs
        strategy.hybrid_configs["sharding_configs"].release_gradients = True
        fleet.init(is_collective=True, strategy=strategy)
    trainer = Trainer(args)

    trainer.train()
