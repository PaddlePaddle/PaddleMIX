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

import os
import time

import matplotlib

matplotlib.use("Agg")
import paddle
from paddlenlp.trainer import PdArgumentParser, TrainingArguments
from sd_image_dataset import SDImageDatasetLMDB
from sd_unified_model import SDUniModel
from sdxl.sdxl_trainer import DMD2Trainer
from sdxl.trainer_args import ModelArguments
from utils import SDTextDataset, cycle

from ppdiffusers.accelerate import Accelerator
from ppdiffusers.accelerate.utils import ProjectConfiguration, set_seed
from ppdiffusers.optimization import get_scheduler
from ppdiffusers.transformers import AutoTokenizer, CLIPTokenizer


class Trainer:
    def __init__(self, args, training_args):
        self.args = args

        accelerator_project_config = ProjectConfiguration(logging_dir=args.log_path)
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision="no",
            log_with="wandb",
            project_config=accelerator_project_config,
            kwargs_handlers=None,
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
                self.model.feedforward_model.set_state_dict(
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

        self._dmd2trainer = DMD2Trainer(
            model=self.model.feedforward_model,
            guidance_model=self.model.guidance_model,
            args=training_args,
            optimizers=(None, self.scheduler_generator),
            optimizers_guidance=(None, self.scheduler_guidance),
            dmd2_args=args,
            unified_model=self.model,
            accelerator=accelerator,
        )
        self._dmd2trainer.train_dataloader = self.dataloader
        self._dmd2trainer.guidance_dataloader = self.guidance_dataloader
        self._dmd2trainer.real_dataloader = self.real_dataloader
        self._dmd2trainer.denoising_dataloader = self.denoising_dataloader

    def load(self, checkpoint_path):
        # this is used for non-fsdp models.
        self.step = int(checkpoint_path.replace("/", "").split("_")[-1])
        print(self.accelerator.load_state(checkpoint_path, strict=False))
        self.accelerator.print(f"Loaded checkpoint from {checkpoint_path}")

    def train(self):
        self._dmd2trainer.train()


if __name__ == "__main__":
    parser = PdArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    model_args.gradient_accumulation_steps = training_args.gradient_accumulation_steps
    model_args.max_grad_norm = training_args.max_grad_norm
    model_args.seed = training_args.seed
    model_args.local_rank = training_args.local_rank
    model_args.output_path = training_args.output_dir
    training_args.max_steps = model_args.train_iters
    trainer = Trainer(model_args, training_args)

    trainer.train()
