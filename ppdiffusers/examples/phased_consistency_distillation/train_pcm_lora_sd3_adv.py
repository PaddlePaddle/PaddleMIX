#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import argparse
import copy
import gc
import itertools
import logging
import math
import os
os.environ["USE_PEFT_BACKEND"] = "True"
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from PIL.ImageOps import exif_transpose
from paddle.vision import transforms
from tqdm.auto import tqdm
from discriminator_sd3 import Discriminator

import paddle
import ppdiffusers
from ppdiffusers.accelerate import Accelerator
from ppdiffusers.accelerate.logging import get_logger
from ppdiffusers.accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from ppdiffusers.peft import LoraConfig, set_peft_model_state_dict
from ppdiffusers.peft.utils import get_peft_model_state_dict
from paddlenlp.transformers import PretrainedConfig
from ppdiffusers.transformers import CLIPTokenizer, T5Tokenizer
from ppdiffusers.accelerate import Accelerator
from ppdiffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available
)
from ppdiffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from ppdiffusers.optimization import get_scheduler
from pcm_fm_deterministic_scheduler import PCMFMDeterministicScheduler

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.30.0.dev0")

logger = get_logger(__name__)


def load_text_encoders(class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="text_encoder_3",
        revision=args.revision,
        variant=args.variant,
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    step,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt: {pipeline_args['prompt']}"
    )
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = (
        paddle.Generator().manual_seed(args.seed)
        if args.seed
        else None
    )
    autocast_ctx = nullcontext()

    with autocast_ctx:
        images = [
            pipeline(**pipeline_args, generator=generator).images[0]
            for _ in range(args.num_validation_images)
        ]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, step, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {pipeline_args['prompt']}")
                        for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    if paddle.device.cuda.device_count() >= 1:
        paddle.device.cuda.empty_cache()

    return images


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.take_along_axis(axis=-1, indices=t, broadcast=False)
    return out.reshape([b, *((1,) * (len(x_shape) - 1))])


class EulerSolver:
    def __init__(self, sigmas, timesteps=1000, euler_timesteps=50):
        self.step_ratio = timesteps // euler_timesteps
        self.euler_timesteps = (
            np.arange(1, euler_timesteps + 1) * self.step_ratio
        ).round().astype(np.int64) - 1
        self.euler_timesteps_prev = np.asarray([0] + self.euler_timesteps[:-1].tolist())
        self.sigmas = sigmas[self.euler_timesteps]
        self.sigmas_prev = np.asarray(
            [sigmas[0]] + sigmas[self.euler_timesteps[:-1]].tolist()
        )  # either use sigma0 or 0

        self.euler_timesteps = paddle.to_tensor(self.euler_timesteps).astype(dtype="int64")
        self.euler_timesteps_prev = paddle.to_tensor(self.euler_timesteps_prev).astype(dtype="int64")
        self.sigmas = paddle.to_tensor(self.sigmas)
        self.sigmas_prev = paddle.to_tensor(self.sigmas_prev)

    def to(self, device):
        self.euler_timesteps = self.euler_timesteps.to(device)
        self.euler_timesteps_prev = self.euler_timesteps_prev.to(device)

        self.sigmas = self.sigmas.to(device)
        self.sigmas_prev = self.sigmas_prev.to(device)
        return self

    def euler_step(self, sample, model_pred, timestep_index):
        sigma = extract_into_tensor(self.sigmas, timestep_index, model_pred.shape)
        sigma_prev = extract_into_tensor(
            self.sigmas_prev, timestep_index, model_pred.shape
        )
        x_prev = sample + (sigma_prev - sigma) * model_pred
        return x_prev

    def euler_style_multiphase_pred(
        self,
        sample,
        model_pred,
        timestep_index,
        multiphase,
        is_target=False,
    ):

        inference_indices = np.linspace(
            0, len(self.euler_timesteps), num=multiphase, endpoint=False
        )
        inference_indices = np.floor(inference_indices).astype(np.int64)
        inference_indices = (
            paddle.to_tensor(inference_indices).astype(dtype="int64")
        )
        expanded_timestep_index = timestep_index.unsqueeze(1).expand(
            [-1, inference_indices.shape[0]]
        )
        valid_indices_mask = expanded_timestep_index >= inference_indices
        last_valid_index = valid_indices_mask.flip(axis=[1]).astype(dtype="int64").argmax(axis=1)
        last_valid_index = inference_indices.shape[0] - 1 - last_valid_index
        timestep_index_end = inference_indices[last_valid_index]

        if is_target:
            sigma = extract_into_tensor(self.sigmas_prev, timestep_index, sample.shape)
        else:
            sigma = extract_into_tensor(self.sigmas, timestep_index, sample.shape)
        sigma_prev = extract_into_tensor(
            self.sigmas_prev, timestep_index_end, sample.shape
        )
        x_prev = sample + (sigma_prev - sigma) * model_pred

        return x_prev, timestep_index_end


def import_model_class_from_model_name_or_path(
    pretrained_teacher_model: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_teacher_model, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from ppdiffusers.transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from ppdiffusers.transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to train and val data",
    )
    parser.add_argument(
        "--pretrained_teacher_model",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )

    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X steps. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd3-dreambooth",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="sigma_sqrt",
        choices=["sigma_sqrt", "logit_normal", "mode"],
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )
    parser.add_argument(
        "--w_min",
        type=float,
        default=5.0,
        required=False,
        help=(
            "The minimum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--w_max",
        type=float,
        default=15.0,
        required=False,
        help=(
            "The maximum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        ),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l2", "huber"],
        help="The type of loss to use for the LCD loss.",
    )
    parser.add_argument(
        "--huber_c",
        type=float,
        default=0.001,
        help="The huber loss parameter. Only used if `--loss_type=huber`.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_decouple",
        type=bool,
        default=True,
        help="Use AdamW style decoupled weight decay",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for unet params",
    )
    parser.add_argument(
        "--adam_weight_decay_text_encoder",
        type=float,
        default=1e-03,
        help="Weight decay to use for text_encoder",
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument("--num_euler_timesteps", type=int, default=50)
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument("--not_apply_cfg_solver", action="store_true")
    parser.add_argument("--multiphase", default=8, type=int)
    parser.add_argument("--adv_weight", default=0.1, type=float)
    parser.add_argument("--adv_lr", default=1e-5, type=float)
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class CustomImageDataset(paddle.io.Dataset):
    def __init__(self, img_dir, sample_size):
        """
        Args:
            img_dir (string): Directory with all the images and text files.
            sample_size (tuple): Desired sample size as (height, width).
        """
        self.img_dir = img_dir
        self.sample_size = sample_size
        self.img_names = [
            f for f in os.listdir(img_dir) if f.endswith((".png", ".jpg"))
        ]
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.sample_size, interpolation="lanczos"
                ),
                transforms.CenterCrop(self.sample_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):

        while True:
            try:
                img_name = self.img_names[idx]
                img_path = os.path.join(self.img_dir, img_name)
                image = Image.open(img_path).convert("RGB")
                image = self.transform(image)
                text_name = img_name.rsplit(".", 1)[0] + ".txt"
                text_path = os.path.join(self.img_dir, text_name)
                with open(text_path, "r") as f:
                    text = f.read().strip()
            except:
                idx = random.randint(0, len(self.img_names))
                continue

            return {"pixel_values": image, "prompts": text}


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pd",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    prompt=None,
    num_images_per_prompt=1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pd",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids)[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.tile(repeat_times=[1, num_images_per_prompt, 1])
    prompt_embeds = prompt_embeds.view([batch_size * num_images_per_prompt, seq_len, -1])

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pd",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.tile(repeat_times=[1, num_images_per_prompt, 1])
    prompt_embeds = prompt_embeds.view([batch_size * num_images_per_prompt, seq_len, -1])

    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = paddle.concat(x=clip_prompt_embeds_list, axis=-1)
    pooled_prompt_embeds = paddle.concat(x=clip_pooled_prompt_embeds_list, axis=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
    )

    clip_prompt_embeds = paddle.nn.functional.pad(
        clip_prompt_embeds,
        (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
        pad_from_left_axis=False,
    )
    prompt_embeds = paddle.concat(x=[clip_prompt_embeds, t5_prompt_embed], axis=-2)

    return prompt_embeds, pooled_prompt_embeds


def cast_training_params(model: Union[paddle.nn.Layer, List[paddle.nn.Layer]], dtype=paddle.float32):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if not param.stop_gradient:
                param.set_value(param.to(dtype=dtype))

def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    # if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
    #     # due to pytorch#99272, MPS does not yet support bfloat16.
    #     raise ValueError(
    #         "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
    #     )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    # if torch.backends.mps.is_available():
    #     accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        ppdiffusers.utils.logging.set_verbosity_warning()
        ppdiffusers.utils.logging.set_verbosity_info()
    else:
        ppdiffusers.utils.logging.set_verbosity_error()
        ppdiffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizers
    print(args.pretrained_teacher_model)
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5Tokenizer.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_teacher_model, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_teacher_model, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_teacher_model, args.revision, subfolder="text_encoder_3"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_teacher_model, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )
    teacher_transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )
    discriminator = Discriminator(teacher_transformer)
    discriminator_params = []
    for param in discriminator.heads.parameters():
        param.requires_grad = True
        discriminator_params.append(param)

    transformer.requires_grad_(False)
    teacher_transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = "float32"
    if accelerator.mixed_precision == "fp16":
        weight_dtype = "float16"
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = "bfloat16"

    # if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
    #     # due to pytorch#99272, MPS does not yet support bfloat16.
    #     raise ValueError(
    #         "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
    #     )

    vae.to(dtype=paddle.float32)
    transformer.to(dtype=weight_dtype)
    # teacher_transformer.to(dtype=weight_dtype)
    text_encoder_one.to(dtype=weight_dtype)
    text_encoder_two.to(dtype=weight_dtype)
    text_encoder_three.to(dtype=weight_dtype)
    solver = EulerSolver(
        noise_scheduler.sigmas.numpy()[::-1],
        noise_scheduler.config.num_train_timesteps,
        euler_timesteps=args.num_euler_timesteps,
    )

    # not available yet
    # transformer.enable_xformers_memory_efficient_attention()
    # teacher_transformer.enable_xformers_memory_efficient_attention()
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        teacher_transformer.enable_gradient_checkpointing()

    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.lora_rank,
        # lora_alpha=args.lora_rank,
        # init_lora_weights="gaussian",
        # target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        target_modules=[
            "to_k",
            "to_q",
            "to_v",
            ".add_q_proj",
            ".add_k_proj",
            ".add_v_proj",
            "to_add_out",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
            "norm1.linear",
            "norm1_context.linear",
            "context_embedder",
            "text_embedder.linear_1",
            "text_embedder.linear_2",
            "timestep_embedder.linear_1",
            "timestep_embedder.linear_2",
            "pos_embed.proj",
        ],
    )
    transformer.add_adapter(transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if ppdiffusers.utils.paddle_utils.is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    pass  # just hacked design
                    # raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            StableDiffusion3Pipeline.save_lora_weights(
                output_dir, transformer_lora_layers=transformer_lora_layers_to_save
            )
            paddle.save(accelerator.unwrap_model(discriminator).heads.state_dict(), os.path.join(output_dir,"heads.pdparams"))

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                pass  # just hacked design
                # raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v
            for k, v in lora_state_dict[0].items()
            if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(
            transformer_, transformer_state_dict, adapter_name="default"
        )
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

        accelerator.unwrap_model(discriminator).load_state_dict(paddle.load(os.path.join(output_dir,"heads.pdparams")))

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # if args.allow_tf32 and torch.cuda.is_available():
    #     torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=paddle.float32)

    transformer_lora_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters())
    )

    # Optimization parameters
    transformer_parameters_with_lr = {
        "params": transformer_lora_parameters,
        # "lr": args.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = paddle.optimizer.AdamW

        optimizer = optimizer_class(
            parameters=params_to_optimize,
            learning_rate=args.learning_rate,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            weight_decay=args.adam_weight_decay,
            epsilon=args.adam_epsilon,
        )
        adv_optimizer = optimizer_class(
            parameters=discriminator_params,
            learning_rate=args.adv_lr,
            beta1=0,
            beta2=0.999,
            weight_decay=args.adam_weight_decay,
            epsilon=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError(
                "To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`"
            )

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            parameters=params_to_optimize,
            lr=args.learning_rate,
            beta1=args.adam_beta1,
            beta2=args.adam_beta2,
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            epsilon=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation:
    print("start training")
    train_dataset = CustomImageDataset(args.data_path, args.resolution)
    train_dataloader = paddle.io.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with paddle.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, prompt
            )
        return prompt_embeds, pooled_prompt_embeds

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        learning_rate=args.learning_rate,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    (
        discriminator,
        adv_optimizer,
        transformer,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        discriminator,
        adv_optimizer,
        transformer,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    uncond_prompt_embeds, uncond_pooled_prompt_embeds = compute_text_embeddings(
        [""] * args.train_batch_size, text_encoders, tokenizers
    )
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer, discriminator]
            with accelerator.accumulate(models_to_accumulate):
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                prompts = batch["prompts"]

                prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                    prompts, text_encoders, tokenizers
                )

                # Convert images to latent space
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = model_input * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = paddle.randn(shape=model_input.shape, dtype=model_input.dtype)
                bsz = model_input.shape[0]

                index = paddle.randint(
                    0, args.num_euler_timesteps, (bsz,)
                ).astype(dtype="int64")

                # Add noise according to flow matching.
                # sigmas = get_sigmas(start_timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                sigmas = extract_into_tensor(solver.sigmas, index, model_input.shape)
                sigmas_prev = extract_into_tensor(
                    solver.sigmas_prev, index, model_input.shape
                )
                timesteps = (
                    sigmas * noise_scheduler.config.num_train_timesteps
                ).squeeze([1, 2, 3])
                timesteps_prev = (
                    sigmas_prev * noise_scheduler.config.num_train_timesteps
                ).squeeze([1, 2, 3])
                noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                # if accelerator.is_main_process:
                model_pred, end_index = solver.euler_style_multiphase_pred(
                    noisy_model_input, model_pred, index, args.multiphase
                )
                # # TODO (kashif, sayakpaul): weighting sceme needs to be experimented with :)
                # if args.weighting_scheme == "sigma_sqrt":
                #     weighting = (sigmas**-2.0).astype(dtype="float32")
                # elif args.weighting_scheme == "logit_normal":
                #     # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
                #     u = torch.normal(mean=args.logit_mean, std=args.logit_std, size=(bsz,), device=accelerator.device)
                #     weighting = torch.nn.functional.sigmoid(u)
                # elif args.weighting_scheme == "mode":
                #     # See sec 3.1 in the SD3 paper (20).
                #     u = torch.rand(size=(bsz,), device=accelerator.device)
                #     weighting = 1 - u - args.mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
                weighting = 1.0
                # # simplified flow matching aka 0-rectified flow matching loss
                # # target = model_input - noise
                # target = model_input
                adv_index = paddle.empty_like(end_index)
                for i in range(end_index.shape[0]):
                    adv_index[i] = paddle.randint(
                        end_index[i].item(),
                        end_index[i].item()
                        + args.num_euler_timesteps // args.multiphase,
                        (1,),
                        dtype=end_index.dtype,
                    )
                sigmas_end = extract_into_tensor(
                    solver.sigmas_prev, end_index, model_input.shape
                )
                sigmas_adv = extract_into_tensor(
                    solver.sigmas_prev, adv_index, model_input.shape
                )
                timesteps_end = (
                    sigmas_end * noise_scheduler.config.num_train_timesteps
                ).squeeze([1, 2, 3])
                timesteps_adv = (
                    sigmas_adv * noise_scheduler.config.num_train_timesteps
                ).squeeze([1, 2, 3])

                with paddle.no_grad():
                    w = 3
                    with paddle.amp.auto_cast():
                        cond_teacher_output = teacher_transformer(
                            hidden_states=noisy_model_input.astype(dtype="float32"),
                            timestep=timesteps,
                            encoder_hidden_states=prompt_embeds.astype(dtype="float32"),
                            pooled_projections=pooled_prompt_embeds.astype(dtype="float32"),
                        ).sample
                        if args.not_apply_cfg_solver:
                            uncond_teacher_output = cond_teacher_output
                        else:
                            # Get teacher model prediction on noisy_latents and unconditional embedding
                            uncond_teacher_output = teacher_transformer(
                                hidden_states=noisy_model_input.astype(dtype="float32"),
                                timestep=timesteps,
                                encoder_hidden_states=uncond_prompt_embeds.astype(dtype="float32"),
                                pooled_projections=uncond_pooled_prompt_embeds.astype(dtype="float32"),
                            ).sample
                        teacher_output = cond_teacher_output + w * (
                            cond_teacher_output - uncond_teacher_output
                        )
                        x_prev = solver.euler_step(
                            noisy_model_input, teacher_output, index
                        )

                # 20.4.12. Get target LCM prediction on x_prev, w, c, t_n
                with paddle.no_grad():
                    with paddle.amp.auto_cast(dtype=weight_dtype):
                        target_pred = transformer(
                            hidden_states=x_prev.astype(dtype="float32"),
                            timestep=timesteps_prev,
                            encoder_hidden_states=prompt_embeds.astype(dtype="float32"),
                            pooled_projections=pooled_prompt_embeds.astype(dtype="float32"),
                        ).sample

                    target, end_index = solver.euler_style_multiphase_pred(
                        x_prev, target_pred, index, args.multiphase, True
                    )

                real_adv = (
                    (1 - sigmas_adv) * target
                    + (sigmas_adv - sigmas_end) * paddle.randn_like(target)
                ) / (1 - sigmas_end)
                fake_adv = (
                    (1 - sigmas_adv) * model_pred
                    + (sigmas_adv - sigmas_end) * paddle.randn_like(model_pred)
                ) / (1 - sigmas_end)

                if global_step % 2 == 0:
                    adv_optimizer.zero_grad()

                    # adversarial consistency loss

                    loss = discriminator(
                        "d_loss",
                        fake_adv.astype(dtype="float32"),
                        real_adv.astype(dtype="float32"),
                        timesteps_adv,
                        prompt_embeds.astype(dtype="float32"),
                        pooled_prompt_embeds.astype(dtype="float32"),
                        1.0,
                    )
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            discriminator.parameters(), args.max_grad_norm
                        )
                    adv_optimizer.step()
                    adv_optimizer.zero_grad()

                else:
                    # 20.4.13. Calculate loss
                    if args.loss_type == "l2":
                        loss = paddle.nn.functional.mse_loss(
                            model_pred.astype(dtype="float32"), target.astype(dtype="float32"), reduction="mean"
                        )
                    elif args.loss_type == "huber":
                        loss = paddle.mean(
                            paddle.sqrt(
                                (model_pred.astype(dtype="float32") - target.astype(dtype="float32")) ** 2
                                + args.huber_c**2
                            )
                            - args.huber_c
                        )

                    g_loss = args.adv_weight * discriminator(
                        "g_loss",
                        fake_adv.astype(dtype="float32"),
                        timesteps_adv,
                        prompt_embeds.astype(dtype="float32"),
                        pooled_prompt_embeds.astype(dtype="float32"),
                        1.0,
                    )
                    loss += g_loss
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = transformer_lora_parameters
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if (global_step - 1) % 2 == 0:
                    logs = {
                        "d_loss": loss.detach().item(),
                        "lr": lr_scheduler.get_lr(),
                    }
                else:
                    logs = {
                        "loss_cm": loss.detach().item() - g_loss.detach().item(),
                        "g_loss": g_loss.detach().item(),
                        "lr": lr_scheduler.get_lr(),
                    }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

                if accelerator.is_main_process:
                    if global_step % args.validation_steps == 0:
                        # create pipeline
                        text_encoder_one, text_encoder_two, text_encoder_three = (
                            load_text_encoders(
                                text_encoder_cls_one,
                                text_encoder_cls_two,
                                text_encoder_cls_three,
                            )
                        )
                        pipeline = StableDiffusion3Pipeline.from_pretrained(
                            args.pretrained_teacher_model,
                            vae=vae,
                            text_encoder=accelerator.unwrap_model(text_encoder_one),
                            text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                            text_encoder_3=accelerator.unwrap_model(text_encoder_three),
                            transformer=accelerator.unwrap_model(transformer),
                            revision=args.revision,
                            variant=args.variant,
                            paddle_dtype=weight_dtype,
                            scheduler=PCMFMDeterministicScheduler(1000, 3.0, args.num_euler_timesteps),

                        )
                        # pipeline.enable_xformers_memory_efficient_attention()
                        validation_prompts = [
                            "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
                            # "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
                            # "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                            # "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
                        ]
                        for prompt in validation_prompts:
                            pipeline_args = {
                                "prompt": prompt,
                                "num_inference_steps": args.multiphase,
                                "guidance_scale": 1.0,
                            }
                            with paddle.no_grad():
                                images = log_validation(
                                    pipeline=pipeline,
                                    args=args,
                                    accelerator=accelerator,
                                    pipeline_args=pipeline_args,
                                    step=global_step,
                                )
                        del text_encoder_one, text_encoder_two, text_encoder_three
                        del pipeline
                        paddle.device.cuda.empty_cache()
                        gc.collect()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        transformer = transformer.astype(dtype="float32")
        transformer_lora_layers = get_peft_model_state_dict(transformer)

        StableDiffusion3Pipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )

        # pipeline = StableDiffusion3Pipeline.from_pretrained(
        #     args.pretrained_teacher_model,
        #     revision=args.revision,
        #     variant=args.variant,
        #     paddle_dtype=weight_dtype,
        # )

        # Final inference
        # Load previous pipeline
        # pipeline = StableDiffusion3Pipeline.from_pretrained(
        #     args.pretrained_teacher_model,
        #     revision=args.revision,
        #     variant=args.variant,
        #     paddle_dtype=weight_dtype,
        # )
        # # load attention processors
        # pipeline.load_lora_weights(args.output_dir)

        # # run inference
        # images = []
        # if args.validation_prompt and args.num_validation_images > 0:
        #     pipeline_args = {"prompt": args.validation_prompt}
        #     images = log_validation(
        #         pipeline=pipeline,
        #         args=args,
        #         accelerator=accelerator,
        #         pipeline_args=pipeline_args,
        #         step=global_step,
        #         is_final_validation=True,
        #     )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
