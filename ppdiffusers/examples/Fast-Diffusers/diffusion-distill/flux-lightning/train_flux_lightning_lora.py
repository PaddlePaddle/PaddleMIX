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

import argparse
import base64
import logging
import math
import os

# import gc
from io import BytesIO

os.environ["USE_PEFT_BACKEND"] = "True"
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import List, Union

import numpy as np
import paddle
import paddle.nn as nn
from discriminator_flux import Discriminator
from paddle.vision import transforms
from paddlenlp.transformers import PretrainedConfig
from PIL import Image
from tqdm.auto import tqdm

import ppdiffusers
from ppdiffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from ppdiffusers.accelerate import Accelerator
from ppdiffusers.accelerate.logging import get_logger
from ppdiffusers.accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from ppdiffusers.optimization import get_scheduler
from ppdiffusers.peft import LoraConfig, set_peft_model_state_dict
from ppdiffusers.peft.utils import get_peft_model_state_dict
from ppdiffusers.transformers import CLIPTokenizer, T5Tokenizer
from ppdiffusers.utils import convert_unet_state_dict_to_peft, is_wandb_available

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def load_text_encoders(class_one, class_two):
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
    return text_encoder_one, text_encoder_two


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
    generator = paddle.Generator().manual_seed(args.seed) if args.seed else None
    autocast_ctx = nullcontext()

    with autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, step, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {pipeline_args['prompt']}") for i, image in enumerate(images)
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
    def __init__(self, sigmas, timesteps=1000, euler_timesteps=50, mu=1.15):
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1) ** 1.0)
        self.step_ratio = timesteps // euler_timesteps
        self.euler_timesteps = (np.arange(1, euler_timesteps + 1) * self.step_ratio).round().astype(np.int64) - 1
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
        sigma_prev = extract_into_tensor(self.sigmas_prev, timestep_index, model_pred.shape)
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

        inference_indices = np.linspace(0, len(self.euler_timesteps), num=multiphase, endpoint=False)
        inference_indices = np.floor(inference_indices).astype(np.int64)
        inference_indices = paddle.to_tensor(inference_indices).astype(dtype="int64")
        expanded_timestep_index = timestep_index.unsqueeze(1).expand([-1, inference_indices.shape[0]])
        valid_indices_mask = expanded_timestep_index >= inference_indices
        last_valid_index = valid_indices_mask.flip(axis=[1]).astype(dtype="int64").argmax(axis=1)
        last_valid_index = inference_indices.shape[0] - 1 - last_valid_index
        timestep_index_end = inference_indices[last_valid_index]

        if is_target:
            sigma = extract_into_tensor(self.sigmas_prev, timestep_index, sample.shape)
        else:
            sigma = extract_into_tensor(self.sigmas, timestep_index, sample.shape)
        sigma_prev = extract_into_tensor(self.sigmas_prev, timestep_index_end, sample.shape)
        x_prev = sample + (sigma_prev - sigma) * model_pred

        return x_prev, timestep_index_end


def import_model_class_from_model_name_or_path(
    pretrained_teacher_model: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_teacher_model, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from ppdiffusers.transformers import CLIPTextModel

        return CLIPTextModel
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
        "--file_list_path",
        type=str,
        required=True,
        help="Path to file list",
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
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
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
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
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
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
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
        help=("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16)."),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=("Choose prior generation precision between fp32, fp16 and bf16 (bfloat16)."),
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
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        choices=["O0", "O1", "O2"],
        help=("For fp16/bf16: AMP optimization level selected in ['O0', 'O1', and 'O2']."),
    )
    parser.add_argument("--not_apply_cfg_solver", action="store_true")
    parser.add_argument("--multiphase", default=8, type=int)
    parser.add_argument("--adv_weight", default=0.1, type=float)
    parser.add_argument("--adv_lr", default=1e-5, type=float)
    parser.add_argument("--adv_loss_type", default="pcm", type=str)
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    )
    parser.add_argument(
        "--pre_alloc_memory",
        type=int,
        default=0,
        help="allocate memory for paddle in advance to save memory",
    )
    parser.add_argument("--use_dmd_loss", action="store_true", help="whether use dmd loss")
    parser.add_argument("--dmd_weight", default=0.1, type=float, help="dmd loss weight")
    parser.add_argument("--apply_reflow_loss", action="store_true", help="whether apply reflow loss")
    parser.add_argument("--reflow_loss_weight", default=1.0, type=float, help="reflow loss weight")
    parser.add_argument(
        "--use_distill_loss", action="store_true", help="whether use distill loss instead of consistency model loss"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class LaionDataset(paddle.io.Dataset):
    def __init__(self, data_dir, list_path, sample_size):
        """
        Args:
            img_dir (string): Directory with all the images and text files.
            sample_size (tuple): Desired sample size as (height, width).
        """
        self.data_dir = data_dir
        self.list_path = list_path
        self.sample_size = sample_size
        with open(self.list_path, "r") as f:
            self.indexes = f.readlines()
        self.data_number = len(self.indexes)
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.sample_size, interpolation="lanczos"),
                transforms.CenterCrop(self.sample_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return self.data_number

    def __getitem__(self, idx):
        text_path = self.indexes[idx].strip()
        text_path = os.path.join(self.data_dir, text_path)
        if not os.path.exists(text_path):
            raise ValueError("can't find {}".format(text_path))

        while True:
            try:
                with open(text_path, "r") as f:
                    lines = f.readlines()
                line = lines[0].split("\t")
                caption_json_str = eval(line[2])
                text = caption_json_str["blip_caption_en"]
                img_base64 = line[5]
                image_bytes = base64.b64decode(img_base64)
                image_io = BytesIO(image_bytes)
                image = Image.open(image_io).convert("RGB")
                image = self.transform(image)
            except:
                idx = random.randint(0, self.data_number)
                continue

            return {"pixel_values": image, "prompts": text}


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pd",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pd",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids)[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.astype(dtype)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.tile([1, num_images_per_prompt, 1])
    prompt_embeds = prompt_embeds.reshape([batch_size * num_images_per_prompt, seq_len, -1])

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pd",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids, output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.astype(dtype=text_encoder.dtype)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.tile([1, num_images_per_prompt, 1])
    prompt_embeds = prompt_embeds.reshape([batch_size * num_images_per_prompt, -1])

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = paddle.zeros([prompt_embeds.shape[1], 3]).astype(dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def cast_training_params(model: Union[paddle.nn.Layer, List[paddle.nn.Layer]], dtype=paddle.float32):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if not param.stop_gradient:
                param.set_value(param.to(dtype=dtype))


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        fp16_opt_level=args.fp16_opt_level,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    if args.pre_alloc_memory > 0:
        memory_size = int(args.pre_alloc_memory * 1024 * 1024 * 1024)
        x = paddle.empty([memory_size], dtype=paddle.uint8)  # noqa: F841
        del x

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

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

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = "float32"
    if accelerator.mixed_precision == "fp16":
        weight_dtype = "float16"
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = "bfloat16"

    # Load the tokenizers
    print(args.pretrained_teacher_model)
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5Tokenizer.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_teacher_model, args.revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_teacher_model, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_teacher_model, subfolder="scheduler"
    )
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_teacher_model,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        paddle_dtype=weight_dtype,
    )

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    discriminator = Discriminator(transformer, args.resolution)
    discriminator_params = []
    for param in discriminator.heads.parameters():
        param.requires_grad = True
        discriminator_params.append(param)

    vae.to(dtype=paddle.float32)
    transformer.to(dtype=weight_dtype)
    text_encoder_one.to(dtype=weight_dtype)
    text_encoder_two.to(dtype=weight_dtype)

    image_seq_len = (args.resolution * args.resolution) // (16 * 16)
    mu = calculate_shift(
        image_seq_len,
        noise_scheduler.config.base_image_seq_len,
        noise_scheduler.config.max_image_seq_len,
        noise_scheduler.config.base_shift,
        noise_scheduler.config.max_shift,
    )
    solver = EulerSolver(
        noise_scheduler.sigmas.numpy()[::-1],
        noise_scheduler.config.num_train_timesteps,
        euler_timesteps=args.num_euler_timesteps,
        mu=mu,
    )

    # not available yet
    # transformer.enable_xformers_memory_efficient_attention()
    # teacher_transformer.enable_xformers_memory_efficient_attention()
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # now we will add new LoRA weights to the attention layers
    transformer_lora_config = LoraConfig(
        r=args.lora_rank,
        target_modules=[
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
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

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            FluxPipeline.save_lora_weights(output_dir, transformer_lora_layers=transformer_lora_layers_to_save)
            paddle.save(
                accelerator.unwrap_model(discriminator).heads.state_dict(), os.path.join(output_dir, "heads.pdparams")
            )

    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model

        lora_state_dict = FluxPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v
            for k, v in lora_state_dict[0].items()
            if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
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
        # if args.mixed_precision == "fp16":
        #     models = [transformer_]
        #     # only upcast trainable parameters (LoRA) into fp32
        #     cast_training_params(models)

        accelerator.unwrap_model(discriminator).heads.set_state_dict(
            paddle.load(os.path.join(input_dir, "heads.pdparams"))
        )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16" or args.mixed_precision == "bf16":
        models = [transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=paddle.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {
        "params": transformer_lora_parameters,
    }
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    optimizer_class = paddle.optimizer.AdamW

    optimizer_kwargs = {}
    if hasattr(optimizer_class, "_create_master_weight") and accelerator.fp16_opt_level == "O2":
        optimizer_kwargs["multi_precision"] = True
    optimizer = optimizer_class(
        parameters=params_to_optimize,
        learning_rate=args.learning_rate,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.adam_weight_decay,
        epsilon=args.adam_epsilon,
        grad_clip=nn.ClipGradByGlobalNorm(args.max_grad_norm) if args.max_grad_norm > 0 else None,
        **optimizer_kwargs,
    )
    adv_optimizer = optimizer_class(
        parameters=discriminator_params,
        learning_rate=args.adv_lr,
        beta1=0,
        beta2=0.999,
        weight_decay=args.adam_weight_decay,
        epsilon=args.adam_epsilon,
        grad_clip=nn.ClipGradByGlobalNorm(args.max_grad_norm) if args.max_grad_norm > 0 else None,
        **optimizer_kwargs,
    )

    # Dataset and DataLoaders creation:
    print("start training")
    train_dataset = LaionDataset(args.data_path, args.file_list_path, args.resolution)
    batch_sampler = paddle.io.DistributedBatchSampler(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    train_dataloader = paddle.io.DataLoader(
        dataset=train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.dataloader_num_workers,
    )
    print("the number of training data ", len(train_dataloader))

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with paddle.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
        return prompt_embeds, pooled_prompt_embeds, text_ids

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
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
    (discriminator, adv_optimizer, transformer, optimizer, train_dataloader, lr_scheduler,) = accelerator.prepare(
        discriminator,
        adv_optimizer,
        transformer,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
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

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer, discriminator]
            with accelerator.accumulate(models_to_accumulate):
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                prompts = batch["prompts"]

                prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                    prompts, text_encoders, tokenizers
                )

                # Convert images to latent space
                with paddle.no_grad():
                    model_input = vae.encode(pixel_values).latent_dist.sample()

                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    weight_dtype,
                )

                # Sample noise that we'll add to the latents
                noise = paddle.randn(shape=model_input.shape, dtype=model_input.dtype)
                bsz = model_input.shape[0]

                index = paddle.randint(0, args.num_euler_timesteps, (bsz,)).astype(dtype="int64")

                # Add noise according to flow matching.
                sigmas = extract_into_tensor(solver.sigmas, index, model_input.shape)
                sigmas_prev = extract_into_tensor(solver.sigmas_prev, index, model_input.shape)
                timesteps = (sigmas * noise_scheduler.config.num_train_timesteps).squeeze([1, 2, 3])
                timesteps_prev = (sigmas_prev * noise_scheduler.config.num_train_timesteps).squeeze([1, 2, 3])
                noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                # handle guidance
                guidance = paddle.to_tensor([args.guidance_scale])
                guidance = guidance.expand(packed_noisy_model_input.shape[0])

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]

                model_pred, end_index = solver.euler_style_multiphase_pred(
                    packed_noisy_model_input, model_pred, index, args.multiphase
                )
                adv_index = paddle.empty_like(end_index)
                for i in range(end_index.shape[0]):
                    adv_index[i] = paddle.randint(
                        end_index[i].item(),
                        end_index[i].item() + args.num_euler_timesteps // args.multiphase,
                        (1,),
                        dtype=end_index.dtype,
                    )
                sigmas_end = extract_into_tensor(solver.sigmas_prev, end_index, packed_noisy_model_input.shape)
                sigmas_adv = extract_into_tensor(solver.sigmas_prev, adv_index, packed_noisy_model_input.shape)
                timesteps_adv = (sigmas_adv * noise_scheduler.config.num_train_timesteps).squeeze([1, 2])

                if args.use_distill_loss:
                    delta_index = index[0].item() - end_index[0].item()
                    with paddle.no_grad():
                        with paddle.amp.auto_cast():
                            unwrapped_transformer = accelerator.unwrap_model(transformer)
                            unwrapped_transformer.disable_adapters()
                            packed_noisy_model_input_teacher = packed_noisy_model_input
                            for di in range(delta_index):
                                idx = index - di
                                sigmas = extract_into_tensor(solver.sigmas, idx, model_input.shape)
                                timesteps_ = (sigmas * noise_scheduler.config.num_train_timesteps).squeeze([1, 2, 3])
                                teacher_output = unwrapped_transformer(
                                    hidden_states=packed_noisy_model_input_teacher.astype(dtype="float32"),
                                    timestep=timesteps_ / 1000,
                                    guidance=guidance,
                                    pooled_projections=pooled_prompt_embeds.astype(dtype="float32"),
                                    encoder_hidden_states=prompt_embeds.astype(dtype="float32"),
                                    txt_ids=text_ids,
                                    img_ids=latent_image_ids,
                                ).sample
                                packed_noisy_model_input_teacher = solver.euler_step(
                                    packed_noisy_model_input_teacher.astype(dtype="float32"), teacher_output, idx
                                )
                        unwrapped_transformer.enable_adapters()
                        target = packed_noisy_model_input_teacher
                else:
                    with paddle.no_grad():
                        with paddle.amp.auto_cast():
                            unwrapped_transformer = accelerator.unwrap_model(transformer)
                            unwrapped_transformer.disable_adapters()
                            teacher_output = unwrapped_transformer(
                                hidden_states=packed_noisy_model_input.astype(dtype="float32"),
                                timestep=timesteps / 1000,
                                guidance=guidance,
                                pooled_projections=pooled_prompt_embeds.astype(dtype="float32"),
                                encoder_hidden_states=prompt_embeds.astype(dtype="float32"),
                                txt_ids=text_ids,
                                img_ids=latent_image_ids,
                            ).sample
                            unwrapped_transformer.enable_adapters()
                            x_prev = solver.euler_step(packed_noisy_model_input, teacher_output, index)

                    # 20.4.12. Get target LCM prediction on x_prev, w, c, t_n
                    with paddle.no_grad():
                        with paddle.amp.auto_cast(dtype=weight_dtype):
                            target_pred = transformer(
                                hidden_states=x_prev.astype(dtype="float32"),
                                timestep=timesteps_prev / 1000,
                                guidance=guidance,
                                pooled_projections=pooled_prompt_embeds.astype(dtype="float32"),
                                encoder_hidden_states=prompt_embeds.astype(dtype="float32"),
                                txt_ids=text_ids,
                                img_ids=latent_image_ids,
                            ).sample

                        target, end_index = solver.euler_style_multiphase_pred(
                            x_prev, target_pred, index, args.multiphase, True
                        )

                real_adv = ((1 - sigmas_adv) * target + (sigmas_adv - sigmas_end) * paddle.randn_like(target)) / (
                    1 - sigmas_end
                )
                fake_adv = (
                    (1 - sigmas_adv) * model_pred + (sigmas_adv - sigmas_end) * paddle.randn_like(model_pred)
                ) / (1 - sigmas_end)

                if global_step % 2 == 0:
                    adv_optimizer.zero_grad()

                    # adversarial consistency loss

                    loss = discriminator(
                        "d_loss",
                        fake_adv.astype(dtype="float32"),
                        real_adv.astype(dtype="float32"),
                        timesteps_adv / 1000,
                        prompt_embeds.astype(dtype="float32"),
                        pooled_prompt_embeds.astype(dtype="float32"),
                        guidance,
                        text_ids,
                        latent_image_ids,
                        1.0,
                        args.adv_loss_type,
                    )
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(discriminator.parameters(), args.max_grad_norm)
                    adv_optimizer.step()
                    adv_optimizer.zero_grad()

                else:
                    loss = 0.0
                    if args.use_dmd_loss:
                        dmd_index = paddle.empty_like(end_index)
                        for i in range(end_index.shape[0]):
                            dmd_index[i] = paddle.randint(
                                end_index[i].item(),
                                end_index[i].item() + args.num_euler_timesteps // args.multiphase,
                                (1,),
                                dtype=end_index.dtype,
                            )
                        sigmas_end = extract_into_tensor(solver.sigmas_prev, end_index, packed_noisy_model_input.shape)
                        sigmas_dmd = extract_into_tensor(solver.sigmas_prev, dmd_index, packed_noisy_model_input.shape)

                        timesteps_dmd = (sigmas_dmd * noise_scheduler.config.num_train_timesteps).squeeze([1, 2])
                        fake_dmd = (
                            (1 - sigmas_dmd) * model_pred + (sigmas_dmd - sigmas_end) * paddle.randn_like(model_pred)
                        ) / (1 - sigmas_end)

                        with paddle.no_grad():
                            with paddle.amp.auto_cast(dtype=weight_dtype):
                                fake_noise_pred = transformer(
                                    hidden_states=fake_dmd.astype(dtype="float32"),
                                    timestep=timesteps_dmd / 1000,
                                    guidance=guidance,
                                    pooled_projections=pooled_prompt_embeds.astype(dtype="float32"),
                                    encoder_hidden_states=prompt_embeds.astype(dtype="float32"),
                                    txt_ids=text_ids,
                                    img_ids=latent_image_ids,
                                ).sample

                                unwrapped_transformer.disable_adapters()
                                real_noise_pred = unwrapped_transformer(
                                    hidden_states=fake_dmd.astype(dtype="float32"),
                                    timestep=timesteps_dmd / 1000,
                                    guidance=guidance,
                                    pooled_projections=pooled_prompt_embeds.astype(dtype="float32"),
                                    encoder_hidden_states=prompt_embeds.astype(dtype="float32"),
                                    txt_ids=text_ids,
                                    img_ids=latent_image_ids,
                                ).sample
                                unwrapped_transformer.enable_adapters()

                        score_real = -real_noise_pred
                        score_fake = -fake_noise_pred
                        coeff = score_fake - score_real
                        pred_x_0_student = real_noise_pred

                        weight = (
                            1.0 / ((model_pred - pred_x_0_student).abs().mean([1, 2], keepdim=True) + 1e-5).detach()
                        )
                        loss_dmd = args.dmd_weight * paddle.nn.functional.mse_loss(
                            model_pred, (model_pred - weight * coeff).detach(), reduction="mean"
                        )
                        loss += loss_dmd

                    if args.apply_reflow_loss:
                        reflow_index = paddle.empty_like(end_index)
                        for i in range(end_index.shape[0]):
                            reflow_index[i] = paddle.randint(
                                end_index[i].item(),
                                end_index[i].item() + args.num_euler_timesteps // args.multiphase,
                                (1,),
                                dtype=end_index.dtype,
                            )
                        sigmas_end = extract_into_tensor(solver.sigmas_prev, end_index, packed_noisy_model_input.shape)
                        sigmas_reflow = extract_into_tensor(
                            solver.sigmas_prev, reflow_index, packed_noisy_model_input.shape
                        )
                        timesteps_reflow = (sigmas_reflow * noise_scheduler.config.num_train_timesteps).squeeze([1, 2])

                        reflow_input = (
                            (1 - sigmas_reflow) * model_pred.detach()
                            + (sigmas_reflow - sigmas_end) * paddle.randn_like(model_pred.detach())
                        ) / (1 - sigmas_end)

                        noise_pred_reflow = transformer(
                            hidden_states=reflow_input.astype(dtype="float32"),
                            timestep=timesteps_reflow / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds.astype(dtype="float32"),
                            encoder_hidden_states=prompt_embeds.astype(dtype="float32"),
                            txt_ids=text_ids,
                            img_ids=latent_image_ids,
                            return_dict=False,
                        )[0]
                        latents_pred_reflow, _ = solver.euler_style_multiphase_pred(
                            reflow_input, noise_pred_reflow, reflow_index, args.multiphase
                        )
                        loss_reflow = args.reflow_loss_weight * paddle.mean(
                            paddle.sqrt(
                                (
                                    latents_pred_reflow.astype(dtype="float32")
                                    - model_pred.detach().astype(dtype="float32")
                                )
                                ** 2
                                + args.huber_c**2
                            )
                            - args.huber_c
                        )
                        loss += loss_reflow

                    # 20.4.13. Calculate loss
                    if args.loss_type == "l2":
                        loss_cm = paddle.nn.functional.mse_loss(
                            model_pred.astype(dtype="float32"), target.astype(dtype="float32"), reduction="mean"
                        )
                        loss += loss_cm
                    elif args.loss_type == "huber":
                        loss_cm = paddle.mean(
                            paddle.sqrt(
                                (model_pred.astype(dtype="float32") - target.astype(dtype="float32")) ** 2
                                + args.huber_c**2
                            )
                            - args.huber_c
                        )
                        loss += loss_cm

                    g_loss = args.adv_weight * discriminator(
                        "g_loss",
                        fake_adv.astype(dtype="float32"),
                        timesteps_adv / 1000,
                        prompt_embeds.astype(dtype="float32"),
                        pooled_prompt_embeds.astype(dtype="float32"),
                        guidance,
                        text_ids,
                        latent_image_ids,
                        1.0,
                        args.adv_loss_type,
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
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if (global_step - 1) % 2 == 0:
                    logs = {
                        "d_loss": loss.detach().item(),
                        "lr": lr_scheduler.get_lr(),
                    }
                else:
                    logs = {
                        "loss_cm": loss_cm.detach().item(),
                        "loss_dmd": loss_dmd.detach().item() if args.use_dmd_loss else None,
                        "loss_reflow": loss_reflow.detach().item() if args.apply_reflow_loss else None,
                        "g_loss": g_loss.detach().item(),
                        "lr": lr_scheduler.get_lr(),
                    }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        transformer = transformer.astype(dtype="float32")
        transformer_lora_layers = get_peft_model_state_dict(transformer)

        FluxPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
