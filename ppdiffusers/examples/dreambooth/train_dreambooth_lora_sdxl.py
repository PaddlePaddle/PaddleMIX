# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
import contextlib
import gc
import hashlib
import math
import os

# import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Type

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import requests
from huggingface_hub import HfFolder, create_repo, upload_folder, whoami
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
)
from paddle.io import BatchSampler, DataLoader, Dataset, DistributedBatchSampler
from paddle.optimizer import AdamW
from paddle.vision import BaseTransform, transforms
from paddlenlp.trainer import set_seed
from paddlenlp.transformers import AutoTokenizer, PretrainedConfig
from paddlenlp.utils.log import logger
from PIL import Image
from PIL.ImageOps import exif_transpose
from tqdm.auto import tqdm

from ppdiffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    is_ppxformers_available,
)
from ppdiffusers.loaders import LoraLoaderMixin, text_encoder_lora_state_dict
from ppdiffusers.models.attention_processor import (
    LoRAAttnProcessor,
    LoRAAttnProcessor2_5,
)
from ppdiffusers.optimization import get_scheduler
from ppdiffusers.training_utils import freeze_params, unwrap_model

# from ppdiffusers.utils import check_min_version
# from ppdiffusers.utils import TEXT_ENCODER_ATTN_MODULE
from ppdiffusers.utils import check_min_version

# Will error if the minimal version of ppdiffusers is not installed. Remove at your own risks.
check_min_version("0.19.3")
paddle_dtype = paddle.float32

# Since HF sometimes timeout, we need to retry uploads
# Credit: https://github.com/huggingface/datasets/blob/06ae3f678651bfbb3ca7dd3274ee2f38e0e0237e/src/datasets/utils/file_utils.py#L265


def _retry(
    func,
    func_args: Optional[tuple] = None,
    func_kwargs: Optional[dict] = None,
    exceptions: Type[requests.exceptions.RequestException] = requests.exceptions.RequestException,
    max_retries: int = 0,
    base_wait_time: float = 0.5,
    max_wait_time: float = 2,
):
    func_args = func_args or ()
    func_kwargs = func_kwargs or {}
    retry = 0
    while True:
        try:
            return func(*func_args, **func_kwargs)
        except exceptions as err:
            if retry >= max_retries:
                raise err
            else:
                sleep_time = min(max_wait_time, base_wait_time * 2**retry)  # Exponential backoff
                logger.info(f"{func} timed out, retrying in {sleep_time}s... [{retry/max_retries}]")
                time.sleep(sleep_time)
                retry += 1


def url_or_path_join(*path_list):
    return os.path.join(*path_list) if os.path.isdir(os.path.join(*path_list)) else "/".join(path_list)


def save_model_card(
    repo_id: str, images=None, base_model=str, train_text_encoder=False, prompt=str, repo_folder=None, vae_path=None
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
instance_prompt: {prompt}
tags:
- stable-diffusion-xl
- stable-diffusion-xl-diffusers
- text-to-image
- ppdiffusers
- lora
inference: false
---
    """

    model_card = f"""
# LoRA DreamBooth - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/). You can find some example images in the following. \n
{img_str}

LoRA for the text encoder was enabled: {train_text_encoder}.

Special VAE used for training: {vae_path}.

## License

[SDXL 1.0 License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
"""

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, subfolder: str = "text_encoder"):
    # try:
    #     text_encoder_config = PretrainedConfig.from_pretrained(
    #         url_or_path_join(pretrained_model_name_or_path, subfolder)
    #     )
    #     model_class = text_encoder_config.architectures[0]
    # except Exception:
    #     model_class = "LDMBertModel"
    print(url_or_path_join(pretrained_model_name_or_path, subfolder))
    text_encoder_config = PretrainedConfig.from_pretrained(url_or_path_join(pretrained_model_name_or_path, subfolder))
    model_class = text_encoder_config.architectures[0]

    text_encoder_config = PretrainedConfig.from_pretrained(url_or_path_join(pretrained_model_name_or_path, subfolder))
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from paddlenlp.transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from paddlenlp.transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from paddlenlp.transformers import T5EncoderModel

        return T5EncoderModel
    elif model_class == "BertModel":
        from paddlenlp.transformers import BertModel

        return BertModel
    elif model_class == "LDMBertModel":
        from ppdiffusers.pipelines.latent_diffusion.pipeline_latent_diffusion import (
            LDMBertModel,
        )

        return LDMBertModel
    else:
        raise ValueError(f"{model_class} is not supported.")


class Lambda(BaseTransform):
    def __init__(self, fn, keys=None):
        super().__init__(keys)
        self.fn = fn

    def _apply_image(self, img):
        return self.fn(img)


def get_report_to(args):
    if args.report_to == "visualdl":
        from visualdl import LogWriter

        writer = LogWriter(logdir=args.logging_dir)
    elif args.report_to == "tensorboard":
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(logdir=args.logging_dir)
    else:
        raise ValueError("report_to must be in ['visualdl', 'tensorboard']")
    return writer


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training dreambooth lora script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora-dreambooth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=(
            "The height for input images, all the images in the train/validation dataset will be resized to this"
            " height"
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "The width for input images, all the images in the train/validation dataset will be resized to this"
            " width"
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The rank of lora linear.",
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
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
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
        help=("Save a checkpoint of the training state every X updates."),
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
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
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
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
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
            "[TensorBoard](https://www.tensorflow.org/tensorboard) or [VisualDL](https://www.paddlepaddle.org.cn/paddle/visualdl) log directory. Will default to"
            "*output_dir/logs"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires Paddle >="
            " 2.5.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="visualdl",
        choices=["tensorboard", "visualdl"],
        help="Log writer type.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="fp16_opt_level.",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.instance_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    args.logging_dir = os.path.join(args.output_dir, args.logging_dir)
    if args.height is None or args.width is None and args.resolution is not None:
        args.height = args.width = args.resolution

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        instance_data_root,
        class_data_root=None,
        class_num=None,
        height=1024,
        width=1024,
        center_crop=False,
        interpolation="bilinear",
        random_flip=False,
    ):
        self.height = height
        self.width = width
        self.center_crop = center_crop

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        ext = ["png", "jpg", "jpeg", "bmp", "PNG", "JPG", "JPEG", "BMP"]
        self.instance_images_path = []
        for p in Path(instance_data_root).iterdir():
            if any(suffix in p.name for suffix in ext):
                self.instance_images_path.append(p)
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = []
            for p in Path(class_data_root).iterdir():
                if any(suffix in p.name for suffix in ext):
                    self.class_images_path.append(p)
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=interpolation),
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

        return example


# def tokenize_prompt(tokenizer, prompt,):

#     text_inputs = tokenizer(
#         prompt,
#         truncation=True,
#         padding="do_not_pad",
#         max_length=tokenizer.model_max_length,
#         return_attention_mask=True,
#         return_tensors="pd",
#     )
#     text_input_ids = text_inputs.input_ids
#     return text_inputs


def tokenize_prompt(
    tokenizer,
    prompt,
):

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pd",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None, args=None):
    prompt_embeds_list = []
    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        with paddle.amp.auto_cast(
            enable=args.mixed_precision in ["bf16", "fp16"] and args.train_text_encoder,
            level=args.fp16_opt_level,
            custom_black_list=["reduce_sum", "c_softmax_with_cross_entropy"],
            custom_white_list=["lookup_table", "lookup_table_v2"] if args.fp16_opt_level == "O2" else ["layer_norm"],
            dtype="bfloat16" if args.mixed_precision == "bf16" else "float16",
        ):
            prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.reshape([bs_embed, seq_len, -1])
        prompt_embeds_list.append(prompt_embeds)
    prompt_embeds = paddle.concat(x=prompt_embeds_list, axis=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.reshape([bs_embed, -1])
    return prompt_embeds, pooled_prompt_embeds


def unet_attn_processors_state_dict(unet) -> Dict[str, paddle.Tensor]:
    """
    Returns:
        a state dict containing just the attention processor parameters.
    """
    attn_processors = unet.attn_processors

    attn_processors_state_dict = {}

    for attn_processor_key, attn_processor in attn_processors.items():
        for parameter_key, parameter in attn_processor.state_dict().items():
            attn_processors_state_dict[f"{attn_processor_key}.{parameter_key}"] = parameter

    return attn_processors_state_dict


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main():
    args = parse_args()
    rank = paddle.distributed.get_rank()
    is_main_process = rank == 0
    num_processes = paddle.distributed.get_world_size()
    if num_processes > 1:
        paddle.distributed.init_parallel_env()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                safety_checker=None,
                requires_safety_checker=False,
                paddle_dtype=paddle_dtype,
            )
            if args.enable_xformers_memory_efficient_attention and is_ppxformers_available():
                try:
                    pipeline.unet.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warning(
                        "Could not enable memory efficient attention. Make sure develop paddlepaddle is installed"
                        f" correctly and a GPU is available: {e}"
                    )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            batch_sampler = (
                DistributedBatchSampler(sample_dataset, batch_size=args.sample_batch_size, shuffle=False)
                if num_processes > 1
                else BatchSampler(sample_dataset, batch_size=args.sample_batch_size, shuffle=False)
            )
            sample_dataloader = DataLoader(
                sample_dataset, batch_sampler=batch_sampler, num_workers=args.dataloader_num_workers
            )

            for example in tqdm(sample_dataloader, desc="Generating class images", disable=not is_main_process):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)
            pipeline.to("cpu")
            del pipeline
            gc.collect()

    if is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    tokenizer_one = AutoTokenizer.from_pretrained(url_or_path_join(args.pretrained_model_name_or_path, "tokenizer"))
    tokenizer_two = AutoTokenizer.from_pretrained(url_or_path_join(args.pretrained_model_name_or_path, "tokenizer_2"))

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        url_or_path_join(args.pretrained_model_name_or_path, "text_encoder")
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        url_or_path_join(args.pretrained_model_name_or_path, "text_encoder_2")
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        url_or_path_join(vae_path, "vae") if args.pretrained_vae_model_name_or_path is None else vae_path
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )

    # We only train the additional adapter LoRA layers
    freeze_params(vae.parameters())
    freeze_params(text_encoder_one.parameters())
    freeze_params(text_encoder_two.parameters())
    freeze_params(unet.parameters())

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = "float32"
    do_grad_scaling = False
    scaler = None
    if args.mixed_precision == "fp16":
        weight_dtype = "float16"
        scaler = paddle.amp.GradScaler(
            enable=True,
            init_loss_scaling=65536.0,
            incr_every_n_steps=2000,
        )
        do_grad_scaling = True
    elif args.mixed_precision == "bf16":
        weight_dtype = "bfloat16"

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    if weight_dtype != "float32":
        paddle.amp.decorate(
            models=[unet],
            level="O2",
            dtype=weight_dtype,
        )
        nn.Layer._to_impl(vae, dtype=weight_dtype, floating_only=True)
        if args.train_text_encoder:
            paddle.amp.decorate(
                models=[text_encoder_one],
                level="O2",
                dtype=weight_dtype,
            )
            paddle.amp.decorate(
                models=[text_encoder_two],
                level="O2",
                dtype=weight_dtype,
            )
        else:
            nn.Layer._to_impl(text_encoder_one, dtype=weight_dtype, floating_only=True)
            nn.Layer._to_impl(text_encoder_two, dtype=weight_dtype, floating_only=True)

    # # Move unet, vae and text_encoder to device and cast to paddle_dtype
    # # The VAE is in float32 to avoid NaN losses.
    # unet.to(dtype=paddle_dtype)
    # if args.pretrained_vae_model_name_or_path is None:
    #     vae.to(dtype=paddle.float32)
    # else:
    #     vae.to(dtype=paddle_dtype)
    # text_encoder_one.to(dtype=paddle_dtype)
    # text_encoder_two.to(dtype=paddle_dtype)

    if args.enable_xformers_memory_efficient_attention and is_ppxformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure develop paddlepaddle is installed"
                f" correctly and a GPU is available: {e}"
            )
    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    unet_lora_attn_procs = {}
    unet_lora_parameters = []
    for name, attn_processor in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_processor_class = (
            LoRAAttnProcessor2_5 if hasattr(F, "scaled_dot_product_attention_") else LoRAAttnProcessor
        )

        module = lora_attn_processor_class(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        unet_lora_attn_procs[name] = module
        unet_lora_parameters.extend(module.parameters())

    unet.set_attn_processor(unet_lora_attn_procs)

    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks.
    if args.train_text_encoder:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        text_lora_parameters_one = LoraLoaderMixin._modify_text_encoder(text_encoder_one, dtype=paddle.float32)
        text_lora_parameters_two = LoraLoaderMixin._modify_text_encoder(text_encoder_two, dtype=paddle.float32)

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[timesteps].cast("float32")
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[timesteps].cast("float32")
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    # # create custom saving & loading functions for Lora
    # def save_model_func(models, weights, output_dir):
    #     # there are only two options here. Either are just the unet attn processor layers
    #     # or there are the unet and text encoder atten layers
    #     unet_lora_layers_to_save = None
    #     text_encoder_lora_layers_to_save = None

    #     for model in models:
    #         if isinstance(model, type(unwrap_model(unet))):
    #             unet_lora_layers_to_save = unet_attn_processors_state_dict(model)
    #         elif isinstance(model, type(unwrap_model(text_encoder))):
    #             text_encoder_lora_layers_to_save = text_encoder_lora_state_dict(model)
    #         else:
    #             raise ValueError(f"unexpected save model: {model.__class__}")

    #         # make sure to pop weight so that corresponding model is not saved again
    #         weights.pop()

    #     LoraLoaderMixin.save_lora_weights(
    #         output_dir,
    #         unet_lora_layers=unet_lora_layers_to_save,
    #         text_encoder_lora_layers=text_encoder_lora_layers_to_save,
    #     )

    # def load_model_func(models, input_dir):
    #     unet_ = None
    #     text_encoder_ = None

    #     while len(models) > 0:
    #         model = models.pop()

    #         if isinstance(model, type(unwrap_model(unet))):
    #             unet_ = model
    #         elif isinstance(model, type(unwrap_model(text_encoder))):
    #             text_encoder_ = model
    #         else:
    #             raise ValueError(f"unexpected save model: {model.__class__}")

    #     lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(input_dir)
    #     LoraLoaderMixin.load_lora_into_unet(lora_state_dict, network_alphas=network_alphas, unet=unet_)
    #     LoraLoaderMixin.load_lora_into_text_encoder(
    #         lora_state_dict, network_alphas=network_alphas, text_encoder=text_encoder_
    #     )

    def compute_time_ids():
        original_size = args.resolution, args.resolution
        target_size = args.resolution, args.resolution
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = paddle.to_tensor(data=[add_time_ids])
        add_time_ids = add_time_ids.cast(dtype=paddle_dtype)
        return add_time_ids

    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with paddle.no_grad():
                prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, args=args)
            return prompt_embeds, pooled_prompt_embeds

    instance_time_ids = compute_time_ids()
    if not args.train_text_encoder:
        instance_prompt_hidden_states, instance_pooled_prompt_embeds = compute_text_embeddings(
            args.instance_prompt, text_encoders, tokenizers
        )

    if args.with_prior_preservation:
        class_time_ids = compute_time_ids()
        if not args.train_text_encoder:
            class_prompt_hidden_states, class_pooled_prompt_embeds = compute_text_embeddings(
                args.class_prompt, text_encoders, tokenizers
            )

    if not args.train_text_encoder:
        del tokenizers, text_encoders
        gc.collect()
        paddle.device.cuda.empty_cache()

    add_time_ids = instance_time_ids

    if args.with_prior_preservation:
        add_time_ids = paddle.concat(x=[add_time_ids, class_time_ids], axis=0)
    if not args.train_text_encoder:
        prompt_embeds = instance_prompt_hidden_states
        unet_add_text_embeds = instance_pooled_prompt_embeds
        if args.with_prior_preservation:
            prompt_embeds = paddle.concat(x=[prompt_embeds, class_prompt_hidden_states], axis=0)
            unet_add_text_embeds = paddle.concat(x=[unet_add_text_embeds, class_pooled_prompt_embeds], axis=0)
    else:
        tokens_one = tokenize_prompt(tokenizer_one, args.instance_prompt)
        tokens_two = tokenize_prompt(tokenizer_two, args.instance_prompt)
        if args.with_prior_preservation:
            class_tokens_one = tokenize_prompt(tokenizer_one, args.class_prompt)
            class_tokens_two = tokenize_prompt(tokenizer_two, args.class_prompt)
            tokens_one = paddle.concat(x=[tokens_one, class_tokens_one], axis=0)
            tokens_two = paddle.concat(x=[tokens_two, class_tokens_two], axis=0)

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_data_root=args.instance_data_dir,
        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
        class_num=args.num_class_images,
        height=args.height,
        width=args.width,
        center_crop=args.center_crop,
        interpolation="bilinear",
        random_flip=args.random_flip,
    )

    def collate_fn(examples, with_prior_preservation=False):
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = paddle.stack(pixel_values).astype("float32")

        batch = {"pixel_values": pixel_values}

        return batch

    train_sampler = (
        DistributedBatchSampler(train_dataset, batch_size=args.train_batch_size, shuffle=True)
        if num_processes > 1
        else BatchSampler(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=args.dataloader_num_workers
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * num_processes
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        learning_rate=args.learning_rate,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    params_to_optimize = (
        unet_lora_parameters + text_lora_parameters_one + text_lora_parameters_two
        if args.train_text_encoder
        else unet_lora_parameters
    )
    # Optimizer creation
    optimizer = AdamW(
        learning_rate=lr_scheduler,
        parameters=params_to_optimize,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        weight_decay=args.adam_weight_decay,
        epsilon=args.adam_epsilon,
        grad_clip=nn.ClipGradByGlobalNorm(args.max_grad_norm) if args.max_grad_norm > 0 else None,
    )

    if num_processes > 1:
        unet = paddle.DataParallel(unet)
        if args.train_text_encoder:
            text_encoder_one = paddle.DataParallel(text_encoder_one)
            text_encoder_two = paddle.DataParallel(text_encoder_two)

    if is_main_process:
        logger.info("-----------  Configuration Arguments -----------")
        for arg, value in sorted(vars(args).items()):
            logger.info("%s: %s" % (arg, value))
        logger.info("------------------------------------------------")
        writer = get_report_to(args)

    # Train!
    total_batch_size = args.train_batch_size * num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not is_main_process)
    progress_bar.set_description("Train Steps")
    global_step = 0
    vae.eval()
    if args.train_text_encoder:
        text_encoder_one.train()
        text_encoder_two.train()
    else:
        text_encoder_one.eval()
        text_encoder_two.eval()

    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            text_encoder_two.train()
        for step, batch in enumerate(train_dataloader):
            # # Convert images to latent space
            # latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
            # latents = latents * vae.config.scaling_factor
            if args.pretrained_vae_model_name_or_path is None:
                pixel_values = batch["pixel_values"]
            else:
                pixel_values = batch["pixel_values"].cast(dtype=paddle_dtype)

            # Convert images to latent space
            model_input = vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * vae.config.scaling_factor
            if args.pretrained_vae_model_name_or_path is None:
                model_input = model_input.cast(paddle_dtype)

            # Sample noise that we'll add to the latents
            noise = paddle.randn(model_input.shape, dtype=model_input.dtype)
            if args.noise_offset:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += args.noise_offset * paddle.randn(
                    (model_input.shape[0], model_input.shape[1], 1, 1), dtype=model_input.dtype
                )
            bsz = model_input.shape[0]
            # Sample a random timestep for each image
            # timesteps = paddle.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,)).cast("int64")
            timesteps = paddle.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,)).cast("int64")

            # Add noise to the model input according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            if num_processes > 1 and (
                args.gradient_checkpointing or ((step + 1) % args.gradient_accumulation_steps != 0)
            ):
                # grad acc, no_sync when (step + 1) % args.gradient_accumulation_steps != 0:
                # gradient_checkpointing, no_sync every where
                # gradient_checkpointing + grad_acc, no_sync every where
                unet_ctx_manager = unet.no_sync()
            else:
                unet_ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()

            # if use_attention_mask:
            #     attention_mask = (batch["input_ids"] != tokenizer.pad_token_id).cast("int64")
            # else:
            #     attention_mask = None

            # # Get the text embedding for conditioning
            # # encoder_hidden_states = text_encoder(batch["input_ids"], attention_mask=attention_mask)[0]
            # if args.pre_compute_text_embeddings:
            #     encoder_hidden_states = batch["input_ids"]
            # else:
            #     encoder_hidden_states = encode_prompt(
            #         text_encoder,
            #         batch["input_ids"],
            #         batch["attention_mask"],
            #         text_encoder_use_attention_mask=args.text_encoder_use_attention_mask,
            #     )
            # if args.class_labels_conditioning == "timesteps":
            #     class_labels = timesteps
            # else:
            #     class_labels = None

            with unet_ctx_manager:
                # Calculate the elements to repeat depending on the use of prior-preservation.
                elems_to_repeat = bsz // 2 if args.with_prior_preservation else bsz

                # Predict the noise residual
                if not args.train_text_encoder:
                    unet_added_conditions = {
                        "time_ids": add_time_ids.tile(repeat_times=[elems_to_repeat, 1]),
                        "text_embeds": unet_add_text_embeds.tile(repeat_times=[elems_to_repeat, 1]),
                    }
                    prompt_embeds = prompt_embeds.tile(repeat_times=[elems_to_repeat, 1, 1])
                    model_pred = unet(
                        noisy_model_input, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions
                    ).sample
                else:
                    unet_added_conditions = {"time_ids": add_time_ids.tile(repeat_times=[elems_to_repeat, 1])}
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders=[text_encoder_one, text_encoder_two],
                        tokenizers=None,
                        prompt=None,
                        text_input_ids_list=[tokens_one, tokens_two],
                        args=args,
                    )
                    unet_added_conditions.update(
                        {"text_embeds": pooled_prompt_embeds.tile(repeat_times=[elems_to_repeat, 1])}
                    )
                    prompt_embeds = prompt_embeds.tile(repeat_times=[elems_to_repeat, 1, 1])
                    model_pred = unet(
                        noisy_model_input, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions
                    ).sample

                # Predict the noise residual / sample
                with paddle.amp.auto_cast(
                    enable=args.mixed_precision in ["bf16", "fp16"],
                    level=args.fp16_opt_level,
                    custom_black_list=["reduce_sum", "c_softmax_with_cross_entropy"],
                    custom_white_list=["lookup_table", "lookup_table_v2"]
                    if args.fp16_opt_level == "O2"
                    else ["layer_norm"],
                    dtype="bfloat16" if args.mixed_precision == "bf16" else "float16",
                ):
                    model_pred = unet(
                        noisy_model_input, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions
                    ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = model_pred.chunk(2, axis=0)
                    target, target_prior = target.chunk(2, axis=0)

                    # Compute instance loss
                    # loss = F.mse_loss(model_pred, target, reduction="mean")
                    if args.snr_gamma is None:
                        loss = F.mse_loss(
                            model_pred.cast("float32"),
                            target.cast("float32"),
                            reduction="mean",
                        )
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(timesteps)
                        mse_loss_weights = (
                            paddle.stack([snr, args.snr_gamma * paddle.ones_like(timesteps)], axis=1,).min(
                                1
                            )[0]
                            / snr
                        )
                        # We first calculate the original loss. Then we mean over the non-batch dimensions and
                        # rebalance the sample-wise losses with their respective loss weights.
                        # Finally, we take the mean of the rebalanced loss.
                        loss = F.mse_loss(
                            model_pred.cast("float32"),
                            target.cast("float32"),
                            reduction="none",
                        )
                        loss = loss.mean(axis=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                    # Compute prior loss
                    # prior_loss = F.mse_loss(model_pred_prior, target_prior, reduction="mean")
                    if args.snr_gamma is None:
                        prior_loss = F.mse_loss(
                            model_pred_prior.cast("float32"),
                            target_prior.cast("float32"),
                            reduction="mean",
                        )
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(timesteps)
                        mse_loss_weights = (
                            paddle.stack([snr, args.snr_gamma * paddle.ones_like(timesteps)], axis=1,).min(
                                1
                            )[0]
                            / snr
                        )
                        # We first calculate the original loss. Then we mean over the non-batch dimensions and
                        # rebalance the sample-wise losses with their respective loss weights.
                        # Finally, we take the mean of the rebalanced loss.
                        prior_loss = F.mse_loss(
                            model_pred_prior.cast("float32"),
                            target_prior.cast("float32"),
                            reduction="none",
                        )
                        prior_loss = prior_loss.mean(axis=list(range(1, len(loss.shape)))) * mse_loss_weights
                        prior_loss = prior_loss.mean()

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    # loss = F.mse_loss(model_pred, target, reduction="mean")
                    if args.snr_gamma is None:
                        loss = F.mse_loss(
                            model_pred.cast("float32"),
                            target.cast("float32"),
                            reduction="mean",
                        )
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = compute_snr(timesteps)
                        mse_loss_weights = (
                            paddle.stack([snr, args.snr_gamma * paddle.ones_like(timesteps)], axis=1,).min(
                                1
                            )[0]
                            / snr
                        )
                        # We first calculate the original loss. Then we mean over the non-batch dimensions and
                        # rebalance the sample-wise losses with their respective loss weights.
                        # Finally, we take the mean of the rebalanced loss.
                        loss = F.mse_loss(
                            model_pred.cast("float32"),
                            target.cast("float32"),
                            reduction="none",
                        )
                        loss = loss.mean(axis=list(range(1, len(loss.shape)))) * mse_loss_weights
                        loss = loss.mean()

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if do_grad_scaling:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if num_processes > 1 and args.gradient_checkpointing:
                    fused_allreduce_gradients(params_to_optimize, None)

                if do_grad_scaling:
                    scale_before = scaler._scale.numpy()
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = scaler._scale.numpy()
                    optimizer_was_run = not scaler._cache_founf_inf
                    if not optimizer_was_run:
                        logger.warning(
                            f"optimizer not run, scale_before: {scale_before[0]}, scale_after: {scale_after[0]}"
                        )
                else:
                    optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                progress_bar.update(1)
                global_step += 1
                step_loss = loss.item() * args.gradient_accumulation_steps
                logs = {
                    "epoch": str(epoch).zfill(4),
                    "train_loss": round(step_loss, 10),
                    "lr": lr_scheduler.get_lr(),
                }
                progress_bar.set_postfix(**logs)

                if is_main_process:
                    for name, val in logs.items():
                        if name == "epoch":
                            continue
                        writer.add_scalar(f"train/{name}", val, global_step)

                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # We combine the text encoder and UNet LoRA parameters with a simple
                        # custom logic. So, use `LoraLoaderMixin.save_lora_weights()`.

                        # save_model_func(models=[unet_lora_parameters, text_lora_parameters], weights=[unet_lora_parameters + text_lora_parameters if args.train_text_encoder else unet_lora_parameters], output_dir=save_path)
                        # LoraLoaderMixin.save_lora_weights(
                        #     save_directory=save_path,
                        #     unet_lora_layers=unet_attn_processors_state_dict(unet),
                        #     text_encoder_lora_layers=unet_attn_processors_state_dict(text_encoder) if args.train_text_encoder else None,
                        # )

                        if args.train_text_encoder:
                            text_encoder_one = unwrap_model(text_encoder_one)
                            text_encoder_lora_layers = text_encoder_lora_state_dict(
                                text_encoder_one.to(paddle.float32)
                            )
                            text_encoder_two = unwrap_model(text_encoder_two)
                            text_encoder_2_lora_layers = text_encoder_lora_state_dict(
                                text_encoder_two.to(paddle.float32)
                            )
                        else:
                            text_encoder_lora_layers = None
                            text_encoder_2_lora_layers = None

                        StableDiffusionXLPipeline.save_lora_weights(
                            save_directory=save_path,
                            # unet_lora_layers=unet_lora_layers,
                            unet_lora_layers=unet_attn_processors_state_dict(unet),
                            text_encoder_lora_layers=text_encoder_lora_layers,
                            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
                        )

                        logger.info(f"Saved lora weights to {save_path}")

                if global_step >= args.max_train_steps:
                    break

        if is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                # # create pipeline
                if not args.train_text_encoder:
                    text_encoder_one = text_encoder_cls_one.from_pretrained(
                        url_or_path_join(args.pretrained_model_name_or_path, "text_encoder")
                    )
                    text_encoder_two = text_encoder_cls_two.from_pretrained(
                        url_or_path_join(args.pretrained_model_name_or_path, "text_encoder_2")
                    )
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    text_encoder=unwrap_model(text_encoder_one),
                    text_encoder_2=unwrap_model(text_encoder_two),
                    unet=unwrap_model(unet),
                    paddle_dtype=paddle_dtype,
                )

                # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
                scheduler_args = {}

                if "variance_type" in pipeline.scheduler.config:
                    variance_type = pipeline.scheduler.config.variance_type

                    if variance_type in ["learned", "learned_range"]:
                        variance_type = "fixed_small"

                    scheduler_args["variance_type"] = variance_type

                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config, **scheduler_args
                )

                pipeline.set_progress_bar_config(disable=True)

                # run inference
                # generator = paddle.Generator().manual_seed(args.seed) if args.seed else None
                # images = [
                #     pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                #     for _ in range(args.num_validation_images)
                # ]
                generator = paddle.Generator().manual_seed(args.seed) if args.seed else None
                pipeline_args = {"prompt": args.validation_prompt}

                # if args.validation_images is None:
                #     images = []
                #     for _ in range(args.num_validation_images):
                #         with paddle.amp.auto_cast():
                #             image = pipeline(**pipeline_args, generator=generator).images[0]
                #             images.append(image)
                # else:
                #     images = []
                #     for image in args.validation_images:
                #         image = Image.open(image)
                #         with paddle.amp.auto_cast():
                #             image = pipeline(**pipeline_args, image=image, generator=generator).images[0]
                #         images.append(image)

                # with paddle.amp.auto_cast():
                images = [
                    pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)
                ]

                np_images = np.stack([np.asarray(img) for img in images])
                if args.report_to == "tensorboard":
                    writer.add_images("test", np_images, epoch, dataformats="NHWC")
                else:
                    writer.add_image("test", np_images, epoch, dataformats="NHWC")

                del pipeline
                gc.collect()

    # Save the lora layers
    if is_main_process:
        unet = unwrap_model(unet)
        unet = unet.to(paddle.float32)
        unet_lora_layers = unet_attn_processors_state_dict(unet)

        if args.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one)
            text_encoder_lora_layers = text_encoder_lora_state_dict(text_encoder_one.to(paddle.float32))
            text_encoder_two = unwrap_model(text_encoder_two)
            text_encoder_2_lora_layers = text_encoder_lora_state_dict(text_encoder_two.to(paddle.float32))
        else:
            text_encoder_lora_layers = None
            text_encoder_2_lora_layers = None

        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )

        # Final inference
        # Load previous pipeline
        vae = AutoencoderKL.from_pretrained(
            url_or_path_join(vae_path, "vae") if args.pretrained_vae_model_name_or_path is None else None,
            paddle_dtype=paddle_dtype,
        )
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path, vae=vae, paddle_dtype=paddle_dtype
        )

        # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
        scheduler_args = {}

        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)

        # load attention processors
        pipeline.load_lora_weights(args.output_dir)

        # run inference
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            generator = paddle.Generator().manual_seed(args.seed) if args.seed else None
            images = [
                pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                for _ in range(args.num_validation_images)
            ]
            np_images = np.stack([np.asarray(img) for img in images])

            if args.report_to == "tensorboard":
                writer.add_images("test", np_images, epoch, dataformats="NHWC")
            else:
                writer.add_image("test", np_images, epoch, dataformats="NHWC")

        writer.close()

        # logic to push to HF Hub
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_id = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_id = args.hub_model_id

            _retry(
                create_repo,
                func_kwargs={"repo_id": repo_id, "exist_ok": True, "token": args.hub_token},
                base_wait_time=1.0,
                max_retries=5,
                max_wait_time=10.0,
            )

            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                prompt=args.instance_prompt,
                repo_folder=args.output_dir,
                vae_path=args.pretrained_vae_model_name_or_path,
            )
            # Upload model
            logger.info(f"Pushing to {repo_id}")
            _retry(
                upload_folder,
                func_kwargs={
                    "repo_id": repo_id,
                    "repo_type": "model",
                    "folder_path": args.output_dir,
                    "commit_message": "End of training",
                    "token": args.hub_token,
                    "ignore_patterns": ["checkpoint-*/*", "logs/*"],
                },
                base_wait_time=1.0,
                max_retries=5,
                max_wait_time=20.0,
            )


if __name__ == "__main__":
    main()
