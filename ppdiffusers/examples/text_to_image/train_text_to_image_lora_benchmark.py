# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import math
import os

os.environ["USE_PEFT_BACKEND"] = "False"
os.environ["FLAG_USE_OLD_RECOMPUTE"] = "True"
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from datasets import DatasetDict, concatenate_datasets
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    fused_allreduce_gradients,
)
from paddle.io import BatchSampler, DataLoader, DistributedBatchSampler
from paddle.optimizer import AdamW
from paddle.vision import BaseTransform, transforms
from paddlenlp.trainer import set_seed
from paddlenlp.utils.log import logger
from tqdm.auto import tqdm

from ppdiffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
    is_ppxformers_available,
)
from ppdiffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from ppdiffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_5,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_5,
)
from ppdiffusers.optimization import get_scheduler
from ppdiffusers.training_utils import freeze_params, main_process_first, unwrap_model
from ppdiffusers.transformers import AutoTokenizer, PretrainedConfig
from ppdiffusers.utils import str2bool

TEXT_ENCODER_ATTN_MODULE = ".self_attn"
if str2bool(os.getenv("FLAG_FUSED_LINEAR", "False")):
    paddle.nn.Linear = paddle.incubate.nn.FusedLinear


class AverageStatistical(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_cnt = 0
        self.time = 0

    def record(self, val, cnt=1):
        self.time += val
        self.total_cnt += cnt

    def get_average(self):
        if self.total_cnt == 0:
            return 0

        return self.time / self.total_cnt

    def get_average_per_sec(self):
        if self.time == 0.0:
            return 0.0

        return float(self.total_cnt) / self.time

    def get_total_cnt(self):
        return self.total_cnt

    def get_total_time(self):
        return self.time


def url_or_path_join(*path_list):
    return os.path.join(*path_list) if os.path.isdir(os.path.join(*path_list)) else "/".join(path_list)


def save_model_card(repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-ppdiffusers
- text-to-image
- ppdiffusers
- lora
inference: false
---
    """

    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""

    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str):
    try:
        text_encoder_config = PretrainedConfig.from_pretrained(
            url_or_path_join(pretrained_model_name_or_path, "text_encoder")
        )
        model_class = text_encoder_config.architectures[0]
    except Exception:
        model_class = "LDMBertModel"
    if model_class == "CLIPTextModel":
        from ppdiffusers.transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from ppdiffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    elif model_class == "BertModel":
        from ppdiffusers.transformers import BertModel

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
    parser = argparse.ArgumentParser(description="Simple example of a training text to image lora script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="./CompVis-stable-diffusion-v1-4-paddle-init",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="pokemon-blip-captions",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is sampled during training for inference.",
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
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
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
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The rank of lora linear.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="The rank of lora linear.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
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
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="logging_steps.",
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
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
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
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
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
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="fp16_opt_level.",
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None:
        raise ValueError("Need either a dataset name or a training folder.")
    args.logging_dir = os.path.join(args.output_dir, args.logging_dir)
    if args.height is None or args.width is None and args.resolution is not None:
        args.height = args.width = args.resolution

    if args.rank is not None:
        args.lora_rank = args.rank
    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
    "pokemon-blip-captions": ("image", "text"),
}


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

    # Handle the repository creation
    if is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(url_or_path_join(args.pretrained_model_name_or_path, "tokenizer"))

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        url_or_path_join(args.pretrained_model_name_or_path, "text_encoder")
    )
    text_config = text_encoder.config if isinstance(text_encoder.config, dict) else text_encoder.config.to_dict()
    if text_config.get("use_attention_mask", None) is not None and text_config["use_attention_mask"]:
        use_attention_mask = True
    else:
        use_attention_mask = False
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )

    # We only train the additional adapter LoRA layers
    freeze_params(vae.parameters())
    freeze_params(text_encoder.parameters())
    freeze_params(unet.parameters())

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
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
                models=[text_encoder],
                level="O2",
                dtype=weight_dtype,
            )
        else:
            nn.Layer._to_impl(text_encoder, dtype=weight_dtype, floating_only=True)
    if args.enable_xformers_memory_efficient_attention and is_ppxformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
            vae.enable_xformers_memory_efficient_attention()
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

        if isinstance(attn_processor, AttnProcessor):
            lora_attn_processor_class = LoRAAttnProcessor
        elif isinstance(attn_processor, AttnProcessor2_5):
            lora_attn_processor_class = LoRAAttnProcessor2_5
        else:
            raise ValueError(f"Unknown attention processor type: {attn_processor.__class__.__name__}")

        unet_lora_attn_procs[name] = lora_attn_processor_class(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=args.lora_rank,
        )

    unet.set_attn_processor(unet_lora_attn_procs)
    unet_lora_layers = AttnProcsLayers(unet.attn_processors)

    # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
    # So, instead, we monkey-patch the forward calls of its attention-blocks. For this,
    # we first load a dummy pipeline with the text encoder and then do the monkey-patching.
    text_encoder_lora_layers = None
    if args.train_text_encoder:
        text_lora_attn_procs = {}
        for name, module in text_encoder.named_sublayers(include_self=True):
            if name.endswith(TEXT_ENCODER_ATTN_MODULE):
                text_lora_attn_procs[name] = LoRAAttnProcessor(
                    hidden_size=module.out_proj.weight.shape[1],
                    cross_attention_dim=None,
                    rank=args.lora_rank,
                )
        text_encoder_lora_layers = AttnProcsLayers(text_lora_attn_procs)
        temp_pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path, text_encoder=text_encoder
        )
        temp_pipeline._modify_text_encoder(text_lora_attn_procs)
        text_encoder = temp_pipeline.text_encoder
        del temp_pipeline

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

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # if args.benchmark:
    #     file_path = get_path_from_url_with_filelock(
    #         "https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/pokemon-blip-captions.tar.gz",
    #         PPDIFFUSERS_CACHE,
    #     )
    #     dataset = DatasetDict.load_from_disk(file_path)
    #     args.dataset_name = "lambdalabs/naruto-blip-captions"
    # else:
    #     if args.dataset_name is not None:
    #         # Downloading and loading a dataset from the hub.
    #         dataset = load_dataset(
    #             args.dataset_name,
    #             args.dataset_config_name,
    #             cache_dir=args.cache_dir,
    #         )
    #     else:
    #         data_files = {}
    #         if args.train_data_dir is not None:
    #             data_files["train"] = os.path.join(args.train_data_dir, "**")
    #         dataset = load_dataset(
    #             "imagefolder",
    #             data_files=data_files,
    #             cache_dir=args.cache_dir,
    #         )
    #         # See more about loading custom images at
    #         # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder
    dataset = DatasetDict.load_from_disk(args.dataset_name)

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="do_not_pad",
            truncation=True,
            return_attention_mask=False,
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.height, args.width), interpolation="bilinear"),
            transforms.CenterCrop((args.height, args.width))
            if args.center_crop
            else transforms.RandomCrop((args.height, args.width)),
            transforms.RandomHorizontalFlip() if args.random_flip else Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with main_process_first():
        repeat_dataset = concatenate_datasets([dataset["train"]] * 250)
        dataset["train"] = repeat_dataset
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = paddle.stack([example["pixel_values"] for example in examples]).cast("float32")
        input_ids = [example["input_ids"] for example in examples]
        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pd",
        ).input_ids
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_sampler = (
        DistributedBatchSampler(train_dataset, batch_size=args.train_batch_size, shuffle=False)
        if num_processes > 1
        else BatchSampler(train_dataset, batch_size=args.train_batch_size, shuffle=False)
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
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
        list(unet_lora_layers.parameters()) + list(text_encoder_lora_layers.parameters())
        if args.train_text_encoder
        else unet_lora_layers.parameters()
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
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            if hasattr(text_encoder, "gradient_checkpointing_enable"):
                text_encoder.gradient_checkpointing_enable()

    if num_processes > 1:
        unet = paddle.DataParallel(unet)
        if args.train_text_encoder:
            text_encoder = paddle.DataParallel(text_encoder)

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
    progress_bar = tqdm(range(args.max_train_steps), disable=True)
    progress_bar.set_description("Train Steps")
    global_step = 0
    vae.eval()
    if args.train_text_encoder:
        text_encoder.train()
    else:
        text_encoder.eval()

    # For benchmark
    reader_cost_avg = AverageStatistical()
    batch_cost_avg = AverageStatistical()
    batch_ips_avg = AverageStatistical()
    sample_per_cards = args.train_batch_size * args.gradient_accumulation_steps

    train_loss = 0.0

    batch_size_chunks = []
    if args.resolution == 512:
        if args.train_batch_size == 104:
            batch_size_chunks = [56, 48]
        elif args.train_batch_size == 96:
            batch_size_chunks = [56, 40]

    for epoch in range(args.num_train_epochs):
        unet.train()
        epoch_start = time.time()
        batch_start = time.time()
        for step, batch in enumerate(train_dataloader):
            train_reader_cost = time.time() - batch_start
            # Convert images to latent space
            if len(batch_size_chunks) > 1:
                latents_list = []
                for mini_pixel_values in batch["pixel_values"].split(batch_size_chunks, axis=0):
                    latents_list.append(vae.encode(mini_pixel_values).latent_dist.sample())
                latents = paddle.concat(latents_list, axis=0)
            else:
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = paddle.randn(latents.shape, dtype=latents.dtype)
            if args.noise_offset:
                # https://www.crosslabs.org/blog/diffusion-with-offset-noise
                noise += args.noise_offset * paddle.randn(
                    (latents.shape[0], latents.shape[1], 1, 1), dtype=latents.dtype
                )
            batch_size = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = paddle.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,)).cast("int64")

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            if num_processes > 1 and (
                args.gradient_checkpointing or ((step + 1) % args.gradient_accumulation_steps != 0)
            ):
                # grad acc, no_sync when (step + 1) % args.gradient_accumulation_steps != 0:
                # gradient_checkpointing, no_sync every where
                # gradient_checkpointing + grad_acc, no_sync every where
                unet_ctx_manager = unet.no_sync()
                if args.train_text_encoder:
                    text_encoder_ctx_manager = text_encoder.no_sync()
                else:
                    text_encoder_ctx_manager = (
                        contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()
                    )

            else:
                unet_ctx_manager = contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()
                text_encoder_ctx_manager = (
                    contextlib.nullcontext() if sys.version_info >= (3, 7) else contextlib.suppress()
                )

            if use_attention_mask:
                attention_mask = (batch["input_ids"] != tokenizer.pad_token_id).cast("int64")
            else:
                attention_mask = None

            with text_encoder_ctx_manager, unet_ctx_manager:
                with paddle.amp.auto_cast(
                    enable=args.mixed_precision in ["bf16", "fp16"] and args.train_text_encoder,
                    level=args.fp16_opt_level,
                    custom_black_list=["reduce_sum", "c_softmax_with_cross_entropy"],
                    custom_white_list=["lookup_table", "lookup_table_v2"]
                    if args.fp16_opt_level == "O2"
                    else ["layer_norm"],
                    dtype="bfloat16" if args.mixed_precision == "bf16" else "float16",
                ):
                    encoder_hidden_states = text_encoder(batch["input_ids"], attention_mask=attention_mask)[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                with paddle.amp.auto_cast(
                    enable=args.mixed_precision in ["bf16", "fp16"],
                    level=args.fp16_opt_level,
                    custom_black_list=["reduce_sum", "c_softmax_with_cross_entropy"],
                    custom_white_list=["lookup_table", "lookup_table_v2"]
                    if args.fp16_opt_level == "O2"
                    else ["layer_norm"],
                    dtype="bfloat16" if args.mixed_precision == "bf16" else "float16",
                ):
                    # Predict the noise residual and compute loss
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

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

                train_loss += loss.item()

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
                logs = {
                    "epoch": str(epoch).zfill(4),
                    "train_loss": round(train_loss, 10),
                    "lr": lr_scheduler.get_lr(),
                }
                progress_bar.set_postfix(**logs)

                if global_step % args.logging_steps == 0:
                    # log on each node
                    train_batch_cost = time.time() - batch_start
                    reader_cost_avg.record(train_reader_cost)
                    batch_cost_avg.record(train_batch_cost)
                    batch_ips_avg.record(train_batch_cost, sample_per_cards)
                    max_mem_reserved_msg = ""
                    max_mem_allocated_msg = ""
                    if paddle.device.is_compiled_with_cuda():
                        max_mem_reserved_msg = (
                            f"max_mem_reserved: {paddle.device.cuda.max_memory_reserved() // (1024 ** 2)} MB,"
                        )
                        max_mem_allocated_msg = (
                            f"max_mem_allocated: {paddle.device.cuda.max_memory_allocated() // (1024 ** 2)} MB"
                        )

                    logger.info(
                        "global step %d / %d, loss: %f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, avg_samples: %.5f, ips: %.5f sample/sec, %s %s"
                        % (
                            global_step,
                            args.max_train_steps,
                            train_loss,
                            reader_cost_avg.get_average(),
                            batch_cost_avg.get_average(),
                            sample_per_cards,
                            batch_ips_avg.get_average_per_sec(),
                            max_mem_reserved_msg,
                            max_mem_allocated_msg,
                        ),
                    )
                    reader_cost_avg.reset()
                    batch_cost_avg.reset()
                    batch_ips_avg.reset()
                else:
                    train_batch_cost = time.time() - batch_start
                    reader_cost_avg.record(train_reader_cost)
                    batch_cost_avg.record(train_batch_cost)
                    batch_ips_avg.record(train_batch_cost, sample_per_cards)

                train_loss = 0.0
                if global_step % args.checkpointing_steps == 0:
                    if is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # We combine the text encoder and UNet LoRA parameters with a simple
                        # custom logic. So, use `LoraLoaderMixin.save_lora_weights()`.
                        LoraLoaderMixin.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_layers,
                            text_encoder_lora_layers=text_encoder_lora_layers,
                        )
                        logger.info(f"Saved lora weights to {save_path}")

                if global_step >= args.max_train_steps:
                    break
            batch_start = time.time()

        train_epoch_cost = time.time() - epoch_start
        logger.info(
            "train epoch: %d, epoch_cost: %.5f s" % (epoch, train_epoch_cost),
        )
        if is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0 and epoch > 0:
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                # create pipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    text_encoder=unwrap_model(text_encoder),
                    safety_checker=None,
                    requires_safety_checker=False,
                )
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = paddle.Generator().manual_seed(args.seed) if args.seed else None
                images = [
                    pipeline(
                        args.validation_prompt,
                        num_inference_steps=30,
                        generator=generator,
                    ).images[0]
                    for _ in range(args.num_validation_images)
                ]
                np_images = np.stack([np.asarray(img) for img in images])

                if args.report_to == "tensorboard":
                    writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                else:
                    writer.add_image("validation", np_images, epoch, dataformats="NHWC")

                del pipeline
                gc.collect()
                if args.train_text_encoder:
                    text_encoder.train()
                unet.train()

    # Save the lora layers
    if is_main_process:
        LoraLoaderMixin.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
        )

        if args.push_to_hub:
            save_model_card(
                repo_name,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                prompt=args.instance_prompt,
                repo_folder=args.output_dir,
            )
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)
        # Final inference
        # Load previous pipeline
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        # load attention processors
        pipeline.load_lora_weights(args.output_dir)

        # run inference
        if args.validation_prompt and args.num_validation_images > 0:
            generator = paddle.Generator().manual_seed(args.seed) if args.seed else None
            images = [
                pipeline(args.validation_prompt, num_inference_steps=30, generator=generator).images[0]
                for _ in range(args.num_validation_images)
            ]
            np_images = np.stack([np.asarray(img) for img in images])

            if args.report_to == "tensorboard":
                writer.add_images("test", np_images, epoch, dataformats="NHWC")
            else:
                writer.add_image("test", np_images, epoch, dataformats="NHWC")

        writer.close()


if __name__ == "__main__":
    main()
