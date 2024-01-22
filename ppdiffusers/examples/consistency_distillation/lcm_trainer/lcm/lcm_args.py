# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

from paddlenlp.trainer import TrainingArguments

__all__ = [
    "LCMTrainingArguments",
    "LCMModelArguments",
    "LCMDataArguments",
]

from ppdiffusers.utils import str2bool


@dataclass
class LCMTrainingArguments(TrainingArguments):
    image_logging_steps: int = field(default=1000, metadata={"help": "Log image every X steps."})
    benchmark: bool = field(
        default=False,
        metadata={"help": "Whether or not run benchmark."},
    )
    profiler_options: Optional[str] = field(
        default=None,
        metadata={"help": "profiler_options."},
    )
    report_to: Optional[List[str]] = field(
        default_factory=lambda: ["custom_visualdl"],
        metadata={"help": "The list of integrations to report the results and logs to."},
    )
    resolution: int = field(
        default=512,
        metadata={
            "help": "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution."
        },
    )
    enable_xformers_memory_efficient_attention: bool = field(
        default=True, metadata={"help": "enable_xformers_memory_efficient_attention."}
    )
    only_save_updated_model: bool = field(
        default=True, metadata={"help": "Whether or not save only_save_updated_model"}
    )

    def __post_init__(self):
        super().__post_init__()
        self.image_logging_steps = (
            (math.ceil(self.image_logging_steps / self.logging_steps) * self.logging_steps)
            if self.image_logging_steps > 0
            else -1
        )
        self.enable_xformers_memory_efficient_attention = (
            str2bool(os.getenv("FLAG_XFORMERS", "False")) or self.enable_xformers_memory_efficient_attention
        )
        self.recompute = str2bool(os.getenv("FLAG_RECOMPUTE", "False")) or self.recompute
        self.benchmark = str2bool(os.getenv("FLAG_BENCHMARK", "False")) or self.benchmark


@dataclass
class LCMModelArguments:
    vae_name_or_path: Optional[str] = field(default=None, metadata={"help": "vae_name_or_path"})
    text_encoder_name_or_path: Optional[str] = field(default=None, metadata={"help": "text_encoder_name_or_path"})
    # for sdxl
    text_encoder_2_name_or_path: Optional[str] = field(default=None, metadata={"help": "text_encoder_2_name_or_path"})

    teacher_unet_name_or_path: Optional[str] = field(default=None, metadata={"help": "unet_name_or_path"})
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as pretrained_model_name_or_path"},
    )
    # for sdxl
    tokenizer_2_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer_2 name or path if not the same as pretrained_model_name_or_path"},
    )
    pretrained_model_name_or_path: str = field(
        default="runwayml/stable-diffusion-v1-5",
        metadata={"help": "Path to pretrained model or model, when we want to resume training."},
    )
    model_max_length: int = field(default=77, metadata={"help": "Pretrained tokenizer model_max_length"})
    num_inference_steps: int = field(default=4, metadata={"help": "num_inference_steps"})

    vae_encode_batch_size: int = field(default=None, metadata={"help": "vae_encode_batch_size"})
    # ----Latent Consistency Distillation (LCD) Specific Arguments----
    w_min: float = field(
        default=None,
        metadata={
            "help": "The minimum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        },
    )
    w_max: float = field(
        default=15.0,
        metadata={
            "help": "The maximum guidance scale value for guidance scale sampling. Note that we are using the Imagen CFG"
            " formulation rather than the LCM formulation, which means all guidance scales have 1 added to them as"
            " compared to the original paper."
        },
    )
    num_ddim_timesteps: int = field(default=50, metadata={"help": "The number of timesteps to use for DDIM sampling"})
    loss_type: str = field(
        default="l2", metadata={"help": '["l2", "huber"] The type of loss to use for the LCD loss.'}
    )
    huber_c: float = field(
        default=0.001, metadata={"help": "The huber loss parameter. Only used if `--loss_type=huber`."}
    )
    timestep_scaling_factor: float = field(
        default=10.0,
        metadata={
            "help": "The multiplicative timestep scaling factor used when calculating the boundary scalings for LCM. The"
            " higher the scaling is, the lower the approximation error, but the default value of 10.0 should typically"
            " suffice."
        },
    )

    # --------------------- NON LORA --------------------------
    unet_time_cond_proj_dim: int = field(
        default=256,
        metadata={
            "help": "The dimension of the guidance scale embedding in the U-Net, which will be used if the teacher U-Net"
            " does not have `time_cond_proj_dim` set. Only used if `--is_lora=False`."
        },
    )
    ema_decay: float = field(
        default=0.95,
        metadata={"help": "The exponential moving average (EMA) rate or decay factor."},
    )
    # --------------------- LORA --------------------------
    is_lora: bool = field(default=True, metadata={"help": "Whether or not lora model"})
    lora_rank: int = field(
        default=64, metadata={"help": "The rank of the LoRA projection matrix. Only used if `--is_lora=True`."}
    )
    # --------------------- SDXL --------------------------
    is_sdxl: bool = field(default=False, metadata={"help": "Whether or not sdxl model"})
    use_fix_crop_and_size: bool = field(
        default=True,
        metadata={"help": "Whether or not to use the fixed crop and size for the teacher model"},
    )
    center_crop: bool = field(
        default=False,
        metadata={
            "help": "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        },
    )
    random_flip: bool = field(
        default=False,
        metadata={"help": "whether to randomly flip images horizontally"},
    )

    def __post_init__(self):
        if self.is_sdxl:
            default_w_min = 3.0
            default_vae_encode_batch_size = 8
        else:
            default_w_min = 5.0
            default_vae_encode_batch_size = 32
        self.w_min = self.w_min or default_w_min
        self.vae_encode_batch_size = self.vae_encode_batch_size or default_vae_encode_batch_size


@dataclass
class LCMDataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """

    file_list: str = field(
        default="./data/filelist/train.filelist.list",
        metadata={"help": "The name of the file_list."},
    )
    num_records: int = field(default=10000000, metadata={"help": "num_records"})
    buffer_size: int = field(
        default=100,
        metadata={"help": "Buffer size"},
    )
    shuffle_every_n_samples: int = field(
        default=5,
        metadata={"help": "shuffle_every_n_samples."},
    )
    interpolation: str = field(
        default="lanczos",  # bilinear, bicubic, lanczos
        metadata={"help": "interpolation method"},
    )
    proportion_empty_prompts: float = field(
        default=0.0,
        metadata={
            "help": "Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement)."
        },
    )
