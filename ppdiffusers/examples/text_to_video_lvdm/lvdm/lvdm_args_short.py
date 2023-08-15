# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Parameters for initialization and training
    """

    # for initialization
    task_type: str = field(
        default="short",
        metadata={"help": "Type of train task. Should be one of ['short', 'text2video']"},
    )
    pretrained_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model, when we want to resume training."},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not use pretrained model name or path"},
    )
    vae_type: str = field(
        default="3d",
        metadata={"help": "Type of vae to use. Should be one of ['2d', '3d']"},
    )
    vae_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained vae name or path if not use pretrained model name or path"},
    )
    text_encoder_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained text encoder name or path if not use pretrained model name or path"},
    )
    text_encoder_config_file: Optional[str] = field(
        default=None,
        metadata={"help": "Text encoder config file if not use pretrained text encoder"},
    )
    is_text_encoder_trainable: bool = field(default=False, metadata={"help": "Whether or not use ema"})
    unet_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained unet name or path if not use pretrained model name or path"},
    )
    unet_config_file: Optional[str] = field(
        default=None, metadata={"help": "Unet config file if not use pretrained unet"}
    )
    scheduler_beta_start: Optional[float] = field(
        default=0.0015, metadata={"help": "Train or eval scheduler beta start"}
    )
    scheduler_beta_end: Optional[float] = field(default=0.0155, metadata={"help": "Train or eval scheduler beta end"})
    scheduler_num_train_timesteps: Optional[int] = field(
        default=1000,
        metadata={"help": "Train or eval scheduler number of train timesteps"},
    )
    eval_scheduler_num_inference_steps: Optional[int] = field(
        default=50, metadata={"help": "Eval scheduler number of inference timesteps"}
    )
    # for training
    use_ema: bool = field(default=False, metadata={"help": "Whether or not use ema"})
    enable_xformers_memory_efficient_attention: bool = field(
        default=False, metadata={"help": "enable xformers memory efficient attention"}
    )
    scale_factor: Optional[float] = field(
        default=0.33422927,
        metadata={"help": "The scale factor in the first stage encoding"},
    )
    shift_factor: Optional[float] = field(
        default=1.4606637,
        metadata={"help": "The shift factor in the first stage encoding"},
    )
    loss_type: str = field(
        default="l1",
        metadata={"help": "The loss type to use in training. Should be one of ['l2', 'l1']"},
    )
    # for alignmemnt
    latents_path: str = field(
        default=None,
        metadata={"help": "Path to latents, used for alignment"},
    )
    use_paddle_conv_init: bool = field(default=False, metadata={"help": "Whether or not use paddle conv2d init"})
    if_numpy_genarator_random_alignment: bool = field(
        default=False,
        metadata={"help": "Whether to align random using numpy generator"},
    )
    numpy_genarator_random_seed: Optional[int] = field(
        default=42, metadata={"help": "The random seed for numpy generator"}
    )
    set_seed_for_alignment: bool = field(default=False, metadata={"help": "Whether to set seed again for alignment"})


@dataclass
class TrainerArguments:
    """
    Parameters for logging
    """

    # for log
    image_logging_steps: Optional[int] = field(default=1000, metadata={"help": "Log image every X steps."})


@dataclass
class VideoFrameDatasetArguments:
    """
    Parameters for dataset
    """

    train_data_root: str = field(
        default="/root/data/lvdm/sky",
        metadata={"help": "The root path of train dataset files"},
    )
    train_subset_split: str = field(default="train", metadata={"help": "The train subset split"})
    eval_data_root: str = field(
        default="/root/data/lvdm/sky",
        metadata={"help": "The root path of validation dataset files"},
    )
    eval_subset_split: str = field(default="train", metadata={"help": "The validation subset split"})
    resolution: int = field(
        default=256,
        metadata={"help": "The resolution"},
    )
    video_length: int = field(
        default=16,
        metadata={"help": "The video length"},
    )
    dataset_name: str = field(default="sky", metadata={"help": "The dataset name"})
    spatial_transform: str = field(
        default="center_crop_resize",
        metadata={"help": "The spatial transform type to use"},
    )
    temporal_transform: str = field(default="rand_clips", metadata={"help": "The temporal transform type to use"})
    clip_step: int = field(
        default=None,
        metadata={"help": "The clip step"},
    )
