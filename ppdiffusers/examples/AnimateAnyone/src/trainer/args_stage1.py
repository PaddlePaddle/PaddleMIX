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

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    benchmark: bool = field(
        default=False,
        metadata={"help": "Whether or not run benchmark."},
    )
    profiler_options: Optional[str] = field(
        default=None,
        metadata={"help": "profiler_options."},
    )

    num_train_timesteps: Optional[int] = field(
        default=1000,
        metadata={"help": "num_train_timesteps for scheduler"},
    )

    beta_start: Optional[float] = field(
        default=0.00085,
        metadata={"help": "beta_start for scheduler"},
    )
    beta_end: Optional[float] = field(
        default=0.012,
        metadata={"help": "beta_end  for scheduler"},
    )
    beta_schedule: Optional[str] = field(
        default="scaled_linear",
        metadata={"help": "beta_schedule for scheduler"},
    )
    steps_offset: Optional[int] = field(
        default=1,
        metadata={"help": "steps_offset for scheduler"},
    )
    clip_sample: Optional[bool] = field(
        default=False,
        metadata={"help": "clip_sample for scheduler"},
    )
    rescale_betas_zero_snr: Optional[bool] = field(
        default=True,
        metadata={"help": "rescale_betas_zero_snr for scheduler"},
    )
    timestep_spacing: Optional[str] = field(
        default="trailing",
        metadata={"help": "timestep_spacing for scheduler"},
    )
    prediction_type: Optional[str] = field(
        default="v_prediction",
        metadata={"help": "prediction_type for scheduler"},
    )
    beta_schedule: Optional[str] = field(
        default="scaled_linear",
        metadata={"help": "beta_schedule for scheduler"},
    )

    vae_model_path: Optional[str] = field(
        default="stabilityai/sd-vae-ft-mse",
        metadata={"help": "vae_model_path"},
    )

    image_encoder_path: Optional[str] = field(
        default="lambdalabs/sd-image-variations-diffusers",
        metadata={"help": "image_encoder_path"},
    )

    base_model_path: Optional[str] = field(
        default="lambdalabs/sd-image-variations-diffusers",
        metadata={"help": "base_model_path"},
    )
    denoising_unet_config_path: Optional[str] = field(
        default="./pretrained_weights/tsaiyue/AnimateAnyone_PD/config.json",
        metadata={"help": "denoising_unet_config_path"},
    )

    denoising_unet_base_model_path: Optional[str] = field(
        default="./pretrained_weights/tsaiyue/AnimateAnyone_PD/denoising_unet_initial4stage1.pdparams",
        metadata={"help": "denoising_unet_base_model_path"},
    )

    pose_guider_pretrain: Optional[bool] = field(
        default=True,
        metadata={"help": "pose_guider_pretrain or not"},
    )
    controlnet_openpose_path: Optional[str] = field(
        default="./pretrained_weights/tsaiyue/AnimateAnyone_PD/control_v11p_sd15_openpose.pdparams",
        metadata={"help": "controlnet_openpose_path"},
    )

    noise_offset: Optional[float] = field(default=0.05, metadata={"help": "The noise offset in training loop."})

    uncond_ratio: Optional[float] = field(default=0.1, metadata={"help": "uncond_ratio in training loop."})

    snr_gamma: Optional[float] = field(default=5.0, metadata={"help": "snr_gamma for calculating loss."})


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """

    train_width: int = field(
        default=768,
        metadata={"help": "train dataset width"},
    )

    train_height: int = field(
        default=768,
        metadata={"help": "train dataset height"},
    )

    meta_paths: str = field(
        default="./ubcNbili_data/meta_data/ubcNbili_meta.json",
        metadata={"help": "The path of metadata"},
    )

    sample_margin: int = field(
        default=30,
        metadata={"help": "train dataset sample margin"},
    )
