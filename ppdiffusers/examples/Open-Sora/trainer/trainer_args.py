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

    vae_micro_batch_size: int = field(
        default=1,
        metadata={"help": "vae_micro_batch_size"},
    )

    vae_model_path: Optional[str] = field(
        default="stabilityai/sd-vae-ft-ema",
        metadata={"help": "vae_model_path"},
    )

    text_encoder_model_max_length: int = field(
        default=200,
        metadata={"help": "text_encoder_model_max_length"},
    )

    text_encoder_path: Optional[str] = field(
        default="DeepFloyd/t5-v1_1-xxl",
        metadata={"help": "text_encoder_path"},
    )

    stdit2_pretrained_path: Optional[str] = field(
        default="hpcai-tech/OpenSora-STDiT-v2-stage3",
        metadata={"help": "stdit2_pretrained_path"},
    )

    timestep_respacing: Optional[str] = field(
        default="",
        metadata={"help": "timestep_respacing od IDDPM"},
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """

    train_height: int = field(
        default=None,
        metadata={"help": "train dataset height"},
    )

    train_width: int = field(
        default=None,
        metadata={"help": "train dataset width"},
    )

    meta_paths: str = field(
        default="./OpenSoraData/meta/meta_info.csv",
        metadata={"help": "The path of metadata"},
    )

    num_frames: int = field(
        default=None,
        metadata={"help": "train dataset num_frames"},
    )

    frame_interval: int = field(
        default=3,
        metadata={"help": "train dataset frame_interval"},
    )

    transform_name: str = field(
        default="resize_crop",
        metadata={"help": "transform_name when building dataset"},
    )
