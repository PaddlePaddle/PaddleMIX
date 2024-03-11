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

import paddle
from paddlenlp.trainer import TrainingArguments

__all__ = [
    "IPAdapterTrainingArguments",
    "IPAdapterModelArguments",
    "IPAdapterDataArguments",
]

from ppdiffusers.utils import str2bool

if str2bool(os.getenv("FLAG_FUSED_LINEAR", "0")):
    paddle.nn.Linear = paddle.incubate.nn.FusedLinear


@dataclass
class IPAdapterTrainingArguments(TrainingArguments):
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

    def __post_init__(self):
        super().__post_init__()
        self.image_logging_steps = (
            (math.ceil(self.image_logging_steps / self.logging_steps) * self.logging_steps)
            if self.image_logging_steps > 0
            else -1
        )
        self.recompute = str2bool(os.getenv("FLAG_RECOMPUTE", "False")) or self.recompute
        self.benchmark = str2bool(os.getenv("FLAG_BENCHMARK", "False")) or self.benchmark


@dataclass
class IPAdapterModelArguments:
    vae_name_or_path: Optional[str] = field(default=None, metadata={"help": "vae_name_or_path"})
    text_encoder_name_or_path: Optional[str] = field(default=None, metadata={"help": "text_encoder_name_or_path"})
    unet_name_or_path: Optional[str] = field(default=None, metadata={"help": "unet_name_or_path"})
    image_encoder_name_or_path: Optional[str] = field(default=None, metadata={"help": "image_encoder_name_or_path"})

    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as pretrained_model_name_or_path"},
    )
    pretrained_model_name_or_path: str = field(
        default="CompVis/stable-diffusion-v1-4",
        metadata={"help": "Path to pretrained model or model, when we want to resume training."},
    )
    model_max_length: int = field(default=77, metadata={"help": "Pretrained tokenizer model_max_length"})
    prediction_type: str = field(
        default="epsilon",
        metadata={
            "help": "prediction_type, prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4 https://imagen.research.google/video/paper.pdf)"
        },
    )
    num_inference_steps: int = field(default=50, metadata={"help": "num_inference_steps"})

    noise_offset: float = field(default=0, metadata={"help": "The scale of noise offset."})
    snr_gamma: Optional[float] = field(
        default=None,
        metadata={
            "help": "SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556."
        },
    )
    input_perturbation: Optional[float] = field(
        default=0,
        metadata={"help": "The scale of input perturbation. Recommended 0.1."},
    )


@dataclass
class IPAdapterDataArguments:
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
        default="bilinear",
        metadata={"help": "interpolation method"},
    )

    t_drop_rate: float = field(
        default=0.05,
        metadata={"help": "t_drop_rate"},
    )
    i_drop_rate: float = field(
        default=0.05,
        metadata={"help": "i_drop_rate"},
    )
    ti_drop_rate: float = field(
        default=0.05,
        metadata={"help": "ti_drop_rate"},
    )
