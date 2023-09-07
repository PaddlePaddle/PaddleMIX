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
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    vae_name_or_path: Optional[str] = field(
        default="runwayml/stable-diffusion-v1-5/vae",
        metadata={"help": "pretrained_vae_name_or_path"},
    )
    unet_config_file: Optional[str] = field(
        default="./config/uvit_t2i_small_deep.json", metadata={"help": "unet_config_file"}
    )
    text_encoder_name: Optional[str] = field(
        default="runwayml/stable-diffusion-v1-5/text_encoder",
        metadata={"help": "Pretrained text_encoder name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default="runwayml/stable-diffusion-v1-5/tokenizer",
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    model_max_length: Optional[int] = field(default=77, metadata={"help": "Pretrained tokenizer model_max_length"})
    num_inference_steps: Optional[int] = field(default=50, metadata={"help": "num_inference_steps"})
    use_ema: bool = field(default=True, metadata={"help": "Whether or not use ema"})
    pretrained_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model, when we want to resume training."},
    )
    data_feature: bool = field(default=True, metadata={"help": "Whether train with data feature or image_text pair"})

    image_logging_steps: Optional[int] = field(default=5000, metadata={"help": "Log image every X steps."})
    enable_xformers_memory_efficient_attention: bool = field(
        default=True, metadata={"help": "enable_xformers_memory_efficient_attention."}
    )
    to_static: bool = field(default=False, metadata={"help": "Whether or not to_static"})
    prediction_type: Optional[str] = field(
        default="epsilon",
        metadata={
            "help": "prediction_type, prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4 https://imagen.research.google/video/paper.pdf)"
        },
    )
    benchmark: bool = field(
        default=False,
        metadata={"help": "Whether or not run benchmark."},
    )
    profiler_options: Optional[str] = field(
        default=None,
        metadata={"help": "profiler_options."},
    )
    noise_offset: Optional[int] = field(default=0, metadata={"help": "The scale of noise offset."})


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """

    file_list: str = field(
        default="./data/filelist/train.filelist.list",
        metadata={"help": "The name of the file_list."},
    )
    data_path: str = field(
        default="./data/coco256_features",
        metadata={"help": "The name of the file_list."},
    )
    resolution: int = field(
        default=256,
        metadata={
            "help": "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution."
        },
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
