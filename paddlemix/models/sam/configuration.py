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
""" Sam model configuration"""
import os
from typing import Union

from paddlenlp.transformers.configuration_utils import PretrainedConfig

from paddlemix.utils.log import logger

__all__ = ["SamConfig"]


class SamConfig(PretrainedConfig):

    model_type = "Sam"

    def __init__(
        self,
        modelname="Sam",
        prompt_embed_dim=256,
        image_size=1024,
        vit_patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        input_type=None,
        **kwargs,
    ):
        super().__init__()
        self.modelname = modelname
        self.prompt_embed_dim = prompt_embed_dim
        self.image_size = image_size
        self.vit_patch_size = vit_patch_size
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_depth = encoder_depth
        self.encoder_num_heads = encoder_num_heads
        self.encoder_global_attn_indexes = encoder_global_attn_indexes
        self.input_type = input_type
        self.pixel_mean = ([123.675, 116.28, 103.53],)
        self.pixel_std = [58.395, 57.12, 57.375]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)
