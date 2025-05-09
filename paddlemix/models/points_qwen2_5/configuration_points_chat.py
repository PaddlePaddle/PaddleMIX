# -*- coding: utf-8 -*-

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

# @Time    : 2025/4/19 下午8:37
# @Author  : zhaop-l(zhaopuzxjc@126.com)

import copy
from typing import Any, Dict

from paddlenlp.transformers import CLIPVisionConfig
from paddlenlp.transformers.configuration_utils import PretrainedConfig

from .configuration_llama import CustomLlamaConfig


class POINTSChatConfig(PretrainedConfig):
    model_type = "points_chat"
    is_composition = True
    """Configuration class for `POINTS`."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        vision_config = kwargs.pop("vision_config", None)
        llm_config = kwargs.pop("llm_config", None)
        if isinstance(vision_config, dict):
            self.vision_config = CLIPVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config
        if isinstance(llm_config, dict):
            self.llm_config = CustomLlamaConfig(**llm_config)
        else:
            self.llm_config = llm_config

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["llm_config"] = self.llm_config.to_dict()
        return output
