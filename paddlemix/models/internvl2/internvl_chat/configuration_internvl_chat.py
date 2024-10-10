# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import copy

from paddlenlp.transformers import LlamaConfig, Qwen2Config

# from ..phi3.configuration_phi3 import Phi3Config
from paddlenlp.transformers.configuration_utils import PretrainedConfig

from paddlemix.utils.log import logger

from ..internlm2.configuration_internlm2 import InternLM2Config
from .configuration_intern_vit import InternVisionConfig


class InternVLChatConfig(PretrainedConfig):
    model_type = "internvl_chat"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        pad2square=False,  #
        select_layer=-1,
        force_image_size=None,  #
        downsample_ratio=0.5,
        template=None,  #
        dynamic_image_size=False,  #
        use_thumbnail=False,  #
        ps_version="v1",  #
        min_dynamic_patch=1,
        max_dynamic_patch=6,  #
        **kwargs
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. Initializing the InternVisionConfig with default values.")

        if llm_config is None:
            llm_config = {}
            logger.info("llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).")

        self.vision_config = InternVisionConfig(**vision_config)
        if llm_config["architectures"][0] == "LlamaForCausalLM":
            self.llm_config = LlamaConfig(**llm_config)
        elif llm_config["architectures"][0] == "InternLM2ForCausalLM":
            self.llm_config = InternLM2Config(**llm_config)
        # elif llm_config['architectures'][0] == 'Phi3ForCausalLM':
        #     self.llm_config = Phi3Config(**llm_config)
        elif llm_config["architectures"][0] == "Qwen2ForCausalLM":
            self.llm_config = Qwen2Config(**llm_config)
        else:
            raise ValueError("Unsupported architecture: {}".format(llm_config["architectures"][0]))
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch

        logger.info(f"vision_select_layer: {self.select_layer}")
        logger.info(f"ps_version: {self.ps_version}")
        logger.info(f"min_dynamic_patch: {self.min_dynamic_patch}")
        logger.info(f"max_dynamic_patch: {self.max_dynamic_patch}")

    def to_dict(self, saving_file=False):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["llm_config"] = self.llm_config.to_dict()
        output["model_type"] = self.__class__.model_type
        output["use_backbone_lora"] = self.use_backbone_lora
        output["use_llm_lora"] = self.use_llm_lora
        output["pad2square"] = self.pad2square
        output["select_layer"] = self.select_layer
        output["force_image_size"] = self.force_image_size
        output["downsample_ratio"] = self.downsample_ratio
        output["template"] = self.template
        output["dynamic_image_size"] = self.dynamic_image_size
        output["use_thumbnail"] = self.use_thumbnail
        output["ps_version"] = self.ps_version
        output["min_dynamic_patch"] = self.min_dynamic_patch
        output["max_dynamic_patch"] = self.max_dynamic_patch

        return output
