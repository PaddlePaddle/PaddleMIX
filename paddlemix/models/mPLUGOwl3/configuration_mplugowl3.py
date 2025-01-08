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

from paddlemix.utils.log import logger

from .configuration_hyper_qwen2 import HyperQwen2Config
from .modeling_navit_siglip import SigLipVisionConfig


class mPLUGOwl3Config(HyperQwen2Config):
    model_type = "mplugowl3"
    keys_to_ignore_at_inference = ["past_key_values"]

    default_vision_config = {
        "hidden_size": 1152,
        "image_size": 378,
        "intermediate_size": 4304,
        "model_type": "siglip_vision_model",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
    }

    def __init__(
        self,
        use_cache=True,
        vision_config=None,
        **kwargs,
    ):
        self.use_cache = use_cache

        # same as HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit add tgt_sizes
        if vision_config is None:
            self.vision_config = SigLipVisionConfig(**self.default_vision_config)
            logger.info("vision_config is None, using default vision config")
        elif isinstance(vision_config, dict):
            self.vision_config = SigLipVisionConfig(**vision_config)
        elif isinstance(vision_config, SigLipVisionConfig):
            self.vision_config = vision_config
        self.image_size = 378
        self.patch_size = self.vision_config.patch_size

        super().__init__(**kwargs)
