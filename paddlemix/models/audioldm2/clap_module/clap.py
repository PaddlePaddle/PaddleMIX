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

import dataclasses
from dataclasses import dataclass, field

from .model import CLAP, CLAPAudioCfg, CLAPTextCfg


@dataclass
class CLAPConfig:
    embed_dim: int = 1024
    audio_cfg: CLAPAudioCfg = field(default_factory=CLAPAudioCfg())
    text_cfg: CLAPTextCfg = field(default_factory=CLAPTextCfg())


def create_clap_model(
    amodel_name: str,
    tmodel_name: str,
    pretrained: str = "",
    precision: str = "fp32",
    force_quick_gelu: bool = False,
    enable_fusion: bool = False,
    fusion_type: str = "None",
):
    pretrained = pretrained.lower()

    model_cfg = CLAPConfig()
    model_cfg = dataclasses.asdict(model_cfg)
    if force_quick_gelu:
        # override for use of QuickGELU on non-OpenAI transformer models
        model_cfg["quick_gelu"] = True

    model_cfg["text_cfg"]["model_type"] = tmodel_name
    model_cfg["enable_fusion"] = enable_fusion
    model_cfg["fusion_type"] = fusion_type
    model = CLAP(**model_cfg)

    return model, model_cfg
