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


def get_backbone_type(root_config: dict, nested_config: dict = None):
    if nested_config is None:
        nested_config = root_config
    return nested_config.get(
        "backbone_type", root_config.get("backbone_type", root_config.get("diff_decoder_type", "wavenet"))
    )


def get_backbone_args(config: dict, backbone_type: str):
    args = config.get("backbone_args")
    if args is not None:
        return args
    elif backbone_type == "wavenet":
        return {
            "num_layers": config.get("residual_layers"),
            "num_channels": config.get("residual_channels"),
            "dilation_cycle_length": config.get("dilation_cycle_length"),
        }
    else:
        return None
