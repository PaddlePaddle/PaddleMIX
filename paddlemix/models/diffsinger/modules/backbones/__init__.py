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

import paddle
from paddlemix.models.diffsinger.modules.backbones.lynxnet import LYNXNet
from paddlemix.models.diffsinger.modules.backbones.wavenet import WaveNet
from paddlemix.models.diffsinger.utils import filter_kwargs

BACKBONES = {"wavenet": WaveNet, "lynxnet": LYNXNet}


def build_backbone(out_dims: int, num_feats: int, backbone_type: str, backbone_args: dict) -> paddle.nn.Layer:
    backbone = BACKBONES[backbone_type]
    kwargs = filter_kwargs(backbone_args, backbone)
    return BACKBONES[backbone_type](out_dims, num_feats, **kwargs)
