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

import re

import paddle


class IdentityMap(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(paddle.nn.Layer):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = paddle.nn.LayerNorm(normalized_shape=channels)
        self.proj = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=channels, out_features=channels),
            paddle.nn.GELU(),
            paddle.nn.Linear(in_features=channels, out_features=channels),
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")
    if projector_type == "linear":
        return paddle.nn.Linear(in_features=config.mm_hidden_size, out_features=config.hidden_size)
    mlp_gelu_match = re.match("^mlp(\\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [paddle.nn.Linear(in_features=config.mm_hidden_size, out_features=config.hidden_size)]

        for _ in range(1, mlp_depth):
            modules.append(paddle.nn.GELU())
            modules.append(paddle.nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))

        return paddle.nn.Sequential(*modules)
    if projector_type == "identity":
        return IdentityMap()
    raise ValueError(f"Unknown projector type: {projector_type}")
