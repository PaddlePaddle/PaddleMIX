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

from typing import Tuple

import paddle

from ppdiffusers.models.animate_anyone.motion_module import zero_module
from ppdiffusers.models.animate_anyone.resnet import InflatedConv3d
from ppdiffusers.models.modeling_utils import ContextManagers, ModelMixin


class PoseGuider(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
        weight_dtype=None,
    ):
        super().__init__()

        init_contexts = []
        if weight_dtype is not None:
            init_contexts.append(paddle.dtype_guard(weight_dtype))

        with ContextManagers(init_contexts):
            self.conv_in = InflatedConv3d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

            self.blocks = paddle.nn.LayerList(sublayers=[])

            for i in range(len(block_out_channels) - 1):
                channel_in = block_out_channels[i]
                channel_out = block_out_channels[i + 1]
                self.blocks.append(InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1))
                self.blocks.append(InflatedConv3d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

            self.conv_out = zero_module(
                InflatedConv3d(
                    block_out_channels[-1],
                    conditioning_embedding_channels,
                    kernel_size=3,
                    padding=1,
                )
            )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = paddle.nn.functional.silu(x=embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = paddle.nn.functional.silu(x=embedding)

        embedding = self.conv_out(embedding)

        return embedding
