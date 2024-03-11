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

from argparse import Namespace

import paddle

from ..cogagent.cross_visual import GLU, PatchEmbedding, Transformer


class EVA2CLIPModel(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        vision_config = Namespace(**config.vision_config)
        self.patch_embedding = PatchEmbedding(vision_config)
        self.transformer = Transformer(vision_config)
        self.linear_proj = GLU(config, in_features=vision_config.hidden_size)
        out_2 = paddle.create_parameter(
            shape=paddle.zeros(shape=[1, 1, config.hidden_size]).shape,
            dtype=paddle.zeros(shape=[1, 1, config.hidden_size]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=[1, 1, config.hidden_size])),
        )
        out_2.stop_gradient = not True
        self.boi = out_2
        out_3 = paddle.create_parameter(
            shape=paddle.zeros(shape=[1, 1, config.hidden_size]).shape,
            dtype=paddle.zeros(shape=[1, 1, config.hidden_size]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=[1, 1, config.hidden_size])),
        )
        out_3.stop_gradient = not True
        self.eoi = out_3

    def forward(self, images):
        x = self.patch_embedding(images)
        x = self.transformer(x)
        x = x[:, 1:]
        x = self.linear_proj(x)
        boi = self.boi.expand(shape=[x.shape[0], -1, -1])
        eoi = self.eoi.expand(shape=[x.shape[0], -1, -1])
        x = paddle.concat(x=(boi, x, eoi), axis=1)
        return x
