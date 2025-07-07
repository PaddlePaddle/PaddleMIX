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

""" Global Response Normalization Module

Based on the GRN layer presented in
`ConvNeXt-V2 - Co-designing and Scaling ConvNets with Masked Autoencoders` - https://arxiv.org/abs/2301.00808

This implementation
* works for both NCHW and NHWC tensor layouts
* uses affine param names matching existing torch norm layers
* slightly improves eager mode performance via fused addcmul

Hacked together by / Copyright 2023 Ross Wightman
"""

import paddle
from paddle import nn as nn


class GlobalResponseNorm(nn.Layer):
    """Global Response Normalization layer"""

    def __init__(self, dim, eps=1e-6, channels_last=True):
        super().__init__()
        self.eps = eps
        if channels_last:
            self.spatial_dim = (1, 2)
            self.channel_dim = -1
            self.wb_shape = (1, 1, 1, -1)
        else:
            self.spatial_dim = (2, 3)
            self.channel_dim = 1
            self.wb_shape = (1, -1, 1, 1)

        out_0 = paddle.create_parameter(
            shape=paddle.zeros(shape=dim).shape,
            dtype=paddle.zeros(shape=dim).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=dim)),
        )
        out_0.stop_gradient = not True
        self.weight = out_0
        out_1 = paddle.create_parameter(
            shape=paddle.zeros(shape=dim).shape,
            dtype=paddle.zeros(shape=dim).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=dim)),
        )
        out_1.stop_gradient = not True
        self.bias = out_1

    def forward(self, x):
        x_g = x.norm(p=2, axis=self.spatial_dim, keepdim=True)
        x_n = x_g / (x_g.mean(axis=self.channel_dim, keepdim=True) + self.eps)
        return x + paddle.add(self.bias.reshape(self.wb_shape), self.weight.reshape(self.wb_shape), x * x_n)
