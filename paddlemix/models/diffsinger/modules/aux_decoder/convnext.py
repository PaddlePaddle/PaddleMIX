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

import sys
from typing import Optional

import paddle
from paddlemix.models.diffsinger.utils import paddle_aux


class ConvNeXtBlock(paddle.nn.Layer):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
        self, dim: int, intermediate_dim: int, layer_scale_init_value: Optional[float] = None, drop_out: float = 0.0
    ):
        super().__init__()
        self.dwconv = paddle.nn.Conv1D(in_channels=dim, out_channels=dim, kernel_size=7, padding=3, groups=dim)
        self.norm = paddle.nn.LayerNorm(normalized_shape=dim, epsilon=1e-06)
        self.pwconv1 = paddle.nn.Linear(in_features=dim, out_features=intermediate_dim)
        self.act = paddle.nn.GELU()
        self.pwconv2 = paddle.nn.Linear(in_features=intermediate_dim, out_features=dim)
        self.gamma = (
            paddle.base.framework.EagerParamBase.from_tensor(
                tensor=layer_scale_init_value * paddle.ones(shape=dim), trainable=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = paddle.nn.Identity()
        self.dropout = paddle.nn.Dropout(p=drop_out) if drop_out > 0.0 else paddle.nn.Identity()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 1, 2))
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 1, 2))
        x = self.dropout(x)
        x = residual + self.drop_path(x)
        return x


class ConvNeXtDecoder(paddle.nn.Layer):
    def __init__(self, in_dims, out_dims, /, *, num_channels=512, num_layers=6, kernel_size=7, dropout_rate=0.1):
        super().__init__()
        self.inconv = paddle.nn.Conv1D(
            in_channels=in_dims,
            out_channels=num_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.conv = paddle.nn.LayerList(
            sublayers=(
                ConvNeXtBlock(
                    dim=num_channels,
                    intermediate_dim=num_channels * 4,
                    layer_scale_init_value=1e-06,
                    drop_out=dropout_rate,
                )
                for _ in range(num_layers)
            )
        )
        self.outconv = paddle.nn.Conv1D(
            in_channels=num_channels,
            out_channels=out_dims,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )

    def forward(self, x, infer=False):
        x = x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 1, 2))
        x = self.inconv(x)
        for conv in self.conv:
            x = conv(x)
        x = self.outconv(x)
        x = x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 1, 2))
        return x
