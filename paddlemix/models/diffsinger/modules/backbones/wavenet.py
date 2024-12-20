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

import math
import sys
from math import sqrt

import paddle
from paddlemix.models.diffsinger.utils import paddle_aux

from paddlemix.models.diffsinger.modules.commons.common_layers import SinusoidalPosEmb
from paddlemix.models.diffsinger.utils.hparams import hparams


class Conv1d(paddle.nn.Conv1D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init_KaimingNormal = paddle.nn.initializer.KaimingNormal(nonlinearity="leaky_relu")
        init_KaimingNormal(self.weight)


class ResidualBlock(paddle.nn.Layer):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = paddle.nn.Conv1D(
            in_channels=residual_channels,
            out_channels=2 * residual_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.diffusion_projection = paddle.nn.Linear(in_features=residual_channels, out_features=residual_channels)
        self.conditioner_projection = paddle.nn.Conv1D(
            in_channels=encoder_hidden, out_channels=2 * residual_channels, kernel_size=1
        )
        self.output_projection = paddle.nn.Conv1D(
            in_channels=residual_channels, out_channels=2 * residual_channels, kernel_size=1
        )

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(axis=-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner
        gate, filter = paddle_aux.split(x=y, num_or_sections=[self.residual_channels, self.residual_channels], axis=1)
        y = paddle.nn.functional.sigmoid(x=gate) * paddle.nn.functional.tanh(x=filter)
        y = self.output_projection(y)
        residual, skip = paddle_aux.split(
            x=y, num_or_sections=[self.residual_channels, self.residual_channels], axis=1
        )
        return (x + residual) / math.sqrt(2.0), skip


class WaveNet(paddle.nn.Layer):
    def __init__(self, in_dims, n_feats, *, num_layers=20, num_channels=256, dilation_cycle_length=4):
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.input_projection = Conv1d(in_dims * n_feats, num_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(num_channels)
        self.mlp = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=num_channels, out_features=num_channels * 4),
            paddle.nn.Mish(),
            paddle.nn.Linear(in_features=num_channels * 4, out_features=num_channels),
        )
        self.residual_layers = paddle.nn.LayerList(
            sublayers=[
                ResidualBlock(
                    encoder_hidden=hparams["hidden_size"],
                    residual_channels=num_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                )
                for i in range(num_layers)
            ]
        )
        self.skip_projection = Conv1d(num_channels, num_channels, 1)
        self.output_projection = Conv1d(num_channels, in_dims * n_feats, 1)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """
        :param spec: [B, F, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, H, T]
        :return:
        """
        if self.n_feats == 1:
            x = spec.squeeze(axis=1)
        else:
            x = spec.flatten(start_axis=1, stop_axis=2)
        x = self.input_projection(x)
        x = paddle.nn.functional.relu(x=x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond, diffusion_step)
            skip.append(skip_connection)
        x = paddle.sum(x=paddle.stack(x=skip), axis=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = paddle.nn.functional.relu(x=x)
        x = self.output_projection(x)
        if self.n_feats == 1:
            x = x[:, None, :, :]
        else:
            x = x.reshape(-1, self.n_feats, self.in_dims, tuple(x.shape)[2])
        return x
