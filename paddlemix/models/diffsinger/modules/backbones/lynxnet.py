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

import paddle
from paddlemix.models.diffsinger.utils import paddle_aux

from paddlemix.models.diffsinger.modules.commons.common_layers import SinusoidalPosEmb
from paddlemix.models.diffsinger.utils.hparams import hparams


class SwiGLU(paddle.nn.Layer):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = paddle_aux.split(x=x, num_or_sections=x.shape[self.dim] // 2, axis=self.dim)
        return out * paddle.nn.functional.silu(x=gate)


class Transpose(paddle.nn.Layer):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, "dims must be a tuple of two dimensions"
        self.dims = dims

    def forward(self, x):
        # return x.transpose(*self.dims)
        # return x.transpose(perm=list(self.dims))  # or tuple(self.dims)
        return x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, *self.dims))


class LYNXConvModule(paddle.nn.Layer):
    @staticmethod
    def calc_same_padding(kernel_size):
        pad = kernel_size // 2
        return pad, pad - (kernel_size + 1) % 2

    def __init__(self, dim, expansion_factor, kernel_size=31, activation="PReLU", dropout=0.0):
        super().__init__()
        inner_dim = dim * expansion_factor
        activation_classes = {
            "SiLU": paddle.nn.Silu,
            "ReLU": paddle.nn.ReLU,
            "PReLU": lambda: paddle.nn.PReLU(num_parameters=inner_dim),
        }
        activation = activation if activation is not None else "PReLU"
        if activation not in activation_classes:
            raise ValueError(f"{activation} is not a valid activation")
        _activation = activation_classes[activation]()
        padding = self.calc_same_padding(kernel_size)
        if float(dropout) > 0.0:
            _dropout = paddle.nn.Dropout(p=dropout)
        else:
            _dropout = paddle.nn.Identity()
        self.net = paddle.nn.Sequential(
            paddle.nn.LayerNorm(normalized_shape=dim),
            Transpose((1, 2)),
            paddle.nn.Conv1D(in_channels=dim, out_channels=inner_dim * 2, kernel_size=1),
            SwiGLU(dim=1),
            paddle.nn.Conv1D(
                in_channels=inner_dim,
                out_channels=inner_dim,
                kernel_size=kernel_size,
                padding=padding[0],
                groups=inner_dim,
            ),
            _activation,
            paddle.nn.Conv1D(in_channels=inner_dim, out_channels=dim, kernel_size=1),
            Transpose((1, 2)),
            _dropout,
        )

    def forward(self, x):
        return self.net(x)


class LYNXNetResidualLayer(paddle.nn.Layer):
    def __init__(self, dim_cond, dim, expansion_factor, kernel_size=31, activation="PReLU", dropout=0.0):
        super().__init__()
        self.diffusion_projection = paddle.nn.Conv1D(in_channels=dim, out_channels=dim, kernel_size=1)
        self.conditioner_projection = paddle.nn.Conv1D(in_channels=dim_cond, out_channels=dim, kernel_size=1)
        self.convmodule = LYNXConvModule(
            dim=dim, expansion_factor=expansion_factor, kernel_size=kernel_size, activation=activation, dropout=dropout
        )

    def forward(self, x, conditioner, diffusion_step):
        res_x = x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 1, 2))
        x = x + self.diffusion_projection(diffusion_step) + self.conditioner_projection(conditioner)
        x = x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 1, 2))
        x = self.convmodule(x)
        x = x + res_x
        x = x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 1, 2))
        return x


class LYNXNet(paddle.nn.Layer):
    def __init__(
        self,
        in_dims,
        n_feats,
        *,
        num_layers=6,
        num_channels=512,
        expansion_factor=2,
        kernel_size=31,
        activation="PReLU",
        dropout=0.0
    ):
        """
        LYNXNet(Linear Gated Depthwise Separable Convolution Network)
        TIPS:You can control the style of the generated results by modifying the 'activation',
            - 'PReLU'(default) : Similar to WaveNet
            - 'SiLU' : Voice will be more pronounced, not recommended for use under DDPM
            - 'ReLU' : Contrary to 'SiLU', Voice will be weakened
        """
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.input_projection = paddle.nn.Conv1D(
            in_channels=in_dims * n_feats, out_channels=num_channels, kernel_size=1
        )
        self.diffusion_embedding = paddle.nn.Sequential(
            SinusoidalPosEmb(num_channels),
            paddle.nn.Linear(in_features=num_channels, out_features=num_channels * 4),
            paddle.nn.GELU(),
            paddle.nn.Linear(in_features=num_channels * 4, out_features=num_channels),
        )
        self.residual_layers = paddle.nn.LayerList(
            sublayers=[
                LYNXNetResidualLayer(
                    dim_cond=hparams["hidden_size"],
                    dim=num_channels,
                    expansion_factor=expansion_factor,
                    kernel_size=kernel_size,
                    activation=activation,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = paddle.nn.LayerNorm(normalized_shape=num_channels)
        self.output_projection = paddle.nn.Conv1D(
            in_channels=num_channels, out_channels=in_dims * n_feats, kernel_size=1
        )
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
            x = spec[:, 0]
        else:
            x = spec.flatten(start_axis=1, stop_axis=2)
        x = self.input_projection(x)
        x = paddle.nn.functional.gelu(x=x)
        diffusion_step = self.diffusion_embedding(diffusion_step).unsqueeze(axis=-1)
        for layer in self.residual_layers:
            x = layer(x, cond, diffusion_step)
        x = self.norm(x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 1, 2))).transpose(
            perm=paddle_aux.transpose_aux_func(
                self.norm(x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 1, 2))).ndim, 1, 2
            )
        )
        x = self.output_projection(x)
        if self.n_feats == 1:
            x = x[:, None, :, :]
        else:
            x = x.reshape(-1, self.n_feats, self.in_dims, tuple(x.shape)[2])
        return x
