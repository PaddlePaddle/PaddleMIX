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

import numpy as np
import paddle
import paddle.nn as nn


def load(path="../x.npy"):
    return paddle.to_tensor(np.load(path))


def diff(a, b):
    return (a - b).abs().mean()


class Linear(nn.Linear):
    def reset_parameters(self):
        return None


class Conv2d(nn.Conv2D):
    def reset_parameters(self):
        return None


class Attention2D(nn.Layer):
    def __init__(self, c, nhead, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiHeadAttention(c, nhead, dropout=dropout)

    def forward(self, x, kv, self_attn=False):
        orig_shape = x.shape
        x = x.reshape([x.shape[0], x.shape[1], -1]).transpose([0, 2, 1])
        if self_attn:
            kv = paddle.concat([x, kv], axis=1)
        x = self.attn(x, kv, kv)
        x = x.transpose([0, 2, 1]).reshape(orig_shape)
        return x


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.transpose(perm=[0, 2, 3, 1])).transpose(perm=[0, 3, 1, 2])


class GlobalResponseNorm(nn.Layer):
    def __init__(self, dim):
        super(GlobalResponseNorm, self).__init__()
        self.gamma = self.create_parameter(
            shape=[1, 1, 1, dim], default_initializer=paddle.nn.initializer.Constant(value=0.0)
        )
        self.beta = self.create_parameter(
            shape=[1, 1, 1, dim], default_initializer=paddle.nn.initializer.Constant(value=0.0)
        )
        self.gamma.stop_gradient = False
        self.beta.stop_gradient = False

    def forward(self, x):
        Gx = paddle.norm(x, p=2, axis=(1, 2), keepdim=True)
        Nx = Gx / (paddle.mean(Gx, axis=-1, keepdim=True) + 1e-6)
        x = self.gamma * (x * Nx) + self.beta + x
        return x


class ResBlock(nn.Layer):
    def __init__(self, c, c_skip=0, kernel_size=3, dropout=0.0):
        super().__init__()
        self.depthwise = Conv2d(c, c, kernel_size=kernel_size, padding=kernel_size // 2, groups=c)
        self.norm = LayerNorm2d(c, weight_attr=False, bias_attr=False, epsilon=1e-06)
        self.channelwise = nn.Sequential(
            Linear(c + c_skip, c * 4),
            nn.GELU(),
            GlobalResponseNorm(c * 4),
            nn.Dropout(p=dropout),
            Linear(c * 4, c),
        )

    def forward(self, x, x_skip=None):
        x_res = x
        x = self.depthwise(x)
        x = self.norm(x)
        if x_skip is not None:
            x = paddle.concat(x=[x, x_skip], axis=1)

        x = self.channelwise(x.transpose(perm=[0, 2, 3, 1])).transpose(perm=[0, 3, 1, 2])
        return x + x_res


class AttnBlock(nn.Layer):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0):
        super().__init__()
        self.self_attn = self_attn
        self.norm = LayerNorm2d(c, weight_attr=False, bias_attr=False, epsilon=1e-06)
        self.attention = Attention2D(c, nhead, dropout)
        self.kv_mapper = nn.Sequential(nn.Silu(), Linear(c_cond, c))

    def forward(self, x, kv):
        kv = self.kv_mapper(kv)
        x = x + self.attention(self.norm(x), kv, self_attn=self.self_attn)
        return x


class FeedForwardBlock(nn.Layer):
    def __init__(self, c, dropout=0.0):
        super().__init__()
        self.norm = LayerNorm2d(c, weight_attr=False, bias_attr=False, epsilon=1e-06)
        self.channelwise = nn.Sequential(
            Linear(c, c * 4),
            nn.GELU(),
            GlobalResponseNorm(c * 4),
            nn.Dropout(p=dropout),
            Linear(c * 4, c),
        )

    def forward(self, x):
        x = x + self.channelwise(self.norm(x).transpose(perm=[0, 2, 3, 1])).transpose(perm=[0, 3, 1, 2])
        return x


class TimestepBlock(nn.Layer):
    def __init__(self, c, c_timestep, conds=["sca"], trainable=True):
        super(TimestepBlock, self).__init__()
        self.mapper = nn.Linear(c_timestep, c * 2, bias_attr=trainable)
        self.conds = conds
        for cname in conds:
            setattr(self, f"mapper_{cname}", nn.Linear(c_timestep, c * 2, bias_attr=trainable))

    def forward(self, x, t):
        t = paddle.split(t, num_or_sections=len(self.conds) + 1, axis=1)
        a_b = self.mapper(t[0])
        a, b = a_b[:, : a_b.shape[1] // 2, None, None], a_b[:, a_b.shape[1] // 2 :, None, None]
        for i, c in enumerate(self.conds):
            ac_bc = getattr(self, f"mapper_{c}")(t[i + 1])
            ac, bc = ac_bc[:, : ac_bc.shape[1] // 2, None, None], ac_bc[:, ac_bc.shape[1] // 2 :, None, None]
            a, b = a + ac, b + bc
        return x * (1 + a) + b
