# Copyright (c) 2023 Dominic Rampas MIT License
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import paddle.nn as nn

from ...models.attention_processor import Attention
from ...models.lora import LoRACompatibleConv, LoRACompatibleLinear
from ...utils import USE_PEFT_BACKEND


class WuerstchenLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.transpose([0, 2, 3, 1])
        x = super().forward(x)
        return x.transpose([0, 3, 1, 2])


class TimestepBlock(nn.Layer):
    def __init__(self, c, c_timestep):
        super().__init__()
        linear_cls = nn.Linear if USE_PEFT_BACKEND else LoRACompatibleLinear
        self.mapper = linear_cls(c_timestep, c * 2)

    def forward(self, x, t):
        a, b = self.mapper(t)[:, :, None, None].chunk(2, axis=1)
        return x * (1 + a) + b


class ResBlock(nn.Layer):
    def __init__(self, c, c_skip=0, kernel_size=3, dropout=0.0):
        super().__init__()

        conv_cls = nn.Conv2D if USE_PEFT_BACKEND else LoRACompatibleConv
        linear_cls = nn.Linear if USE_PEFT_BACKEND else LoRACompatibleLinear

        self.depthwise = conv_cls(c + c_skip, c, kernel_size=kernel_size, padding=kernel_size // 2, groups=c)
        norm_elementwise_affine = False
        norm_elementwise_affine_kwargs = {} if norm_elementwise_affine else dict(weight_attr=False, bias_attr=False)
        self.norm = WuerstchenLayerNorm(c, epsilon=1e-6, **norm_elementwise_affine_kwargs)
        self.channelwise = nn.Sequential(
            linear_cls(c, c * 4), nn.GELU(), GlobalResponseNorm(c * 4), nn.Dropout(dropout), linear_cls(c * 4, c)
        )

    def forward(self, x, x_skip=None):
        x_res = x
        if x_skip is not None:
            x = paddle.concat([x, x_skip], axis=1)
        x = self.norm(self.depthwise(x)).transpose([0, 2, 3, 1])
        x = self.channelwise(x).transpose([0, 3, 1, 2])
        return x + x_res


# from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105
class GlobalResponseNorm(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(paddle.zeros([1, 1, 1, dim]))
        self.beta = nn.Parameter(paddle.zeros([1, 1, 1, dim]))
        # NOTE: Reference https://github.com/PaddlePaddle/Paddle/pull/60070 for inconsistent upgrade behavior.
        # Paddle<=2.6.0, when len(axis) == 2 and p is ±1 or ±2, the function computed the vector norm.
        # However, after the upgrade, it now computes the matrix norm instead.
        # In contrast, PyTorch computes the vector norm in such cases.
        self.norm = paddle.linalg.vector_norm if hasattr(paddle.linalg, "vector_norm") else paddle.linalg.norm

    def forward(self, x):
        # lxl: fix normlaization error
        dtype = x.dtype
        x = x.cast("float32")
        agg_norm = self.norm(x, p=2, axis=(1, 2), keepdim=True)
        agg_norm = agg_norm.cast(dtype)
        x = x.cast(dtype)

        stand_div_norm = agg_norm / (agg_norm.mean(axis=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * stand_div_norm) + self.beta + x


class AttnBlock(nn.Layer):
    def __init__(self, c, c_cond, nhead, self_attn=True, dropout=0.0):
        super().__init__()

        linear_cls = nn.Linear if USE_PEFT_BACKEND else LoRACompatibleLinear

        self.self_attn = self_attn
        norm_elementwise_affine = False
        norm_elementwise_affine_kwargs = {} if norm_elementwise_affine else dict(weight_attr=False, bias_attr=False)
        self.norm = WuerstchenLayerNorm(c, epsilon=1e-6, **norm_elementwise_affine_kwargs)
        self.attention = Attention(query_dim=c, heads=nhead, dim_head=c // nhead, dropout=dropout, bias=True)
        self.kv_mapper = nn.Sequential(nn.Silu(), linear_cls(c_cond, c))

    def forward(self, x, kv):
        kv = self.kv_mapper(kv)
        norm_x = self.norm(x)
        if self.self_attn:
            batch_size, channel, _, _ = x.shape
            kv = paddle.concat([norm_x.reshape([batch_size, channel, -1]).transpose([0, 2, 1]), kv], axis=1)
        x = x + self.attention(norm_x, encoder_hidden_states=kv)
        return x
