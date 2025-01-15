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
from collections import namedtuple
from functools import wraps

import paddle
from einops import rearrange
from utils import _FUNCTIONAL_PAD, _STR_2_PADDLE_DTYPE

EfficientAttentionConfig = namedtuple(
    "EfficientAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def compact(arr):
    return [*filter(exists, arr)]


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)


def create_causal_mask(i, j, device):
    return paddle.ones(shape=(i, j), dtype="bool").triu(diagonal=j - i + 1)


def onnx_create_causal_mask(i, j, device):
    r = paddle.arange(end=i)
    causal_mask = rearrange(r, "i -> i 1") < rearrange(r, "j -> 1 j")
    causal_mask = _FUNCTIONAL_PAD(pad=(j - i, 0), value=False, x=causal_mask)
    return causal_mask


class Attend(paddle.nn.Layer):
    def __init__(
        self,
        *,
        dropout=0.0,
        causal=False,
        heads=None,
        scale=None,
        flash=False,
        onnxable=False,
        sdp_kwargs: dict = dict(enable_flash=True, enable_math=True, enable_mem_efficient=True)
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.create_causal_mask = onnx_create_causal_mask if onnxable else create_causal_mask
        self.dropout = dropout
        self.attn_dropout = paddle.nn.Dropout(p=dropout)
        self.flash = flash and paddle.device.cuda.device_count() >= 1
        self.sdp_kwargs = sdp_kwargs

    def flash_attn(self, q, k, v, mask=None, attn_bias=None):
        heads, dtype = tuple(k.shape)[-2], q.dtype
        batch, q_len, _, k_len, device = (
            *tuple(q.shape),
            q.place,
        )
        q, k, v = map(lambda t: t, (q, k, v))
        if exists(self.scale):
            q = q * self.scale / tuple(q.shape)[-1] ** -0.5
        causal = self.causal
        if q_len == 1 and causal:
            causal = False
        if exists(mask):
            assert mask.ndim == 4
            mask = mask.expand(shape=[batch, heads, q_len, k_len])
        if k_len > q_len and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device=device)
            if not exists(mask):
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False
        row_is_entirely_masked = None
        if exists(mask) and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device=device)
            mask = mask & ~causal_mask
            row_is_entirely_masked = ~mask.astype("bool").any(axis=-1)
            mask[..., 0] = mask[..., 0] | row_is_entirely_masked
            causal = False
        if exists(attn_bias):
            attn_bias = rearrange(attn_bias, "h i j -> 1 h i j").expand(batch, heads, -1, -1)
            mask_value = -paddle.finfo(_STR_2_PADDLE_DTYPE(q.dtype)).max
            if exists(mask):
                attn_bias = attn_bias.masked_fill(mask=~mask, value=mask_value // 2)
            elif causal:
                causal_mask = self.create_causal_mask(q_len, k_len, device=device)
                attn_bias = attn_bias.masked_fill(mask=causal_mask, value=mask_value // 2)
                causal = False
            mask = attn_bias
        out = paddle.nn.functional.scaled_dot_product_attention(
            q.astype("float16"),
            k.astype("float16"),
            v.astype("float16"),
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        out = paddle.cast(dtype)
        if exists(row_is_entirely_masked):
            out = out.masked_fill(mask=row_is_entirely_masked[..., None], value=0.0)
        return out

    def forward(self, q, k, v, mask=None, attn_bias=None, prev_attn=None):

        n, device = tuple(q.shape)[-2], q.place
        scale = default(self.scale, tuple(q.shape)[-1] ** -0.5)
        causal = self.causal
        if n == 1 and causal:
            causal = False
        if self.flash:
            assert not exists(prev_attn), "residual attention not compatible with flash attention"
            return self.flash_attn(q, k, v, mask=mask, attn_bias=attn_bias)
        dots = paddle.einsum("b h i d, b h j d -> b h i j", q, k) * scale
        if exists(prev_attn):
            dots = dots + prev_attn
        if exists(attn_bias):
            dots = dots + attn_bias
        i, j = dots.shape[-2:]

        mask_value = -paddle.finfo(dots.dtype).max

        if exists(mask):
            dots = dots.masked_fill(mask=~mask, value=mask_value)
        if causal:
            causal_mask = self.create_causal_mask(i, j, device=device)
            dots = dots.masked_fill(mask=causal_mask, value=mask_value)
        attn = paddle.nn.functional.softmax(dots, axis=-1)
        attn = self.attn_dropout(attn)
        out = paddle.einsum("b h i j, b h j d -> b h i d", attn, v)
        return out
