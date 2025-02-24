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
from functools import partial
from typing import Optional

import paddle
from einops import pack, rearrange, repeat, unpack
from einops.layers.paddle import Rearrange
from rotary_embedding import RotaryEmbedding
from utils import _FUNCTIONAL_PAD

Cache = namedtuple("Cache", ["seq_len", "last_token", "kv_cumsum", "k_cumsum"])


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def shift(t):
    t, t_shift = t.chunk(chunks=2, axis=-1)
    t_shift = _FUNCTIONAL_PAD(pad=(0, 0, 1, -1), value=0.0, x=t_shift)
    return paddle.concat(x=(t, t_shift), axis=-1)


class RMSNorm(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        out_0 = paddle.create_parameter(
            shape=paddle.ones(shape=dim).shape,
            dtype=paddle.ones(shape=dim).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=dim)),
        )
        out_0.stop_gradient = not True
        self.gamma = out_0

    def forward(self, x):
        return self.gamma * paddle.nn.functional.normalize(x=x, axis=-1) * self.scale


def second_taylor_expansion(x: paddle.Tensor):
    x, ps = pack([x], "* d")
    lead_dims = tuple(x.shape)[0]
    x0 = paddle.ones(shape=(lead_dims,), dtype=x.dtype)
    x1 = x
    x2 = paddle.einsum("... i, ... j -> ... i j", x, x) * 0.5**0.5
    out, _ = pack([x0, x1, x2], "b *")
    (out,) = unpack(out, ps, "* d")
    return out


class TaylorSeriesLinearAttn(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        *,
        dim_head=16,
        heads=8,
        causal=False,
        one_headed_kv=False,
        rotary_emb=False,
        combine_heads=True,
        gate_value_heads=False,
        prenorm=False,
        shift_tokens=False,
        dropout=0.0
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        dim_inner = dim_head * heads

        self.shift_tokens = shift_tokens
        self.norm = RMSNorm(dim) if prenorm else paddle.nn.Identity()

        self.heads = heads
        self.dim_hidden = dim_inner

        self.causal = causal
        self.causal_linear_attn_fn = None

        kv_heads = heads if not one_headed_kv else 1
        dim_kv_inner = dim_head * (heads if not one_headed_kv else 1)

        self.rotary_emb = RotaryEmbedding(dim_head) if rotary_emb else None

        self.one_headed_kv = one_headed_kv

        self.to_q = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=dim, out_features=dim_inner, bias_attr=False),
            Rearrange("b n (h d) -> b h n d", h=heads),
        )
        self.to_kv = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=dim, out_features=dim_kv_inner * 2, bias_attr=False),
            Rearrange("b n (kv h d) -> kv b h n d", kv=2, h=kv_heads),
        )

        self.to_v_gates = (
            paddle.nn.Sequential(
                paddle.nn.Linear(in_features=dim, out_features=heads, bias_attr=False),
                paddle.nn.Sigmoid(),
                Rearrange("b n h -> b h n 1"),
            )
            if gate_value_heads
            else None
        )

        self.merge_heads = Rearrange("b h n d -> b n (h d)")
        self.to_out = paddle.nn.Identity()

        if combine_heads:
            self.to_out = paddle.nn.Sequential(
                paddle.nn.Linear(in_features=dim_inner, out_features=dim, bias_attr=False),
                paddle.nn.Dropout(p=dropout),
            )

    def forward(
        self,
        x: paddle.Tensor,
        mask: paddle.Tensor = None,
        context: paddle.Tensor = None,
        eps: float = 1e-05,
        cache: Optional[Cache] = None,
        return_cache=False,
    ):
        """
        einops
        b - batch
        h - heads
        d - query / key head dimension
        e - value head dimension
        n - source query sequence length
        m - target key / value sequence length
        """
        orig_input, seq_len, is_cross_attn = x, tuple(x.shape)[-2], exists(context)
        assert not (exists(self.rotary_emb) and is_cross_attn), "rotary embedding does not work with cross attention"
        if self.shift_tokens:
            if exists(cache):
                x, ps = pack([cache.last_token, x], "b * d")
            x = shift(x)
            if exists(cache):
                _, x = unpack(x, ps, "b * d")
        normed = self.norm(x)
        q = self.to_q(normed)
        k, v = self.to_kv(default(context, normed))
        if exists(self.rotary_emb):
            rotate_fn = self.rotary_emb.rotate_queries_or_keys
            if exists(cache):
                rotate_fn = partial(rotate_fn, offset=cache.seq_len)
            q, k = map(rotate_fn, (q, k))
        q = q * self.scale
        q, k = map(second_taylor_expansion, (q, k))
        if self.causal:
            assert not exists(mask), "masking does not make sense for autoregressive linear attention"
            assert not is_cross_attn, "causal does not make sense with cross attention"
            if self.one_headed_kv:
                k, v = map(lambda t: repeat(t, "b 1 n d -> b h n d", h=self.heads), (k, v))
            if exists(cache):
                assert seq_len == 1
                old_seq_len, _, kv_cumsum_cache, k_cumsum_cache = cache
                kv = paddle.einsum("b h n d, b h n e -> b h d e", k, v)
                kv_cumsum = kv + kv_cumsum_cache
                k_cumsum = k + k_cumsum_cache
                num = paddle.einsum("b h n d, b h d e -> b h n e", q, kv_cumsum)
                den = paddle.einsum("... n d, ... n d -> ... n", q, k_cumsum)
                den = rearrange(den, "... -> ... 1")
                out = num / den.clip(min=eps)
                if return_cache:
                    new_cache = Cache(old_seq_len + 1, orig_input, kv_cumsum, k_cumsum)
            else:
                num = self.causal_linear_attn_fn(q, k, v)
                k_cumsum = k.cumsum(axis=-2)
                den = paddle.einsum("... n d, ... n d -> ... n", q, k_cumsum)
                den = rearrange(den, "... -> ... 1")
                out = num / den.clip(min=eps)
                if return_cache:
                    new_kv_cache = paddle.einsum("b h n d, b h n e -> b h d e", k, v)
                    new_k_cumsum_cache = k_cumsum[(...), -1:, :]
                    new_cache = Cache(seq_len, orig_input[:, -1:], new_kv_cache, new_k_cumsum_cache)
        else:
            assert not return_cache, "cache is only needed for autoregressive"
            if exists(mask):
                mask = rearrange(mask, "b n -> b 1 n 1")
                k = k.masked_fill(mask=~mask, value=0.0)
                v = v.masked_fill(mask=~mask, value=0.0)
            if self.one_headed_kv:
                k, v = map(lambda t: rearrange(t, "b 1 n d -> b n d"), (k, v))
                kv = paddle.einsum("b n d, b n e -> b d e", k, v)
                qk_inv = 1.0 / paddle.einsum("b h n d, b m d -> b h n", q, k).clip(min=eps)
                out = paddle.einsum("b h n d, b d e, b h n -> b h n e", q, kv, qk_inv)
            else:
                kv = paddle.einsum("b h n d, b h n e -> b h d e", k, v)
                qk_inv = 1.0 / paddle.einsum("b h n d, b h m d -> b h n", q, k).clip(min=eps)
                out = paddle.einsum("b h n d, b h d e, b h n -> b h n e", q, kv, qk_inv)
        if exists(self.to_v_gates):
            out = out * self.to_v_gates(x)
        out = self.merge_heads(out)
        out = self.to_out(out)
        if not return_cache:
            return out
        return out, new_cache
