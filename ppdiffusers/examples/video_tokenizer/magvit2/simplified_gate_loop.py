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
from associative_scan import associative_scan
from einops import pack, unpack
from einops.layers.paddle import Rearrange


def exists(v):
    return v is not None


def abs_clamp_eps(t, eps=1e-20):
    sign = paddle.sign(x=t)
    return sign * t.abs().clip(min=eps)


def heinsen_associative_scan(a, kv, eps=1e-20):
    log_a = a.clip(min=eps).log()
    log_kv = abs_clamp_eps(kv, eps=eps).to(dtype="complex64").log()
    a_star = paddle.cumsum(x=log_a, axis=1)
    log_x0_plus_b_star = paddle.logcumsumexp(x=log_kv - a_star, axis=1)
    log_x = a_star + log_x0_plus_b_star
    return a_star.exp().real(), log_x.exp().real()


def binary_operator(a: Tuple[paddle.Tensor, paddle.Tensor], b: Tuple[paddle.Tensor, paddle.Tensor]):
    a_i, kv_i = a
    a_j, kv_j = b
    return a_j * a_i, paddle.add(kv_j, 1 * a_j * kv_i)


def gate_loop_operator(q, kv, a, cache=None, heinsen=False):
    if exists(cache):
        cache_a, cache_kv = cache
        a, a_ps = pack([cache_a, a], "b * d")
        kv, kv_ps = pack([cache_kv, kv], "b * d")
    if heinsen:
        a, kv = heinsen_associative_scan(a, kv)
    else:
        a, kv = associative_scan(binary_operator, (a, kv))
    if exists(cache):
        _, a = unpack(a, a_ps, "b * d")
        _, kv = unpack(kv, kv_ps, "b * d")
    return q * kv, (a[:, (-1)], kv[:, (-1)])


class RMSNorm(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        out_5 = paddle.create_parameter(
            shape=paddle.ones(shape=dim).shape,
            dtype=paddle.ones(shape=dim).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=dim)),
        )
        out_5.stop_gradient = not True
        self.gamma = out_5

    def forward(self, x):
        return paddle.nn.functional.normalize(x=x, axis=-1) * self.scale * self.gamma


class SimpleGateLoopLayer(paddle.nn.Layer):
    """
    simplified gate loop
    seeing if it can supplement attention as shown in https://github.com/lucidrains/mega-pytorch
    """

    def __init__(
        self, dim, prenorm=True, use_heinsen=False, use_jax_associative_scan=False, post_ln=False, reverse=False
    ):
        super().__init__()
        assert int(use_heinsen) + int(use_jax_associative_scan) <= 1
        self.norm = RMSNorm(dim) if prenorm else paddle.nn.Identity()
        self.dim = dim
        self.to_qkva = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=dim, out_features=dim * 3, bias_attr=False),
            Rearrange("b n (qkva d) -> qkva (b d) n 1", qkva=3),
        )
        self.gate_loop_fn = gate_loop_operator
        self.maybe_post_ln = paddle.nn.LayerNorm(normalized_shape=dim) if post_ln else paddle.nn.Identity()
        self.split_heads = Rearrange("(b d) n 1 -> b n d", d=dim)
        self.reverse = reverse

    def forward(self, x, cache=None, return_cache=False):
        if self.reverse:
            x = paddle.flip(x=x, axis=(-2,))
        x = self.norm(x)
        q, kv, a = self.to_qkva(x)
        out, cache = self.gate_loop_fn(q, kv, a.sigmoid(), cache=cache)
        out = self.split_heads(out)
        out = self.maybe_post_ln(out)
        if self.reverse:
            out = paddle.flip(x=out, axis=(-2,))
        if not return_cache:
            return out
        assert not self.reverse, "caching only works with non-reversed seq"
        return out, cache
