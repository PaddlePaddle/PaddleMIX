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

import paddle
from paddle import nn


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(normalized_shape=dim),
        nn.Linear(in_features=dim, out_features=inner_dim, bias_attr=False),
        paddle.nn.GELU(),
        paddle.nn.Linear(in_features=inner_dim, out_features=dim, bias_attr=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.reshape([bs, length, heads, -1])
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose([0, 2, 1])
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape([bs, heads, length, -1])
    return x


class PerceiverAttention(nn.Layer):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm1 = nn.LayerNorm(normalized_shape=dim)
        self.norm2 = nn.LayerNorm(normalized_shape=dim)

        self.to_q = nn.Linear(in_features=dim, out_features=inner_dim, bias_attr=False)
        self.to_kv = nn.Linear(in_features=dim, out_features=inner_dim * 2, bias_attr=False)
        self.to_out = nn.Linear(in_features=inner_dim, out_features=dim, bias_attr=False)

    def forward(self, x, latents):
        """
        Args:
            x (paddle.Tensor): image features
                shape (b, n1, D)
            latents (paddle.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = paddle.concat(x=(x, latents), axis=-2)
        k, v = self.to_kv(kv_input).chunk(chunks=2, axis=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        x = k * scale
        perm = list(range(x.ndim))
        perm[-2] = -1
        perm[-1] = -2
        weight = q * scale @ x.transpose(perm=perm)
        weight = nn.functional.softmax(x=weight.astype(dtype="float32"), axis=-1).astype(weight.dtype)
        out = weight @ v
        out = out.transpose(perm=[0, 2, 1, 3]).reshape([b, l, -1])
        return self.to_out(out)


class Resampler(nn.Layer):
    def __init__(
        self, dim=1024, depth=8, dim_head=64, heads=16, num_queries=8, embedding_dim=768, output_dim=1024, ff_mult=4
    ):
        super().__init__()
        self.latents = nn.Parameter(paddle.randn([1, num_queries, dim]) / dim**0.5)
        self.proj_in = nn.Linear(in_features=embedding_dim, out_features=dim)
        self.proj_out = nn.Linear(in_features=dim, out_features=output_dim)
        self.norm_out = nn.LayerNorm(normalized_shape=output_dim)

        self.layers = nn.LayerList(sublayers=[])
        for _ in range(depth):
            self.layers.append(
                nn.LayerList(
                    sublayers=[
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        latents = self.latents.tile(repeat_times=([x.shape[0], 1, 1]))
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)
