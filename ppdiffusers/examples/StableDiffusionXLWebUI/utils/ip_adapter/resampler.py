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

# modified from https://github.com/tencent-ailab/IP-Adapter.git

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from einops import rearrange
from einops.layers.paddle import Rearrange


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias_attr=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias_attr=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.reshape([bs, length, heads, -1])
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose([0, 2, 1, 3])
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape([bs, heads, length, -1])
    return x


def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(axis=dim)

    denom = mask.sum(axis=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(axis=dim) / denom.clip(min=1e-5)


class PerceiverAttention(nn.Layer):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias_attr=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias_attr=False)
        self.to_out = nn.Linear(inner_dim, dim, bias_attr=False)

    def forward(self, x, latents):
        """
        Args:
            x (paddle.Tensor): image features
                shape (b, n1, D)
            latent (paddle.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape
        q = self.to_q(latents)
        kv_input = paddle.concat([x, latents], axis=-2)
        k, v = self.to_kv(kv_input).chunk(2, axis=-1)

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
        weight = F.softmax(weight.astype("float32"), axis=-1).astype(weight.dtype)
        out = weight @ v

        out = out.transpose([0, 2, 1, 3])
        out = out.reshape([b, l, -1])
        return self.to_out(out)


class Resampler(nn.Layer):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.latents = nn.Parameter(paddle.randn([1, num_queries, dim]) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.LayerList([])
        for _ in range(depth):
            self.layers.append(
                nn.LayerList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        if self.pos_emb is not None:
            device = paddle.get_device()
            n = x.shape[1]
            pos_emb = self.pos_emb(paddle.arange(n).to(device))
            x = x + pos_emb

        latents = self.latents.tile([x.shape[0], 1, 1])

        x = self.proj_in(x)

        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=paddle.ones(x.shape[:2]).to(device).astype("bool"))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = paddle.concat([meanpooled_latents, latents], axis=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)
