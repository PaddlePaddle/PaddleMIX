# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute

try:
    from paddle.incubate.nn.memory_efficient_attention import (  # noqa
        memory_efficient_attention,
    )

    _ppxformers_available = True
except:
    _ppxformers_available = False

import math

import numpy as np
from einops import rearrange, repeat

from ..utils.initializer_utils import constant_, xavier_uniform_
from .lvdm_util import (
    GEGLU,
    Normalize,
    conv_nd,
    default,
    exists,
    normalization,
    zero_module,
)


def finfo(dtype):
    if dtype == paddle.float32:
        return np.finfo(np.float32)
    if dtype == paddle.float16:
        return np.finfo(np.float16)
    if dtype == paddle.float64:
        return np.finfo(np.float64)


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


class FeedForward(paddle.nn.Layer):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            paddle.nn.Sequential(paddle.nn.Linear(in_features=dim, out_features=inner_dim), paddle.nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )
        self.net = paddle.nn.Sequential(
            project_in, paddle.nn.Dropout(p=dropout), paddle.nn.Linear(in_features=inner_dim, out_features=dim_out)
        )

    def forward(self, x):
        return self.net(x)


class RelativePosition(paddle.nn.Layer):
    """https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py"""

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = paddle.nn.Parameter(paddle.empty(shape=[max_relative_position * 2 + 1, num_units]))
        xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = paddle.arange(end=length_q)
        range_vec_k = paddle.arange(end=length_k)
        distance_mat = range_vec_k[(None), :] - range_vec_q[:, (None)]
        distance_mat_clipped = paddle.clip(
            x=distance_mat, min=-self.max_relative_position, max=self.max_relative_position
        )
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.astype(dtype="int64")
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class TemporalCrossAttention(paddle.nn.Layer):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        use_relative_position=False,
        temporal_length=None,
        **kwargs
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.context_dim = context_dim
        self.scale = dim_head**-0.5
        self.heads = heads
        self.temporal_length = temporal_length
        self.use_relative_position = use_relative_position
        self.to_q = paddle.nn.Linear(in_features=query_dim, out_features=inner_dim, bias_attr=False)
        self.to_k = paddle.nn.Linear(in_features=context_dim, out_features=inner_dim, bias_attr=False)
        self.to_v = paddle.nn.Linear(in_features=context_dim, out_features=inner_dim, bias_attr=False)
        self.to_out = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=inner_dim, out_features=query_dim), paddle.nn.Dropout(p=dropout)
        )
        if use_relative_position:
            assert temporal_length is not None
            self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
        constant_(self.to_q.weight, 0)
        constant_(self.to_k.weight, 0)
        constant_(self.to_v.weight, 0)
        constant_(self.to_out[0].weight, 0)
        constant_(self.to_out[0].bias, 0)

    def forward(self, x, context=None, mask=None):
        nh = self.heads
        out = x
        q = self.to_q(out)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=nh), (q, k, v))
        sim = paddle.einsum("b i d, b j d -> b i j", q, k) * self.scale
        if self.use_relative_position:
            len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = paddle.einsum("b t d, t s d -> b t s", q, k2) * self.scale
            sim += sim2
        if mask is not None:
            max_neg_value = -1000000000.0
            sim = sim + (1 - mask.astype(dtype="float32")) * max_neg_value
        attn = paddle.nn.functional.softmax(sim, axis=-1)
        out = paddle.einsum("b i j, b j d -> b i d", attn, v)
        if self.use_relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = paddle.einsum("b t s, t s d -> b t d", attn, v2)
            out += out2
        out = rearrange(out, "(b h) n d -> b n (h d)", h=nh)
        return self.to_out(out)


class CrossAttention(paddle.nn.Layer):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads
        self.to_q = paddle.nn.Linear(in_features=query_dim, out_features=inner_dim, bias_attr=False)
        self.to_k = paddle.nn.Linear(in_features=context_dim, out_features=inner_dim, bias_attr=False)
        self.to_v = paddle.nn.Linear(in_features=context_dim, out_features=inner_dim, bias_attr=False)
        self.to_out = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=inner_dim, out_features=query_dim), paddle.nn.Dropout(p=dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads
        # b = x.shape[0]
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        sim = paddle.einsum("b i d, b j d -> b i j", q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim = masked_fill(sim, ~mask, max_neg_value)
        attn = paddle.nn.functional.softmax(sim, axis=-1)
        out = paddle.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class MemoryEfficientCrossAttention(paddle.nn.Layer):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, **kwargs):
        super().__init__()
        print(
            f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using {heads} heads."
        )
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = paddle.nn.Linear(in_features=query_dim, out_features=inner_dim, bias_attr=False)
        self.to_k = paddle.nn.Linear(in_features=context_dim, out_features=inner_dim, bias_attr=False)
        self.to_v = paddle.nn.Linear(in_features=context_dim, out_features=inner_dim, bias_attr=False)
        self.to_out = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=inner_dim, out_features=query_dim), paddle.nn.Dropout(p=dropout)
        )
        self.attention_op = "cutlass"

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        b, _, _ = q.shape
        q, k, v = map(lambda t: t.reshape([0, 0, self.heads, self.dim_head]), (q, k, v))
        out = F.scaled_dot_product_attention_(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            attention_op=self.attention_op,
            training=True,
        )
        if exists(mask):
            raise NotImplementedError
        out = out.reshape([0, 0, self.heads * self.dim_head])
        return self.to_out(out)


class BasicTransformerBlockST(paddle.nn.Layer):
    """
    if no context is given to forward function, cross-attention defaults to self-attention
    """

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        temporal_length=None,
        use_relative_position=True,
        **kwargs
    ):
        super().__init__()
        if _ppxformers_available:
            self.attn1 = MemoryEfficientCrossAttention(
                query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, **kwargs
            )
            self.attn2 = MemoryEfficientCrossAttention(
                query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, **kwargs
            )
        else:
            self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout, **kwargs)
            self.attn2 = CrossAttention(
                query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout, **kwargs
            )

        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = paddle.nn.LayerNorm(normalized_shape=dim, epsilon=1e-05, weight_attr=None, bias_attr=None)
        self.norm2 = paddle.nn.LayerNorm(normalized_shape=dim, epsilon=1e-05, weight_attr=None, bias_attr=None)
        self.norm3 = paddle.nn.LayerNorm(normalized_shape=dim, epsilon=1e-05, weight_attr=None, bias_attr=None)
        self.checkpoint = checkpoint
        self.attn1_tmp = TemporalCrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            temporal_length=temporal_length,
            use_relative_position=use_relative_position,
            **kwargs,
        )
        self.attn2_tmp = TemporalCrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=None,
            temporal_length=temporal_length,
            use_relative_position=use_relative_position,
            **kwargs,
        )
        self.norm4 = paddle.nn.LayerNorm(normalized_shape=dim, epsilon=1e-05, weight_attr=None, bias_attr=None)
        self.norm5 = paddle.nn.LayerNorm(normalized_shape=dim, epsilon=1e-05, weight_attr=None, bias_attr=None)

    def forward(self, x, context=None, **kwargs):
        if self.checkpoint:
            return recompute(self._forward, x, context)
        else:
            return self._forward(x, context)

    def _forward(self, x, context=None, mask=None):
        assert x.dim() == 5, f"x shape = {x.shape}"
        b, c, t, h, w = x.shape
        x = rearrange(x, "b c t h w -> (b t) (h w) c")
        x = self.attn1(self.norm1(x)) + x
        x = rearrange(x, "(b t) (h w) c -> b c t h w", b=b, h=h)
        x = rearrange(x, "b c t h w -> (b h w) t c")
        x = self.attn1_tmp(self.norm4(x), mask=mask) + x
        x = rearrange(x, "(b h w) t c -> b c t h w", b=b, h=h, w=w)
        x = rearrange(x, "b c t h w -> (b t) (h w) c")
        if context is not None:
            context_ = []
            for i in range(context.shape[0]):
                context_.append(context[i].unsqueeze(axis=0).tile(repeat_times=[t, 1, 1]))
            context_ = paddle.concat(x=context_, axis=0)
        else:
            context_ = None
        x = self.attn2(self.norm2(x), context=context_) + x
        x = rearrange(x, "(b t) (h w) c -> b c t h w", b=b, h=h)
        x = rearrange(x, "b c t h w -> (b h w) t c")
        x = self.attn2_tmp(self.norm5(x), context=None, mask=mask) + x
        x = self.ff(self.norm3(x)) + x
        x = rearrange(x, "(b h w) t c -> b c t h w", b=b, h=h, w=w)
        return x


class SpatialTemporalTransformer(paddle.nn.Layer):
    """
    Transformer block for video-like data (5D tensor).
    First, project the input (aka embedding) with NO reshape.
    Then apply standard transformer action.
    The 5D -> 3D reshape operation will be done in the specific attention module.
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        temporal_length=None,
        use_relative_position=True,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.proj_in = paddle.nn.Conv3D(
            in_channels=in_channels, out_channels=inner_dim, kernel_size=1, stride=1, padding=0
        )
        self.transformer_blocks = paddle.nn.LayerList(
            sublayers=[
                BasicTransformerBlockST(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    temporal_length=temporal_length,
                    use_relative_position=use_relative_position,
                    **kwargs,
                )
                for d in range(depth)
            ]
        )
        self.proj_out = zero_module(
            paddle.nn.Conv3D(in_channels=inner_dim, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, context=None, **kwargs):
        assert x.dim() == 5, f"x shape = {x.shape}"
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context=context, **kwargs)
        x = self.proj_out(x)
        return x + x_in


class STAttentionBlock(paddle.nn.Layer):
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        temporal_length=16,
        use_relative_position=False,
    ):
        super().__init__()
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.temporal_length = temporal_length
        self.use_relative_position = use_relative_position
        self.norm_s = normalization(channels)
        self.norm_t = normalization(channels)
        self.qkv_s = conv_nd(1, channels, channels * 3, 1)
        self.qkv_t = conv_nd(1, channels, channels * 3, 1)
        self.attention_s = QKVAttention(self.num_heads)
        self.attention_t = QKVAttention(self.num_heads)
        if use_relative_position:
            self.relative_position_k = RelativePosition(
                num_units=channels // self.num_heads, max_relative_position=temporal_length
            )
            self.relative_position_v = RelativePosition(
                num_units=channels // self.num_heads, max_relative_position=temporal_length
            )
        self.proj_out_s = zero_module(conv_nd(1, channels, channels, 1))
        self.proj_out_t = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, mask=None):
        b, c, t, h, w = x.shape
        out = rearrange(x, "b c t h w -> (b t) c (h w)")
        qkv = self.qkv_s(self.norm_s(out))
        out = self.attention_s(qkv)
        out = self.proj_out_s(out)
        out = rearrange(out, "(b t) c (h w) -> b c t h w", b=b, h=h)
        x += out
        out = rearrange(x, "b c t h w -> (b h w) c t")
        qkv = self.qkv_t(self.norm_t(out))
        if self.use_relative_position:
            len_q = qkv.shape[-1]
            len_k, len_v = len_q, len_q
            k_rp = self.relative_position_k(len_q, len_k)
            v_rp = self.relative_position_v(len_q, len_v)
            out = self.attention_t(qkv, rp=(k_rp, v_rp), mask=mask)
        else:
            out = self.attention_t(qkv, rp=None, mask=mask)
        out = self.proj_out_t(out)
        out = rearrange(out, "(b h w) c t -> b c t h w", b=b, h=h, w=w)
        return x + out


class QKVAttention(paddle.nn.Layer):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, rp=None, mask=None):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(chunks=3, axis=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = paddle.einsum(
            "bct,bcs->bts",
            (q * scale).reshape([bs * self.n_heads, ch, length]),
            (k * scale).reshape([bs * self.n_heads, ch, length]),
        )
        if rp is not None:
            k_rp, v_rp = rp
            weight2 = paddle.einsum("bct,tsc->bst", (q * scale).reshape([bs * self.n_heads, ch, length]), k_rp)
            weight += weight2
        if mask is not None:
            INF = -100000000.0
            weight = paddle.where(mask == 0, weight.astype(dtype="float32"), INF)
        weight = paddle.nn.functional.softmax(x=weight.astype(dtype="float32"), axis=-1).astype(weight.dtype)
        a = paddle.einsum("bts,bcs->bct", weight, v.reshape([bs * self.n_heads, ch, length]))
        if rp is not None:
            x = paddle.einsum("bts,tsc->btc", weight, v_rp)
            perm_3 = list(range(x.ndim))
            perm_3[1] = 2
            perm_3[2] = 1
            a2 = x.transpose(perm=perm_3)
            a += a2
        return a.reshape([bs, -1, length])
