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

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from einops import rearrange

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .modeling_utils import ModelMixin

USE_TEMPORAL_TRANSFORMER = True


@dataclass
class STUNetOutput(BaseOutput):
    """
    The output of [`SFUNetModel`].

    Args:
        sample (`paddle.Tensor` of shape `(batch_size, num_channels, sample_size)`):
            The hidden states output from the last layer of the model.
    """

    sample: paddle.Tensor


def sinusoidal_embedding_paddle(timesteps: paddle.Tensor, embedding_dim: int):
    half = embedding_dim // 2
    timesteps = timesteps.astype("float32")

    exponent = -paddle.arange(half).astype(timesteps.dtype)
    exponent = paddle.divide(exponent, paddle.to_tensor(half, dtype=timesteps.dtype))
    sinusoid = paddle.outer(timesteps, paddle.pow(paddle.to_tensor(10000, dtype=timesteps.dtype), exponent))
    x = paddle.concat([paddle.cos(sinusoid), paddle.sin(sinusoid)], axis=1)
    if embedding_dim % 2 != 0:
        x = paddle.concat([x, paddle.zeros_like(x[:, :1])], axis=1)
    return x


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return paddle.ones(shape, dtype=paddle.bool)
    elif prob == 0:
        return paddle.zeros(shape, dtype=paddle.bool)
    else:
        mask = paddle.zeros(shape).astype("float32").uniform_(0, 1) < prob
        # aviod mask all, which will cause find_unused_parameters error
        if mask.all():
            mask[0] = False
        return mask


class MemoryEfficientCrossAttention(nn.Layer):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.num_heads = heads
        self.head_size = dim_head
        self.scale = 1 / math.sqrt(self.head_size)
        self.to_q = nn.Linear(query_dim, inner_dim, bias_attr=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias_attr=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias_attr=False)

        self.proj_attn = nn.Linear(inner_dim, query_dim)

        self.dropout = nn.Dropout(dropout)

        self._use_memory_efficient_attention_xformers = True

    def reshape_heads_to_batch_dim(self, tensor, transpose=True):
        tensor = tensor.reshape([0, 0, self.num_heads, self.head_size])
        if transpose:
            tensor = tensor.transpose([0, 2, 1, 3])
        return tensor

    def reshape_batch_dim_to_heads(self, tensor, transpose=True):
        if transpose:
            tensor = tensor.transpose([0, 2, 1, 3])
        tensor = tensor.reshape([0, 0, tensor.shape[2] * tensor.shape[3]])
        return tensor

    def forward(self, x, context=None, mask=None):
        query_proj = self.to_q(x)
        context = default(context, x)
        key_proj = self.to_k(context)
        value_proj = self.to_v(context)

        query_proj = self.reshape_heads_to_batch_dim(
            query_proj, transpose=not self._use_memory_efficient_attention_xformers
        )
        key_proj = self.reshape_heads_to_batch_dim(
            key_proj, transpose=not self._use_memory_efficient_attention_xformers
        )
        value_proj = self.reshape_heads_to_batch_dim(
            value_proj, transpose=not self._use_memory_efficient_attention_xformers
        )

        if self._use_memory_efficient_attention_xformers:
            hidden_states = F.scaled_dot_product_attention_(
                query_proj,
                key_proj,
                value_proj,
                attn_mask=None,
                scale=self.scale,
                dropout_p=0.0,
                training=self.training,
                attention_op=None,
            )
        else:
            attention_scores = paddle.matmul(query_proj, key_proj, transpose_y=True) * self.scale
            attention_probs = F.softmax(attention_scores.cast("float32"), axis=-1).cast(attention_scores.dtype)
            hidden_states = paddle.matmul(attention_probs, value_proj)
        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(
            hidden_states, transpose=not self._use_memory_efficient_attention_xformers
        )

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        # hidden_states = hidden_states.transpose([0, 2, 1])
        out = self.dropout(hidden_states)
        return out


# feedforward
class GEGLU(nn.Layer):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = paddle.chunk(self.proj(x), chunks=2, axis=-1)
        return x * F.gelu(gate)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class FeedForward(nn.Layer):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU()) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.net(x)


class BasicTransformerBlock(nn.Layer):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
    ):
        super().__init__()
        attn_cls = MemoryEfficientCrossAttention
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class TemporalTransformer(nn.Layer):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
        only_self_att=True,
        multiply_zero=False,
    ):
        super().__init__()
        self.multiply_zero = multiply_zero
        self.only_self_att = only_self_att
        if self.only_self_att:
            context_dim = None
        if not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, epsilon=1e-06)
        if not use_linear:
            self.proj_in = nn.Conv1D(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.LayerList(
            [
                BasicTransformerBlock(
                    inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d], checkpoint=use_checkpoint
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv1D(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))

        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if self.only_self_att:
            context = None
        if not isinstance(context, list):
            context = [context]
        b, c, f, h, w = x.shape
        x_in = x
        x = self.norm(x)

        if not self.use_linear:
            x = rearrange(x, "b c f h w -> (b h w) c f")
            x = self.proj_in(x)
        # [16384, 16, 320]
        if self.use_linear:
            x = rearrange(x, "(b f) c h w -> b (h w) f c", f=self.frames)
            x = self.proj_in(x)

        if self.only_self_att:
            x = rearrange(x, "bhw c f -> bhw f c")
            for i, block in enumerate(self.transformer_blocks):
                x = block(x)
            x = rearrange(x, "(b hw) f c -> b hw f c", b=b)
        else:
            x = rearrange(x, "(b hw) c f -> b hw f c", b=b)
            for i, block in enumerate(self.transformer_blocks):
                context[i] = rearrange(context[i], "(b f) l con -> b f l con", f=self.frames)
                # calculate each batch one by one (since number in shape could not greater then 65,535 for some package)
                for j in range(b):
                    context_i_j = repeat(
                        context[i][j], "f l con -> (f r) l con", r=(h * w) // self.frames, f=self.frames
                    )
                    x[j] = block(x[j], context=context_i_j)

        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, "b (h w) f c -> b f c h w", h=h, w=w)
        if not self.use_linear:
            x = rearrange(x, "b hw f c -> (b hw) c f")
            x = self.proj_out(x)
            x = rearrange(x, "(b h w) c f -> b c f h w", b=b, h=h, w=w)

        if self.multiply_zero:
            x = 0.0 * x + x_in
        else:
            x = x + x_in
        return x


class Upsample(nn.Layer):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = nn.Conv2D(self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Layer):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = nn.Conv2D(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class TemporalConvBlock_v2(nn.Layer):
    def __init__(self, in_dim, out_dim=None, dropout=0.0, use_image_dataset=False):
        super(TemporalConvBlock_v2, self).__init__()
        if out_dim is None:
            out_dim = in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_image_dataset = use_image_dataset

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(32, in_dim), nn.Silu(), nn.Conv3D(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0))
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.Silu(),
            nn.Dropout(dropout),
            nn.Conv3D(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.Silu(),
            nn.Dropout(dropout),
            nn.Conv3D(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv4 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.Silu(),
            nn.Dropout(dropout),
            nn.Conv3D(
                out_dim,
                in_dim,
                (3, 1, 1),
                padding=(1, 0, 0),
                weight_attr=nn.initializer.Constant(value=0.0),
                bias_attr=nn.initializer.Constant(value=0.0),
            ),
        )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        if self.use_image_dataset:
            x = identity + 0.0 * x
        else:
            x = identity + x
        return x


class ResBlock(nn.Layer):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        use_temporal_conv=True,
        use_image_dataset=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_temporal_conv = use_temporal_conv

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.Silu(),
            nn.Conv2D(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.Silu(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.Silu(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2D(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2D(channels, self.out_channels, 1)

        if self.use_temporal_conv:
            self.temopral_conv = TemporalConvBlock_v2(
                self.out_channels, self.out_channels, dropout=0.1, use_image_dataset=use_image_dataset
            )

    def forward(self, x, emb, batch_size):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return self._forward(x, emb, batch_size)

    def _forward(self, x, emb, batch_size):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).astype(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        h = self.skip_connection(x) + h

        if self.use_temporal_conv:
            h = rearrange(h, "(b f) c h w -> b c f h w", b=batch_size)
            h = self.temopral_conv(h)
            h = rearrange(h, "b c f h w -> (b f) c h w")
        return h


class TemporalAttentionBlock(nn.Layer):
    def __init__(self, dim, heads=4, dim_head=32, rotary_emb=None, use_image_dataset=False, use_sim_mask=False):
        super().__init__()
        # consider num_heads first, as pos_bias needs fixed num_heads
        dim_head = dim // heads
        assert heads * dim_head == dim
        self.use_image_dataset = use_image_dataset
        self.use_sim_mask = use_sim_mask

        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = nn.GroupNorm(32, dim)
        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3)
        self.to_out = nn.Linear(hidden_dim, dim)

    def masked_fill(self, x, mask, value):
        y = paddle.full(x.shape, value, x.dtype)
        return paddle.where(mask, y, x)

    def forward(self, x, pos_bias=None, focus_present_mask=None, video_mask=None):

        identity = x
        n, height, device = x.shape[2], x.shape[-2], x.device
        x = self.norm(x)
        x = rearrange(x, "b c f h w -> b (h w) f c")
        qkv = paddle.chunk(self.to_qkv(x), chunks=3, axis=-1)
        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values （v=qkv[-1]） through to the output
            values = qkv[-1]
            out = self.to_out(values)
            out = rearrange(out, "b (h w) f c -> b c f h w", h=height)
            return out + identity
        # split out heads
        # shape [b (hw) h n c/h], n=f
        q = rearrange(qkv[0], "... n (h d) -> ... h n d", h=self.heads)
        k = rearrange(qkv[1], "... n (h d) -> ... h n d", h=self.heads)
        v = rearrange(qkv[2], "... n (h d) -> ... h n d", h=self.heads)

        # scale

        q = q * self.scale
        # rotate positions into queries and keys for time attention
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity
        # shape [b (hw) h n n], n=f
        sim = paddle.einsum("... h i d, ... h j d -> ... h i j", q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias
        if focus_present_mask is None and video_mask is not None:
            # video_mask: [B, n]
            mask = video_mask[:, None, :] * video_mask[:, :, None]
            mask = mask.unsqueeze(1).unsqueeze(1)
            sim = self.masked_fill(sim, ~mask, -paddle.finfo(sim.dtype).max)
        elif exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = paddle.ones((n, n), dtype=paddle.bool)
            attend_self_mask = paddle.eye(n, dtype=paddle.bool)

            mask = paddle.where(
                rearrange(focus_present_mask, "b -> b 1 1 1 1"),
                rearrange(attend_self_mask, "i j -> 1 1 1 i j"),
                rearrange(attend_all_mask, "i j -> 1 1 1 i j"),
            )
            sim = self.masked_fill(sim, ~mask, -paddle.finfo(sim.dtype).max)
        if self.use_sim_mask:
            sim_mask = paddle.tril(paddle.ones((n, n), dtype=paddle.bool), diagonal=0)
            sim = self.masked_fill(sim, ~sim_mask, -paddle.finfo(sim.dtype).max)

        # numerical stability
        sim = sim - sim.amax(axis=-1, keepdim=True).detach()
        attn = sim.softmax(axis=-1)

        # aggregate values

        out = paddle.einsum("... h i j, ... h j d -> ... h i d", attn, v)
        out = rearrange(out, "... h n d -> ... n (h d)")
        out = self.to_out(out)

        out = rearrange(out, "b (h w) f c -> b c f h w", h=height)

        if self.use_image_dataset:
            out = identity + 0 * out
        else:
            out = identity + out
        return out


class TemporalAttentionMultiBlock(nn.Layer):
    def __init__(
        self,
        dim,
        heads=4,
        dim_head=32,
        rotary_emb=None,
        use_image_dataset=False,
        use_sim_mask=False,
        temporal_attn_times=1,
    ):
        super().__init__()
        self.att_layers = nn.LayerList(
            [
                TemporalAttentionBlock(dim, heads, dim_head, rotary_emb, use_image_dataset, use_sim_mask)
                for _ in range(temporal_attn_times)
            ]
        )

    def forward(self, x, pos_bias=None, focus_present_mask=None, video_mask=None):
        for layer in self.att_layers:
            x = layer(x, pos_bias, focus_present_mask, video_mask)
        return x


class SpatialTransformer(nn.Layer):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
    ):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = paddle.nn.GroupNorm(num_groups=32, num_channels=in_channels, epsilon=1e-06)

        if not use_linear:
            self.proj_in = nn.Conv2D(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.LayerList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                )
                for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2D(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class STUNetModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 6,
        dim: int = 512,
        y_dim: int = 512,
        num_tokens: int = 4,
        context_channels: int = 512,
        dim_mult: List[int] = [1, 2, 3, 4],
        num_heads=None,
        head_dim=64,
        num_res_blocks=3,
        attn_scales: List[float] = [1 / 2, 1 / 4, 1 / 8],
        use_scale_shift_norm=True,
        dropout=0.1,
        default_fps=8,
        temporal_attn_times=1,
        temporal_attention=True,
        inpainting=True,
        use_sim_mask=False,
        use_image_dataset=False,
        **kwargs
    ):
        embed_dim = dim * 4
        num_heads = num_heads if num_heads else dim // 32
        super().__init__()
        self.in_dim = in_channels
        self.num_tokens = num_tokens
        self.dim = dim
        self.y_dim = y_dim
        self.context_dim = context_channels
        self.embed_dim = embed_dim
        self.out_dim = out_channels
        self.dim_mult = dim_mult
        # for temporal attention
        self.num_heads = num_heads
        # for spatial attention
        self.default_fps = default_fps
        self.head_dim = head_dim
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.use_scale_shift_norm = use_scale_shift_norm
        self.temporal_attn_times = temporal_attn_times
        self.temporal_attention = temporal_attention
        self.inpainting = inpainting

        use_linear_in_temporal = False
        transformer_depth = 1
        disabled_sa = False
        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0

        # embeddings
        self.time_embed = nn.Sequential(nn.Linear(dim, embed_dim), nn.Silu(), nn.Linear(embed_dim, embed_dim))

        self.context_embedding = nn.Sequential(
            nn.Linear(y_dim, embed_dim), nn.Silu(), nn.Linear(embed_dim, context_channels * self.num_tokens)
        )

        self.fps_embedding = nn.Sequential(
            nn.Linear(dim, embed_dim),
            nn.Silu(),
            nn.Linear(
                embed_dim,
                embed_dim,
                weight_attr=nn.initializer.Constant(value=0.0),
                bias_attr=nn.initializer.Constant(value=0.0),
            ),
        )

        # encoder
        self.input_blocks = nn.LayerList()
        init_block = nn.LayerList([nn.Conv2D(self.in_dim, dim, 3, padding=1)])
        if temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                init_block.append(
                    TemporalTransformer(
                        dim,
                        num_heads,
                        head_dim,
                        depth=transformer_depth,
                        context_dim=context_channels,
                        disable_self_attn=disabled_sa,
                        use_linear=use_linear_in_temporal,
                        multiply_zero=use_image_dataset,
                    )
                )
            else:
                init_block.append(
                    TemporalAttentionMultiBlock(
                        dim,
                        num_heads,
                        head_dim,
                        rotary_emb=self.rotary_emb,
                        temporal_attn_times=temporal_attn_times,
                        use_image_dataset=use_image_dataset))

        self.input_blocks.append(init_block)
        shortcut_dims.append(dim)
        for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            for j in range(num_res_blocks):
                block = nn.LayerList(
                    [
                        ResBlock(
                            in_dim,
                            embed_dim,
                            dropout,
                            out_channels=out_dim,
                            use_scale_shift_norm=False,
                            use_image_dataset=use_image_dataset,
                        )
                    ]
                )
                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim,
                            out_dim // head_dim,
                            head_dim,
                            depth=1,
                            context_dim=self.context_dim,
                            disable_self_attn=False,
                            use_linear=True,
                        )
                    )
                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(
                                TemporalTransformer(
                                    out_dim,
                                    out_dim // head_dim,
                                    head_dim,
                                    depth=transformer_depth,
                                    context_dim=context_channels,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_temporal,
                                    multiply_zero=use_image_dataset,
                                )
                            )
                        else:
                            block.append(
                                TemporalAttentionMultiBlock(
                                    out_dim,
                                    num_heads,
                                    head_dim,
                                    rotary_emb=self.rotary_emb,
                                    use_image_dataset=use_image_dataset,
                                    use_sim_mask=use_sim_mask,
                                    temporal_attn_times=temporal_attn_times,
                                )
                            )
                in_dim = out_dim
                self.input_blocks.append(block)
                shortcut_dims.append(out_dim)

                # downsample
                if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                    downsample = Downsample(out_dim, True, dims=2, out_channels=out_dim)
                    shortcut_dims.append(out_dim)
                    scale /= 2.0
                    self.input_blocks.append(downsample)

        self.middle_block = nn.LayerList(
            [
                ResBlock(
                    out_dim,
                    embed_dim,
                    dropout,
                    use_scale_shift_norm=False,
                    use_image_dataset=use_image_dataset,
                ),
                SpatialTransformer(
                    out_dim,
                    out_dim // head_dim,
                    head_dim,
                    depth=1,
                    context_dim=self.context_dim,
                    disable_self_attn=False,
                    use_linear=True,
                ),
            ]
        )

        if self.temporal_attention:
            if USE_TEMPORAL_TRANSFORMER:
                self.middle_block.append(
                    TemporalTransformer(
                        out_dim,
                        out_dim // head_dim,
                        head_dim,
                        depth=transformer_depth,
                        context_dim=context_channels,
                        disable_self_attn=disabled_sa,
                        use_linear=use_linear_in_temporal,
                        multiply_zero=use_image_dataset,
                    )
                )
            else:
                self.middle_block.append(
                    TemporalAttentionMultiBlock(
                        out_dim,
                        num_heads,
                        head_dim,
                        rotary_emb=self.rotary_emb,
                        use_image_dataset=use_image_dataset,
                        use_sim_mask=use_sim_mask,
                        temporal_attn_times=temporal_attn_times,
                    )
                )

        self.middle_block.append(ResBlock(out_dim, embed_dim, dropout, use_scale_shift_norm=False))
        # decoder
        self.output_blocks = nn.LayerList()
        for i, (in_dim, out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
            for j in range(num_res_blocks + 1):
                block = nn.LayerList(
                    [
                        ResBlock(
                            in_dim + shortcut_dims.pop(),
                            embed_dim,
                            dropout,
                            out_dim,
                            use_scale_shift_norm=False,
                            use_image_dataset=use_image_dataset,
                        )
                    ]
                )
                if scale in attn_scales:
                    block.append(
                        SpatialTransformer(
                            out_dim,
                            out_dim // head_dim,
                            head_dim,
                            depth=1,
                            context_dim=1024,
                            disable_self_attn=False,
                            use_linear=True,
                        )
                    )

                    if self.temporal_attention:
                        if USE_TEMPORAL_TRANSFORMER:
                            block.append(
                                TemporalTransformer(
                                    out_dim,
                                    out_dim // head_dim,
                                    head_dim,
                                    depth=transformer_depth,
                                    context_dim=context_channels,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_temporal,
                                    multiply_zero=use_image_dataset,
                                )
                            )
                        else:
                            block.append(
                                TemporalAttentionMultiBlock(
                                    out_dim,
                                    num_heads,
                                    head_dim,
                                    rotary_emb=self.rotary_emb,
                                    use_image_dataset=use_image_dataset,
                                    use_sim_mask=use_sim_mask,
                                    temporal_attn_times=temporal_attn_times,
                                )
                            )

                in_dim = out_dim

                # upsample
                if i != len(dim_mult) - 1 and j == num_res_blocks:
                    upsample = Upsample(out_dim, True, dims=2, out_channels=out_dim)
                    scale *= 2.0
                    block.append(upsample)
                self.output_blocks.append(block)

        # head
        self.out = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.Silu(),
            nn.Conv2D(out_dim, self.out_dim, 3, padding=1, weight_attr=nn.initializer.Constant(value=0.0)),
        )

    def forward(
        self,
        x,
        t,
        y,
        fps=None,
        video_mask=None,
        focus_present_mask=None,
        prob_focus_present=0.0,
        mask_last_frame_num=0,
        return_dict: bool = True,
        **kwargs
    ) -> Union[STUNetOutput, Tuple]:
        batch, c, f, h, w = x.shape
        device = x.place
        self.batch = batch
        if fps is None:
            fps = paddle.to_tensor([self.default_fps] * batch, dtype=paddle.int64, place=device)
        # image and video joint training, if mask_last_frame_num is set, prob_focus_present will be ignored
        if mask_last_frame_num > 0:
            focus_present_mask = None
            video_mask[-mask_last_frame_num:] = False
        else:
            focus_present_mask = default(
                focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device=device)
            )

        time_rel_pos_bias = None
        # embeddings
        embeddings = self.time_embed(sinusoidal_embedding_paddle(t, self.dim)) + self.fps_embedding(
            sinusoidal_embedding_paddle(fps, self.dim)
        )
        context = self.context_embedding(y)
        context = context.reshape([-1, self.num_tokens, self.context_dim])

        # repeat f times for spatial e and context
        embeddings = embeddings.repeat_interleave(repeats=f, axis=0)
        context = context.repeat_interleave(repeats=f, axis=0)

        # always in shape (b f) c h w, except for temporal layer
        x = rearrange(x, "b c f h w -> (b f) c h w")

        # encoder
        xs = []
        for block in self.input_blocks:
            x = self._forward_single(block, x, embeddings, context, time_rel_pos_bias, focus_present_mask, video_mask)
            xs.append(x)

        # middle
        for block in self.middle_block:
            x = self._forward_single(block, x, embeddings, context, time_rel_pos_bias, focus_present_mask, video_mask)

        # decoder
        for block in self.output_blocks:
            x = paddle.concat([x, xs.pop()], axis=1)
            x = self._forward_single(
                block,
                x,
                embeddings,
                context,
                time_rel_pos_bias,
                focus_present_mask,
                video_mask,
                reference=xs[-1] if len(xs) > 0 else None,
            )

        # head
        x = self.out(x)

        # reshape back to (b c f h w)
        sample = rearrange(x, "(b f) c h w -> b c f h w", b=batch)

        if not return_dict:
            return (sample,)

        return STUNetOutput(sample=sample)

    def _forward_single(
        self, module, x, e, context, time_rel_pos_bias, focus_present_mask, video_mask, reference=None
    ):
        if isinstance(module, ResBlock):
            x = module(x, e, self.batch)
        elif isinstance(module, SpatialTransformer):
            x = module(x, context)
        elif isinstance(module, TemporalTransformer):
            x = rearrange(x, "(b f) c h w -> b c f h w", b=self.batch)
            x = module(x, context)
            x = rearrange(x, "b c f h w -> (b f) c h w")
        elif isinstance(module, MemoryEfficientCrossAttention):
            x = module(x, context)
        elif isinstance(module, BasicTransformerBlock):
            x = module(x, context)
        elif isinstance(module, FeedForward):
            x = module(x, context)
        elif isinstance(module, Upsample):
            x = module(x)
        elif isinstance(module, Downsample):
            x = module(x)
        elif isinstance(module, TemporalAttentionBlock):
            x = rearrange(x, "(b f) c h w -> b c f h w", b=self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, "b c f h w -> (b f) c h w")
        elif isinstance(module, TemporalAttentionMultiBlock):
            x = rearrange(x, "(b f) c h w -> b c f h w", b=self.batch)
            x = module(x, time_rel_pos_bias, focus_present_mask, video_mask)
            x = rearrange(x, "b c f h w -> (b f) c h w")
        elif isinstance(module, nn.LayerList):
            for block in module:
                x = self._forward_single(
                    block, x, e, context, time_rel_pos_bias, focus_present_mask, video_mask, reference
                )
        else:
            x = module(x)
        return x
