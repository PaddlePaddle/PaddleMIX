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

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# PixArt: https://github.com/PixArt-alpha/PixArt-alpha
# Latte:  https://github.com/Vchitect/Latte
# DiT:    https://github.com/facebookresearch/DiT/tree/main
# GLIDE:  https://github.com/openai/glide-text2im
# MAE:    https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import collections.abc
import functools
import math
from functools import partial
from math import pi

import numpy as np
import paddle
import paddle.nn.functional as F
from beartype import beartype
from beartype.typing import Literal, Optional, Union
from einops import rearrange, repeat
from paddle.incubate.nn.attn_bias import BlockDiagonalMask
from paddle.incubate.nn.memory_efficient_attention import memory_efficient_attention

from ppdiffusers.models.dit_llama import TimestepEmbedder

approx_gelu = lambda: paddle.nn.GELU(approximate=True)


class LlamaRMSNorm(paddle.nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        out_0 = paddle.create_parameter(
            shape=paddle.ones(shape=hidden_size).shape,
            dtype=paddle.ones(shape=hidden_size).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=hidden_size)),
        )
        out_0.stop_gradient = not True
        self.weight = out_0
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to("float32")
        variance = hidden_states.pow(y=2).mean(axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(x=variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def get_layernorm(hidden_size: paddle.Tensor, eps: float, affine: bool, use_kernel: bool):

    return paddle.nn.LayerNorm(normalized_shape=hidden_size, epsilon=eps, weight_attr=affine, bias_attr=affine)


def modulate(norm_func, x, shift, scale):
    # Suppose x is (B, N, D), shift is (B, D), scale is (B, D)
    dtype = x.dtype
    x = norm_func(x.to("float32")).to(dtype)
    x = x * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
    x = x.to(dtype)
    return x


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


# ===============================================
# General-purpose Layers
# ===============================================


class PatchEmbed3D(paddle.nn.Layer):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = paddle.nn.Conv3D(
            in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
        )

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding

        _, _, D, H, W = x.shape

        if W % self.patch_size[2] != 0:

            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2], 0, 0, 0, 0), data_format="NCDHW")
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1], 0, 0), data_format="NCDHW")
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]), data_format="NCDHW")

        x = self.proj(x)  # (B C T H W)

        if self.norm is not None:
            D, Wh, Ww = x.shape[2], x.shape[3], x.shape[4]
            x = x.flatten(start_axis=2)
            perm_0 = list(range(x.ndim))
            perm_0[1] = 2
            perm_0[2] = 1
            x = x.transpose(perm=perm_0)
            x = self.norm(x)
            x = x
            perm_1 = list(range(x.ndim))
            perm_1[1] = 2
            perm_1[2] = 1
            x = x.transpose(perm=perm_1).reshape([-1, self.embed_dim, D, Wh, Ww])  # view
        if self.flatten:
            x = x.flatten(start_axis=2)
            perm_2 = list(range(x.ndim))
            perm_2[1] = 2
            perm_2[2] = 1
            x = x.transpose(perm=perm_2)

        return x


class Attention(paddle.nn.Layer):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: paddle.nn.Layer = LlamaRMSNorm,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = paddle.nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else paddle.nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else paddle.nn.Identity()
        self.attn_drop = paddle.nn.Dropout(attn_drop)
        self.proj = paddle.nn.Linear(dim, dim)
        self.proj_drop = paddle.nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope

        self._use_memory_efficient_attention_xformers = True

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.reshape(qkv_shape).permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim) # view # permute
        q, k, v = qkv.unbind(0)  # (B, num_heads, N, head_dim)

        # noq
        if self.rope:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)
        q, k = self.q_norm(q), self.k_norm(k)

        if self._use_memory_efficient_attention_xformers:
            q = q.transpose((0, 2, 1, 3))
            k = k.transpose((0, 2, 1, 3))
            v = v.transpose((0, 2, 1, 3))
            attn = F.scaled_dot_product_attention_(
                q,
                k,
                v,
                scale=self.scale,
                dropout_p=self.attn_drop.p,
                attention_op=None,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose((0, 1, 3, 2))
            attn = attn.astype(paddle.float32)
            attn = paddle.nn.functional.softmax(attn, axis=-1)
            attn = attn.astype(x.dtype)
            attn = self.attn_drop(attn)
            attn = attn @ v
            attn = attn.transpose((0, 2, 1, 3))

        x = attn
        x = x.reshape((B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def transform_list(input_list):

    output_list = []
    cumulative_value = 0

    for value in input_list:
        tuple_item = (cumulative_value, cumulative_value + value)
        output_list.append(tuple_item)
        cumulative_value += value

    return output_list


class MultiHeadCrossAttention(paddle.nn.Layer):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / self.head_dim**0.5

        self.q_linear = paddle.nn.Linear(d_model, d_model)
        self.kv_linear = paddle.nn.Linear(d_model, d_model * 2)
        self.attn_drop = paddle.nn.Dropout(attn_drop)
        self.proj = paddle.nn.Linear(d_model, d_model)
        self.proj_drop = paddle.nn.Dropout(proj_drop)

        self._use_memory_efficient_attention_xformers = True

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).reshape((1, -1, self.num_heads, self.head_dim))
        kv = self.kv_linear(cond).reshape((1, -1, 2, self.num_heads, self.head_dim))
        k, v = kv.unbind(2)

        attn_bias_shape = [1, self.num_heads, q.shape[1], k.shape[1]]
        # attn_bias
        attn_bias = paddle.empty(shape=attn_bias_shape[-2:], dtype=q.dtype)
        attn_bias.fill_(-math.inf)
        for i, ((q_start, q_end), (k_start, k_end)) in enumerate(
            zip(
                transform_list([N] * B),
                transform_list(mask),
            )
        ):

            attn_bias[q_start:q_end, k_start:k_end] = paddle.zeros(
                (q_end - q_start, k_end - k_start),
                dtype=q.dtype,
            )
        for _ in range(len(attn_bias_shape) - 2):
            attn_bias = attn_bias.unsqueeze(0)
        attn_bias = attn_bias.expand(attn_bias_shape)

        if self._use_memory_efficient_attention_xformers:
            q = q.astype(paddle.float32)
            k = k.astype(paddle.float32)
            v = v.astype(paddle.float32)

            if mask is not None:
                attn_bias_4mea = BlockDiagonalMask.from_seqlens([N] * B, mask)
            attn = memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias_4mea)

        else:

            q = q.transpose((0, 2, 1, 3)).astype(paddle.float32)
            k = k.transpose((0, 2, 1, 3)).astype(paddle.float32)
            v = v.transpose((0, 2, 1, 3)).astype(paddle.float32)

            attention_scores = paddle.matmul(q * self.scale, k, transpose_y=True)
            attention_scores = attention_scores + attn_bias

            attention_probs = F.softmax(attention_scores, axis=-1)
            attention_probs = self.attn_drop(attention_probs)
            attn = paddle.matmul(attention_probs, v).cast(x.dtype)
            attn = attn.transpose((0, 2, 1, 3))

        x = attn

        x = x.reshape((B, -1, C))
        x = x.astype(paddle.float32)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class T2IFinalLayer(paddle.nn.Layer):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, num_patch, out_channels, d_t=None, d_s=None):
        super().__init__()

        self.norm_final = paddle.nn.LayerNorm(
            normalized_shape=hidden_size, weight_attr=False, bias_attr=False, epsilon=1e-06
        )
        self.linear = paddle.nn.Linear(in_features=hidden_size, out_features=num_patch * out_channels, bias_attr=True)

        paddle_random = paddle.randn([2, hidden_size])

        out_1 = paddle.create_parameter(
            shape=(paddle_random / hidden_size**0.5).shape,
            dtype=(paddle_random / hidden_size**0.5).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle_random / hidden_size**0.5),
        )

        out_1.stop_gradient = not True
        self.scale_shift_table = out_1

        self.out_channels = out_channels
        self.d_t = d_t
        self.d_s = d_s

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)

        x = paddle.where(condition=x_mask[:, :, None, None], x=x, y=masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(self, x, t, x_mask=None, t0=None, T=None, S=None):
        if T is None:
            T = self.d_t
        if S is None:
            S = self.d_s

        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(chunks=2, axis=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        if x_mask is not None:

            shift_zero, scale_zero = (self.scale_shift_table[None] + t0[:, None]).chunk(chunks=2, axis=1)
            x_zero = t2i_modulate(self.norm_final(x), shift_zero, scale_zero)
            x = self.t_mask_select(x_mask, x, x_zero, T, S)
        x = self.linear(x)
        return x


# ===============================================
# Embedding Layers for Timesteps and Class Labels
# ===============================================


class SizeEmbedder(TimestepEmbedder):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = paddle.nn.Sequential(
            paddle.nn.Linear(frequency_embedding_size, hidden_size, bias_attr=True),
            paddle.nn.Silu(),
            paddle.nn.Linear(hidden_size, hidden_size, bias_attr=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    def forward(self, s, bs):
        if s.ndim == 1:
            s = s[:, None]
        assert s.ndim == 2
        if s.shape[0] != bs:

            s = s.tile((bs // s.shape[0], 1))
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)

        s_emb = self.mlp(s_freq)
        s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return s_emb

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class CaptionEmbedder(paddle.nn.Layer):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        uncond_prob,
        act_layer=paddle.nn.GELU(approximate=True),
        token_num=120,
    ):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            drop=0,
        )

        paddle_random = paddle.randn([token_num, in_channels])

        self.register_buffer(
            "y_embedding",
            tensor=paddle_random,
        )

        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            paddle_random = paddle.randn([tuple(caption.shape)[0]])

            drop_ids = paddle_random.cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1

        caption = paddle.where(condition=drop_ids[:, None, None, None], x=self.y_embedding, y=caption)

        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class PositionEmbedding2D(paddle.nn.Layer):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "dim must be divisible by 4"
        half_dim = dim // 2

        inv_freq = 1.0 / 10000 ** (paddle.arange(start=0, end=half_dim, step=2).astype("float32") / half_dim)
        self.register_buffer("inv_freq", inv_freq, persistable=False)

    def _get_sin_cos_emb(self, t: paddle.Tensor):

        out = paddle.einsum("i,d->id", t, self.inv_freq.astype(t.dtype))
        emb_cos = paddle.cos(x=out)
        emb_sin = paddle.sin(x=out)
        return paddle.concat(x=(emb_sin, emb_cos), axis=-1)

    @functools.lru_cache(maxsize=512)
    def _get_cached_emb(
        self,
        dtype: paddle.dtype,
        h: int,
        w: int,
        scale: float = 1.0,
        base_size: Optional[int] = None,
    ):

        grid_h = paddle.arange(end=h) / scale
        grid_w = paddle.arange(end=w) / scale

        if base_size is not None:
            grid_h *= base_size / h
            grid_w *= base_size / w

        grid_h, grid_w = paddle.meshgrid(grid_w, grid_h)

        grid_h = grid_h.t().reshape([-1])
        grid_w = grid_w.t().reshape([-1])
        emb_h = self._get_sin_cos_emb(grid_h)
        emb_w = self._get_sin_cos_emb(grid_w)
        return paddle.concat(x=[emb_h, emb_w], axis=-1).unsqueeze(0).to(dtype)

    def forward(
        self,
        x: paddle.Tensor,
        h: int,
        w: int,
        scale: Optional[float] = 1.0,
        base_size: Optional[int] = None,
    ) -> paddle.Tensor:
        return self._get_cached_emb(x.dtype, h, w, scale, base_size)


# ===============================================
# Sine/Cosine Positional Embedding Functions
# ===============================================
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


# RotaryEmbedding related module
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# broadcast, as tortoise-tts was using it


def broadcast(tensors, dim=-1):
    broadcasted_tensors = paddle.broadcast_tensors(tensors)
    return paddle.concat(x=broadcasted_tensors, axis=dim)


# rotary embedding helper functions


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(axis=-1)
    x = paddle.stack(x=(-x2, x1), axis=-1)
    return rearrange(x, "... d r -> ... (d r)")


@paddle.amp.auto_cast(enable=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:].to(t)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return paddle.concat(x=(t_left, t, t_right), axis=-1)


# learned rotation helpers


def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = paddle.einsum("..., f -> ... f", rotations, freq_ranges)
        rotations = rearrange(rotations, "... r f -> ... (r f)")

    rotations = repeat(rotations, "... n -> ... (n r)", r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)


# classes


class RotaryEmbedding(paddle.nn.Layer):
    @beartype
    def __init__(
        self,
        dim,
        custom_freqs: Optional[paddle.Tensor] = None,
        freqs_for: Union[Literal["lang"], Literal["pixel"], Literal["constant"]] = "lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        use_xpos=False,
        xpos_scale_base=512,
        interpolate_factor=1.0,
        theta_rescale_factor=1.0,
        seq_before_head_dim=False,
        cache_if_possible=True,
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / theta ** (paddle.arange(start=0, end=dim, step=2)[: dim // 2].astype(dtype="float32") / dim)
        elif freqs_for == "pixel":
            freqs = paddle.linspace(start=1.0, stop=max_freq / 2, num=dim // 2) * pi
        elif freqs_for == "constant":
            freqs = paddle.ones(shape=num_freqs).astype(dtype="float32")

        self.cache_if_possible = cache_if_possible

        self.tmp_store("cached_freqs", None)
        self.tmp_store("cached_scales", None)

        out_0 = paddle.create_parameter(
            shape=freqs.shape, dtype=freqs.numpy().dtype, default_initializer=paddle.nn.initializer.Assign(freqs)
        )
        out_0.stop_gradient = not learned_freq
        self.freqs = out_0

        self.learned_freq = learned_freq

        # dummy for device

        self.tmp_store("dummy", paddle.to_tensor(data=0))

        # default sequence dimension

        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors

        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor

        # xpos

        self.use_xpos = use_xpos
        if not use_xpos:
            self.tmp_store("scale", None)
            return

        scale = (paddle.arange(start=0, end=dim, step=2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.tmp_store("scale", scale)

    def tmp_store(self, key, value):
        self.register_buffer(key, value, persistable=False)

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (paddle.arange(dtype=dtype, end=seq_len) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0, freq_seq_len=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert (
            not self.use_xpos
        ), "you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings"

        device, dtype, seq_len = t.place, t.dtype, t.shape[seq_dim]

        if exists(freq_seq_len):
            assert freq_seq_len >= seq_len
            seq_len = freq_seq_len

        freqs = self.forward(
            self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset), seq_len=seq_len, offset=offset
        )

        if seq_dim == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")

        return apply_rotary_emb(freqs, t, seq_dim=seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim=None, offset=0):
        seq_dim = default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        assert q_len <= k_len
        rotated_q = self.rotate_queries_or_keys(q, seq_dim=seq_dim, freq_seq_len=k_len)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim=seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim=None):
        seq_dim = default(seq_dim, self.default_seq_dim)

        assert self.use_xpos
        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)

        freqs = self.forward(seq, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")
            scale = rearrange(scale, "n d -> n 1 d")

        rotated_q = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale**-1, seq_dim=seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    @beartype
    def get_scale(self, t: paddle.Tensor, seq_len: Optional[int] = None, offset=0):
        assert self.use_xpos

        should_cache = self.cache_if_possible and exists(seq_len)

        if should_cache and exists(self.cached_scales) and (seq_len + offset) <= self.cached_scales.shape[0]:
            return self.cached_scales[offset : (offset + seq_len)]

        scale = 1.0
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, "n -> n 1")
            scale = paddle.concat(x=(scale, scale), axis=-1)

        if should_cache:
            self.tmp_store("cached_scales", scale)

        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == "pixel":
                pos = paddle.linspace(start=-1, stop=1, num=dim)
            else:
                pos = paddle.arange(end=dim)

            freqs = self.forward(pos, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = paddle.broadcast_tensors(all_freqs)
        return paddle.concat(x=all_freqs, axis=-1)

    @paddle.amp.auto_cast(enable=False)
    def forward(self, t: paddle.Tensor, seq_len=None, offset=0):
        should_cache = (
            self.cache_if_possible and not self.learned_freq and exists(seq_len) and self.freqs_for != "pixel"
        )

        if should_cache and exists(self.cached_freqs) and (offset + seq_len) <= self.cached_freqs.shape[0]:
            return self.cached_freqs[offset : (offset + seq_len)].detach()

        freqs = self.freqs

        freqs = paddle.einsum("..., f -> ... f", t.astype(freqs.dtype), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        if should_cache:
            self.tmp_store("cached_freqs", freqs.detach())

        return freqs


# DropPath
def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets

    random_tensor = x.new_empty(shape)

    tensor = paddle.to_tensor([keep_prob])
    paddle.assign(paddle.bernoulli(paddle.broadcast_to(tensor, random_tensor.shape)), random_tensor)

    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = x.divide_(keep_prob)
    return x * random_tensor


class DropPath(paddle.nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, scale=1.0, base_size=None):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / scale
    if base_size is not None:
        grid_h *= base_size / grid_size[0]
        grid_w *= base_size / grid_size[1]
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


# Mlp


def _ntuple(n):
    def parse(x):
        from itertools import repeat as repeat_itertools

        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat_itertools(x, n))

    return parse


to_2tuple = _ntuple(2)


class Mlp(paddle.nn.Layer):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=paddle.nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        linear_layer = partial(paddle.nn.Conv2D, kernel_size=1) if use_conv else paddle.nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias_attr=bias[0])
        self.act = act_layer()
        self.drop1 = paddle.nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else paddle.nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias_attr=bias[1])
        self.drop2 = paddle.nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
