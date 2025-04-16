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

import logging
import math
import os
from argparse import Namespace
from dataclasses import dataclass
from functools import partial
from math import pi
from typing import Optional, Tuple, Union

import paddle
from einops import rearrange, repeat
from paddle.incubate.nn.memory_efficient_attention import memory_efficient_attention
from paddlenlp.transformers.activations import ACT2FN
from paddlenlp.transformers.bit.modeling import drop_path


def broadcast(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = dim + shape_len if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatenation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(shape=t[1]), zip(tensors, expandable_shapes)))
    return paddle.concat(x=tensors, axis=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(axis=-1)
    x = paddle.stack(x=(-x2, x1), axis=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbeddingFast(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        patch_dropout=0.0,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / theta ** (paddle.arange(start=0, end=dim, step=2)[: dim // 2].astype(dtype="float32") / dim)
        elif freqs_for == "pixel":
            freqs = paddle.linspace(start=1.0, stop=max_freq / 2, num=dim // 2) * pi
        elif freqs_for == "constant":
            freqs = paddle.ones(shape=num_freqs).astype(dtype="float32")
        else:
            raise ValueError(f"unknown modality {freqs_for}")
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = paddle.arange(end=ft_seq_len) / ft_seq_len * pt_seq_len
        freqs = paddle.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcast((freqs[:, None, :], freqs[None, :, :]), dim=-1)
        freqs_cos = freqs.cos().reshape([-1, freqs.shape[-1]])
        freqs_sin = freqs.sin().reshape([-1, freqs.shape[-1]])
        self.patch_dropout = patch_dropout
        self.register_buffer(name="freqs_cos", tensor=freqs_cos)
        self.register_buffer(name="freqs_sin", tensor=freqs_sin)
        logging.info(f"Shape of rope freq: {self.freqs_cos.shape}")

    def forward(self, t, patch_indices_keep=None):
        if patch_indices_keep is not None:
            batch = t.shape[0]
            batch_indices = paddle.arange(end=batch)
            batch_indices = batch_indices[..., None]
            freqs_cos = repeat(self.freqs_cos, "i j -> n i m j", n=t.shape[0], m=t.shape[1])
            freqs_sin = repeat(self.freqs_sin, "i j -> n i m j", n=t.shape[0], m=t.shape[1])
            freqs_cos = freqs_cos[batch_indices, patch_indices_keep]
            freqs_cos = rearrange(freqs_cos, "n i m j -> n m i j")
            freqs_sin = freqs_sin[batch_indices, patch_indices_keep]
            freqs_sin = rearrange(freqs_sin, "n i m j -> n m i j")
            return t * freqs_cos + rotate_half(t) * freqs_sin
        self.freqs_cos = self.freqs_cos.astype(t.dtype)
        self.freqs_sin = self.freqs_sin.astype(t.dtype)
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin


class PatchDropout(paddle.nn.Layer):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token
        logging.info(f"os.getenv('RoPE')={os.getenv('RoPE')}")

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x
        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        batch = x.shape[0]
        num_tokens = x.shape[1]
        batch_indices = paddle.arange(end=batch)
        batch_indices = batch_indices[..., None]
        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))
        rand = paddle.randn(shape=[batch, num_tokens])
        patch_indices_keep = rand.topk(k=num_patches_keep, axis=-1).indices
        x = x[batch_indices, patch_indices_keep]
        if self.exclude_first_token:
            x = paddle.concat(x=(cls_tokens, x), axis=1)
        if self.training and os.getenv("RoPE") == "1":
            return x, patch_indices_keep
        return x


class DropPath(paddle.nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(paddle.nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=paddle.nn.GELU,
        norm_layer=paddle.nn.LayerNorm,
        drop=0.0,
        subln=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = paddle.nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else paddle.nn.Identity()
        self.fc2 = paddle.nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = paddle.nn.Dropout(p=drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.ffn_ln(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLU(paddle.nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=paddle.nn.Silu,
        drop=0.0,
        norm_layer=paddle.nn.LayerNorm,
        subln=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w1 = paddle.nn.Linear(in_features=in_features, out_features=hidden_features)
        self.w2 = paddle.nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else paddle.nn.Identity()
        self.w3 = paddle.nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = paddle.nn.Dropout(p=drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


class Attention(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=None,
        attn_head_dim=None,
        xattn=False,
        rope=None,
        subln=False,
        norm_layer=paddle.nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.subln = subln
        if self.subln:
            self.q_proj = paddle.nn.Linear(in_features=dim, out_features=all_head_dim, bias_attr=False)
            self.k_proj = paddle.nn.Linear(in_features=dim, out_features=all_head_dim, bias_attr=False)
            self.v_proj = paddle.nn.Linear(in_features=dim, out_features=all_head_dim, bias_attr=False)
        else:
            self.qkv = paddle.nn.Linear(in_features=dim, out_features=all_head_dim * 3, bias_attr=False)
        if qkv_bias:
            out_0 = paddle.create_parameter(
                shape=paddle.zeros(shape=all_head_dim).shape,
                dtype=paddle.zeros(shape=all_head_dim).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=all_head_dim)),
            )
            out_0.stop_gradient = not True
            self.q_bias = out_0
            out_1 = paddle.create_parameter(
                shape=paddle.zeros(shape=all_head_dim).shape,
                dtype=paddle.zeros(shape=all_head_dim).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=all_head_dim)),
            )
            out_1.stop_gradient = not True
            self.v_bias = out_1
        else:
            self.q_bias = None
            self.v_bias = None
        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            out_2 = paddle.create_parameter(
                shape=paddle.zeros(shape=[self.num_relative_distance, num_heads]).shape,
                dtype=paddle.zeros(shape=[self.num_relative_distance, num_heads]).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(
                    paddle.zeros(shape=[self.num_relative_distance, num_heads])
                ),
            )
            out_2.stop_gradient = not True
            self.relative_position_bias_table = out_2
            coords_h = paddle.arange(end=window_size[0])
            coords_w = paddle.arange(end=window_size[1])
            coords = paddle.stack(x=paddle.meshgrid([coords_h, coords_w]))
            coords_flatten = paddle.flatten(x=coords, start_axis=1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.transpose(perm=[1, 2, 0])
            relative_coords[:, :, 0] += window_size[0] - 1
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = paddle.zeros(
                shape=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
            )
            relative_position_index[1:, 1:] = relative_coords.sum(axis=-1)
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
            self.register_buffer(name="relative_position_index", tensor=relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None
        self.attn_drop = paddle.nn.Dropout(p=attn_drop)
        self.inner_attn_ln = norm_layer(all_head_dim) if subln else paddle.nn.Identity()
        self.proj = paddle.nn.Linear(in_features=all_head_dim, out_features=dim)
        self.proj_drop = paddle.nn.Dropout(p=proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop
        self.rope = rope

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape
        if self.subln:
            if self.q_proj.weight.dtype == "uint8":
                import bitsandbytes as bnb

                q = bnb.matmul_4bit(
                    x, self.q_proj.weight.t(), bias=self.q_bias, quant_state=self.q_proj.weight.quant_state
                )
                k = bnb.matmul_4bit(x, self.k_proj.weight.t(), bias=None, quant_state=self.k_proj.weight.quant_state)
                v = bnb.matmul_4bit(
                    x, self.v_proj.weight.t(), bias=self.v_bias, quant_state=self.v_proj.weight.quant_state
                )
            else:
                q = paddle.nn.functional.linear(weight=self.q_proj.weight, bias=self.q_bias, x=x)
                k = paddle.nn.functional.linear(weight=self.k_proj.weight, bias=None, x=x)
                v = paddle.nn.functional.linear(weight=self.v_proj.weight, bias=self.v_bias, x=x)
            q = q.reshape([B, N, self.num_heads, -1]).transpose(perm=[0, 2, 1, 3])
            k = k.reshape([B, N, self.num_heads, -1]).transpose(perm=[0, 2, 1, 3])
            v = v.reshape([B, N, self.num_heads, -1]).transpose(perm=[0, 2, 1, 3])
        else:
            qkv_bias = None
            if self.q_bias is not None:
                out_3 = paddle.zeros_like(x=self.v_bias)
                out_3.stop_gradient = not False
                qkv_bias = paddle.concat(x=(self.q_bias, out_3, self.v_bias))
            qkv = paddle.nn.functional.linear(weight=self.qkv.weight.T, bias=qkv_bias, x=x)
            qkv = qkv.reshape([B, N, 3, self.num_heads, -1]).transpose(perm=[2, 0, 3, 1, 4])
            q, k, v = qkv[0], qkv[1], qkv[2]
        if self.rope:
            q_t = q[:, :, 1:, :]
            ro_q_t = self.rope(q_t)
            q = paddle.concat(x=(q[:, :, :1, :], ro_q_t), axis=-2).astype(dtype=v.dtype)
            k_t = k[:, :, 1:, :]
            ro_k_t = self.rope(k_t)
            k = paddle.concat(x=(k[:, :, :1, :], ro_k_t), axis=-2).astype(dtype=v.dtype)
        if self.xattn:
            q = q.transpose(perm=[0, 2, 1, 3])
            k = k.transpose(perm=[0, 2, 1, 3])
            v = v.transpose(perm=[0, 2, 1, 3])
            x = memory_efficient_attention(q, k, v, p=self.xattn_drop, scale=self.scale)
            x = x.reshape([B, N, -1])
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            q = q * self.scale
            x = k
            perm_0 = list(range(x.ndim))
            perm_0[-2] = -1
            perm_0[-1] = -2
            attn = q @ x.transpose(perm=perm_0)
            if self.relative_position_bias_table is not None:
                relative_position_bias = self.relative_position_bias_table[
                    self.relative_position_index.reshape(-1)
                ].reshape(
                    [self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1]
                )
                relative_position_bias = relative_position_bias.transpose(perm=[2, 0, 1])
                attn = attn + relative_position_bias.unsqueeze(axis=0).astype(dtype=attn.dtype)
            if rel_pos_bias is not None:
                attn = attn + rel_pos_bias.astype(dtype=attn.dtype)
            if attn_mask is not None:
                attn_mask = attn_mask.astype(dtype="bool")
                attn = paddle.where(~attn_mask[:, None, None, :], attn, float("-inf"))
            attn = paddle.nn.functional.softmax(attn, axis=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
            perm_1 = list(range(x.ndim))
            perm_1[1] = 2
            perm_1[2] = 1
            x = x.transpose(perm=perm_1).reshape([B, N, -1])
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Block(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=paddle.nn.GELU,
        norm_layer=paddle.nn.LayerNorm,
        window_size=None,
        attn_head_dim=None,
        xattn=False,
        rope=None,
        postnorm=False,
        subln=False,
        naiveswiglu=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim,
            xattn=xattn,
            rope=rope,
            subln=subln,
            norm_layer=norm_layer,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if naiveswiglu:
            self.mlp = SwiGLU(in_features=dim, hidden_features=mlp_hidden_dim, subln=subln, norm_layer=norm_layer)
        else:
            self.mlp = Mlp(
                in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, subln=subln, drop=drop
            )
        if init_values is not None and init_values > 0:
            out_4 = paddle.create_parameter(
                shape=(init_values * paddle.ones(shape=dim)).shape,
                dtype=(init_values * paddle.ones(shape=dim)).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(init_values * paddle.ones(shape=dim)),
            )
            out_4.stop_gradient = not True
            self.gamma_1 = out_4
            out_5 = paddle.create_parameter(
                shape=(init_values * paddle.ones(shape=dim)).shape,
                dtype=(init_values * paddle.ones(shape=dim)).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(init_values * paddle.ones(shape=dim)),
            )
            out_5.stop_gradient = not True
            self.gamma_2 = out_5
        else:
            self.gamma_1, self.gamma_2 = None, None
        self.postnorm = postnorm

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        elif self.postnorm:
            x = x + self.drop_path(
                self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
            )
            x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
        else:
            x = x + self.drop_path(
                self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
            )
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(paddle.nn.Layer):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0])
        self.patch_shape = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = paddle.nn.Conv2D(
            in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(start_axis=2)
        perm_2 = list(range(x.ndim))
        perm_2[1] = 2
        perm_2[2] = 1
        x = x.transpose(perm=perm_2)
        return x


class RelativePositionBias(paddle.nn.Layer):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        out_6 = paddle.create_parameter(
            shape=paddle.zeros(shape=[self.num_relative_distance, num_heads]).shape,
            dtype=paddle.zeros(shape=[self.num_relative_distance, num_heads]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.zeros(shape=[self.num_relative_distance, num_heads])
            ),
        )
        out_6.stop_gradient = not True
        self.relative_position_bias_table = out_6
        coords_h = paddle.arange(end=window_size[0])
        coords_w = paddle.arange(end=window_size[1])
        coords = paddle.stack(x=paddle.meshgrid([coords_h, coords_w]))
        coords_flatten = paddle.flatten(x=coords, start_axis=1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose(perm=[1, 2, 0])
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = paddle.zeros(
            shape=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(axis=-1)
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.register_buffer(name="relative_position_index", tensor=relative_position_index)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)].reshape(
            [self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1]
        )
        return relative_position_bias.transpose(perm=[2, 0, 1])


class EVAVisionTransformer(paddle.nn.Layer):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=paddle.nn.LayerNorm,
        init_values=None,
        patch_dropout=0.0,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        rope=False,
        use_mean_pooling=True,
        init_scale=0.001,
        grad_checkpointing=False,
        xattn=False,
        postnorm=False,
        pt_hw_seq_len=16,
        intp_freq=False,
        naiveswiglu=False,
        subln=False,
    ):
        super().__init__()
        self.image_size = img_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        out_7 = paddle.create_parameter(
            shape=paddle.zeros(shape=[1, 1, embed_dim]).shape,
            dtype=paddle.zeros(shape=[1, 1, embed_dim]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=[1, 1, embed_dim])),
        )
        out_7.stop_gradient = not True
        self.cls_token = out_7
        if use_abs_pos_emb:
            out_8 = paddle.create_parameter(
                shape=paddle.zeros(shape=[1, num_patches + 1, embed_dim]).shape,
                dtype=paddle.zeros(shape=[1, num_patches + 1, embed_dim]).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=[1, num_patches + 1, embed_dim])),
            )
            out_8.stop_gradient = not True
            self.pos_embed = out_8
        else:
            self.pos_embed = None
        self.pos_drop = paddle.nn.Dropout(p=drop_rate)
        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim, pt_seq_len=pt_hw_seq_len, ft_seq_len=hw_seq_len if intp_freq else None
            )
        else:
            self.rope = None
        self.naiveswiglu = naiveswiglu
        dpr = [x.item() for x in paddle.linspace(start=0, stop=drop_path_rate, num=depth)]
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = paddle.nn.LayerList(
            sublayers=[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                    xattn=xattn,
                    rope=self.rope,
                    postnorm=postnorm,
                    subln=subln,
                    naiveswiglu=naiveswiglu,
                )
                for i in range(depth)
            ]
        )
        self.norm = paddle.nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = (
            paddle.nn.Linear(in_features=embed_dim, out_features=num_classes)
            if num_classes > 0
            else paddle.nn.Identity()
        )
        if self.pos_embed is not None:
            trunc_normal_ = paddle.nn.initializer.TruncatedNormal(mean=0.0, std=0.02)
            trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        self.apply(self._init_weights)
        if isinstance(self.head, paddle.nn.Linear):
            trunc_normal_ = paddle.nn.initializer.TruncatedNormal(mean=0.0, std=0.02)
            trunc_normal_(self.head.weight)
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0.0 else paddle.nn.Identity()
        self.grad_checkpointing = grad_checkpointing

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.divide_(y=paddle.to_tensor(math.sqrt(2.0 * layer_id)))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_cast_dtype(self) -> paddle.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            trunc_normal_ = paddle.nn.initializer.TruncatedNormal(mean=0.0, std=0.02)
            trunc_normal_(m.weight)
            if m.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)
        elif isinstance(m, paddle.nn.LayerNorm):
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(m.bias)
            init_Constant = paddle.nn.initializer.Constant(value=1.0)
            init_Constant(m.weight)

    def get_num_layers(self):
        return len(self.blocks)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, "partial locking not currently supported for this model"
        for param in self.parameters():
            param.stop_gradient = not False

    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            paddle.nn.Linear(in_features=self.embed_dim, out_features=num_classes)
            if num_classes > 0
            else paddle.nn.Identity()
        )

    def forward_features(self, x, return_all_features=False):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.shape
        cls_tokens = self.cls_token.expand(shape=[batch_size, -1, -1])
        x = paddle.concat(x=(cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        if os.getenv("RoPE") == "1":
            if self.training and not isinstance(self.patch_dropout, paddle.nn.Identity):
                x, patch_indices_keep = self.patch_dropout(x)
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=patch_indices_keep)
            else:
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
                x = self.patch_dropout(x)
        else:
            x = self.patch_dropout(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            if i == len(self.blocks) - 1:
                continue
            else:
                x = blk(x, rel_pos_bias=rel_pos_bias)
        if not return_all_features:
            x = self.norm(x)
            if self.fc_norm is not None:
                return self.fc_norm(x.mean(axis=1))
            else:
                return x[:, 0]
        return x

    def forward(self, x, return_all_features=False):
        if return_all_features:
            return self.forward_features(x, return_all_features)
        x = self.forward_features(x)
        x = self.head(x)
        return x


class LayerNorm(paddle.nn.LayerNorm):
    def forward(self, x: paddle.Tensor):
        orig_type = x.dtype
        x = paddle.nn.functional.layer_norm(
            x=x, normalized_shape=self._normalized_shape, weight=self.weight, bias=self.bias, epsilon=self._epsilon
        )
        return x.to(orig_type)


FusedLayerNorm = LayerNorm


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None
    patch_dropout: float = 0.0
    global_average_pool: bool = False
    drop_path_rate: Optional[float] = None
    timm_model_name: str = None
    timm_model_pretrained: bool = False
    timm_pool: str = "avg"
    timm_proj: str = "linear"
    timm_proj_bias: bool = False
    eva_model_name: str = None
    qkv_bias: bool = True
    fusedLN: bool = False
    xattn: bool = False
    postnorm: bool = False
    rope: bool = False
    pt_hw_seq_len: int = 16
    intp_freq: bool = False
    naiveswiglu: bool = False
    subln: bool = False


def _build_vision_tower(embed_dim: int, vision_cfg: CLIPVisionCfg):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)
    if vision_cfg.eva_model_name:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNorm
        visual = EVAVisionTransformer(
            img_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            num_classes=embed_dim,
            use_mean_pooling=vision_cfg.global_average_pool,
            init_values=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            embed_dim=vision_cfg.width,
            depth=vision_cfg.layers,
            num_heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            qkv_bias=vision_cfg.qkv_bias,
            drop_path_rate=vision_cfg.drop_path_rate,
            norm_layer=partial(FusedLayerNorm, epsilon=1e-06)
            if vision_cfg.fusedLN
            else partial(norm_layer, epsilon=1e-06),
            xattn=vision_cfg.xattn,
            rope=vision_cfg.rope,
            postnorm=vision_cfg.postnorm,
            pt_hw_seq_len=vision_cfg.pt_hw_seq_len,
            intp_freq=vision_cfg.intp_freq,
            naiveswiglu=vision_cfg.naiveswiglu,
            subln=vision_cfg.subln,
        )
    return visual


class Eva2LargeEncoder(paddle.nn.Layer):
    def __init__(self, image_size=224):
        super(Eva2LargeEncoder, self).__init__()
        self.config = {
            "embed_dim": 768,
            "vision_cfg": {
                "image_size": 336,
                "layers": 24,
                "width": 1024,
                "drop_path_rate": 0,
                "head_width": 64,
                "mlp_ratio": 2.6667,
                "patch_size": 14,
                "eva_model_name": "eva-clip-l-14-336",
                "xattn": True,
                "fusedLN": True,
                "rope": True,
                "pt_hw_seq_len": 16,
                "intp_freq": True,
                "naiveswiglu": True,
                "subln": True,
            },
        }
        self.config["vision_cfg"]["image_size"] = image_size
        os.environ["delRoPE"] = "1"
        self.model = _build_vision_tower(**self.config)

    def forward(self, images):
        encode = self.model(images, return_all_features=True)[:, 1:, :]
        return encode


class CrossVisionModel(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.vit = Eva2LargeEncoder(image_size=config.cross_image_size)
        self.pos_embed = paddle.create_parameter(
            shape=[
                (self.vit.config["vision_cfg"]["image_size"] // self.vit.config["vision_cfg"]["patch_size"]) ** 2,
                self.vit.config["vision_cfg"]["width"],
            ],
            dtype=config.dtype,
        )
        self.pos_embed.stop_gradient = False

    def forward(self, images):
        enc = self.vit(images)
        return enc + self.pos_embed.unsqueeze(axis=0)


class PatchEmbedding(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.proj = paddle.nn.Conv2D(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        out_11 = paddle.create_parameter(
            shape=paddle.zeros(shape=[1, config.hidden_size]).shape,
            dtype=paddle.zeros(shape=[1, config.hidden_size]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=[1, config.hidden_size])),
        )
        out_11.stop_gradient = not True
        self.cls_embedding = out_11
        self.position_embedding = paddle.nn.Embedding(
            num_embeddings=config.num_positions, embedding_dim=config.hidden_size
        )

    def forward(self, images):
        x = self.proj(images)
        x = x.flatten(start_axis=2)
        perm_6 = list(range(x.ndim))
        perm_6[1] = 2
        perm_6[2] = 1
        x = x.transpose(perm=perm_6)
        cls_token = self.cls_embedding.expand(shape=[x.shape[0], -1, -1])
        x = paddle.concat(x=(cls_token, x), axis=1)
        x += self.position_embedding.weight.unsqueeze(axis=0)
        return x


class TransformerAttention(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        head_dim = config.hidden_size // config.num_heads
        self.scale = head_dim**-0.5
        self.query_key_value = paddle.nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size * 3)
        self.dense = paddle.nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.output_dropout = paddle.nn.Dropout(p=config.dropout_prob)

    def forward(self, x):
        B, L, _ = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.reshape([B, L, 3, self.num_heads, -1]).transpose(perm=[2, 0, 1, 3, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = memory_efficient_attention(q, k, v, scale=self.scale)
        out = out.reshape([B, L, -1])
        output = self.dense(out)
        output = self.output_dropout(output)
        return output

    def attention(self, q, k, v):
        x = k
        perm_7 = list(range(x.ndim))
        perm_7[-2] = -1
        perm_7[-1] = -2
        attn_weights = paddle.matmul(x=q * self.scale, y=x.transpose(perm=perm_7))
        attn_weights = paddle.nn.functional.softmax(attn_weights, axis=-1)
        output = paddle.matmul(x=attn_weights, y=v)
        return output


class TransformerMLP(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = paddle.nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size)
        self.fc2 = paddle.nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class TransformerLayer(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = paddle.nn.LayerNorm(normalized_shape=config.hidden_size, epsilon=config.layer_norm_eps)
        self.attention = TransformerAttention(config)
        self.mlp = TransformerMLP(config)
        self.post_attention_layernorm = paddle.nn.LayerNorm(
            normalized_shape=config.hidden_size, epsilon=config.layer_norm_eps
        )

    def forward(self, hidden_states):
        attention_input = hidden_states
        attention_output = self.input_layernorm(self.attention(attention_input))
        hidden_states = attention_input + attention_output
        mlp_input = hidden_states
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))
        output = mlp_input + mlp_output
        return output


class Transformer(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.layers = paddle.nn.LayerList(
            sublayers=[TransformerLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states):
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class GLU(paddle.nn.Layer):
    def __init__(self, config, in_features):
        super().__init__()
        self.linear_proj = paddle.nn.Linear(in_features=in_features, out_features=config.hidden_size, bias_attr=False)
        self.norm1 = paddle.nn.LayerNorm(normalized_shape=config.hidden_size)
        self.act1 = paddle.nn.GELU()
        self.act2 = paddle.nn.functional.silu
        self.dense_h_to_4h = paddle.nn.Linear(
            in_features=config.hidden_size, out_features=config.intermediate_size, bias_attr=False
        )
        self.gate_proj = paddle.nn.Linear(
            in_features=config.hidden_size, out_features=config.intermediate_size, bias_attr=False
        )
        self.dense_4h_to_h = paddle.nn.Linear(
            in_features=config.intermediate_size, out_features=config.hidden_size, bias_attr=False
        )

    def forward(self, x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)
        return x


class EVA2CLIPModel(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.model_type = config.model_type
        vision_config = Namespace(**config.vision_config)
        self.patch_embedding = PatchEmbedding(vision_config)
        self.transformer = Transformer(vision_config)
        self.linear_proj = GLU(config, in_features=vision_config.hidden_size)
        out_12 = paddle.create_parameter(
            shape=paddle.zeros(shape=[1, 1, config.hidden_size]).shape,
            dtype=paddle.zeros(shape=[1, 1, config.hidden_size]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=[1, 1, config.hidden_size])),
        )
        out_12.stop_gradient = not True
        self.boi = out_12
        out_13 = paddle.create_parameter(
            shape=paddle.zeros(shape=[1, 1, config.hidden_size]).shape,
            dtype=paddle.zeros(shape=[1, 1, config.hidden_size]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=[1, 1, config.hidden_size])),
        )
        out_13.stop_gradient = not True
        self.eoi = out_13
        if self.model_type == "cogagent":
            out_14 = paddle.create_parameter(
                shape=paddle.zeros(
                    shape=[(vision_config.image_size // vision_config.patch_size) ** 2, vision_config.hidden_size]
                ).shape,
                dtype=paddle.zeros(
                    shape=[(vision_config.image_size // vision_config.patch_size) ** 2, vision_config.hidden_size]
                )
                .numpy()
                .dtype,
                default_initializer=paddle.nn.initializer.Assign(
                    paddle.zeros(
                        shape=[(vision_config.image_size // vision_config.patch_size) ** 2, vision_config.hidden_size]
                    )
                ),
            )
            out_14.stop_gradient = not True
            self.pos_embed = out_14
        elif self.model_type != "cogvlm":
            raise ValueError("model_type in config must be cogagent or cogvlm, but got {}".format(self.model_type))

    def forward(self, images):
        x = self.patch_embedding(images)
        x = self.transformer(x)
        x = x[:, 1:]
        if self.model_type == "cogagent":
            x = self.linear_proj(x + self.pos_embed.unsqueeze(axis=0))
        elif self.model_type == "cogvlm":
            x = self.linear_proj(x)
        else:
            raise ValueError("model_type in config must be cogagent or cogvlm, but got {}".format(self.model_type))
        boi = self.boi.expand(shape=[x.shape[0], -1, -1])
        eoi = self.eoi.expand(shape=[x.shape[0], -1, -1])
        x = paddle.concat(x=(boi, x, eoi), axis=1)
        return x
