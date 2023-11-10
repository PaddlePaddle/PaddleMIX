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
from functools import partial
from typing import Callable, List, Optional

import numpy as np
import paddle
import requests
from paddle.vision.transforms import functional as F
from paddlenlp.utils.initializer import normal_
from PIL import Image

from ..groundingdino.layers import MultiHeadAttention


def get_abs_pos(abs_pos, tgt_size):
    src_size = int(math.sqrt(abs_pos.shape[0]))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype
    if src_size != tgt_size:
        return (
            paddle.nn.functional.interpolate(
                x=abs_pos.astype(dtype="float32").reshape([1, src_size, src_size, -1]).transpose(perm=[0, 3, 1, 2]),
                size=(tgt_size, tgt_size),
                mode="bicubic",
                align_corners=False,
            )
            .transpose(perm=[0, 2, 3, 1])
            .flatten(start_axis=0, stop_axis=2)
            .astype(dtype=dtype)
        )
    else:
        return abs_pos


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class Resampler(paddle.nn.Layer):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(self, grid_size, embed_dim, num_heads, kv_dim=None, norm_layer=paddle.nn.LayerNorm):
        super().__init__()
        self.num_queries = grid_size**2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        x = paddle.to_tensor(get_2d_sincos_pos_embed(embed_dim, grid_size), dtype=paddle.get_default_dtype())
        self.pos_embed = paddle.create_parameter(
            shape=x.shape, dtype=x.dtype, default_initializer=paddle.nn.initializer.Assign(x)
        )

        x = paddle.zeros(shape=[self.num_queries, embed_dim], dtype=paddle.get_default_dtype())
        self.query = paddle.create_parameter(
            shape=x.shape, dtype=x.dtype, default_initializer=paddle.nn.initializer.Assign(x)
        )
        self.query.stop_gradient = not True

        normal_(self.query, mean=0.0, std=0.02)
        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = paddle.nn.Linear(in_features=kv_dim, out_features=embed_dim, bias_attr=False)
        else:
            self.kv_proj = paddle.nn.Identity()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            normal_(m.weight, std=0.02)
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)
        elif isinstance(m, paddle.nn.LayerNorm):
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(m.bias)
            init_Constant = paddle.nn.initializer.Constant(value=1.0)
            init_Constant(m.weight)

    def forward(self, x, attn_mask=None):
        pos_embed = get_abs_pos(self.pos_embed, x.shape[1])
        x = self.kv_proj(x)

        x = self.ln_kv(x).transpose(perm=[1, 0, 2])
        N = x.shape[1]

        q = self.ln_q(self.query)

        query = (self._repeat(q, N) + self.pos_embed.unsqueeze(axis=1)).transpose(perm=[1, 0, 2])
        key = (x + pos_embed.unsqueeze(axis=1)).transpose(perm=[1, 0, 2])
        value = x.transpose(perm=[1, 0, 2])

        out = self.attn(query, key, value, attn_mask=attn_mask)

        return out

    def _repeat(self, query, N: int):
        return paddle.tile(query.unsqueeze(axis=1), [1, N, 1])


class VisualAttention(paddle.nn.Layer):
    """self-attention layer class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, embed_dim, num_heads, bias=True, kdim=None, vdim=None):
        super(VisualAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0
        self.hidden_size_per_attention_head = embed_dim // num_heads
        self.num_attention_heads_per_partition = num_heads
        self.hidden_size_per_partition = embed_dim

        assert self._qkv_same_embed_dim, "Only Support SelfAttention Currently"
        self.in_proj = paddle.nn.Linear(in_features=embed_dim, out_features=3 * embed_dim)
        self.out_proj = paddle.nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

    def forward(self, query, key, value, attn_mask=None):

        sq, b, _ = query.shape
        assert query is key, "Only Support Self-Attention Currently"
        sk = sq
        mixed_x_layer = self.in_proj(query)
        new_tensor_shape = mixed_x_layer.shape[:-1] + [
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        ]
        mixed_x_layer = mixed_x_layer.reshape(new_tensor_shape)

        query_layer, key_layer, value_layer = mixed_x_layer.split(
            new_tensor_shape[-1] // self.hidden_size_per_attention_head, axis=-1
        )

        x = query_layer.reshape([sq, b * self.num_attention_heads_per_partition, self.hidden_size_per_attention_head])
        perm_3 = list(range(x.ndim))
        perm_3[0] = 1
        perm_3[1] = 0
        query_layer = x.transpose(perm=perm_3)
        x = key_layer.reshape([sk, b * self.num_attention_heads_per_partition, self.hidden_size_per_attention_head])
        perm_4 = list(range(x.ndim))
        perm_4[0] = 1
        perm_4[1] = 0
        key_layer = x.transpose(perm=perm_4)
        q_scaled = query_layer / self.norm_factor
        if attn_mask is not None:
            x = key_layer
            perm_5 = list(range(x.ndim))
            perm_5[-2] = -1
            perm_5[-1] = -2
            attention_probs = paddle.add(1 * attn_mask, 1 * paddle.bmm(q_scaled, x.transpose(perm=perm_5)))
        else:
            x = key_layer
            perm_6 = list(range(x.ndim))
            perm_6[-2] = -1
            perm_6[-1] = -2
            attention_probs = paddle.bmm(x=q_scaled, y=x.transpose(perm=perm_6))
        attention_probs = paddle.nn.functional.softmax(attention_probs, axis=-1)

        x = value_layer.reshape([sk, b * self.num_attention_heads_per_partition, self.hidden_size_per_attention_head])
        perm_7 = list(range(x.ndim))
        perm_7[0] = 1
        perm_7[1] = 0
        value_layer = x.transpose(perm=perm_7)
        context_layer = paddle.bmm(x=attention_probs, y=value_layer)

        context_layer = context_layer.reshape(
            [b, self.num_attention_heads_per_partition, sq, self.hidden_size_per_attention_head]
        )
        context_layer = context_layer.transpose(perm=[2, 0, 1, 3])
        new_context_layer_shape = context_layer.shape[:-2] + [self.hidden_size_per_partition]
        context_layer = context_layer.reshape(new_context_layer_shape)
        output = self.out_proj(context_layer)
        return output


class VisualAttentionBlock(paddle.nn.Layer):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = paddle.nn.GELU,
        norm_layer: Callable = paddle.nn.LayerNorm,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        self.ln_1 = norm_layer(d_model)
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.attn = VisualAttention(d_model, n_head)
        self.mlp = paddle.nn.Sequential(
            *[
                ("c_fc", paddle.nn.Linear(in_features=d_model, out_features=mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", paddle.nn.Linear(in_features=mlp_width, out_features=d_model)),
            ]
        )

    def attention(
        self,
        q_x: paddle.Tensor,
        k_x: Optional[paddle.Tensor] = None,
        v_x: Optional[paddle.Tensor] = None,
        attn_mask: Optional[paddle.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.astype(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, attn_mask=attn_mask)

    def forward(
        self,
        q_x: paddle.Tensor,
        k_x: Optional[paddle.Tensor] = None,
        v_x: Optional[paddle.Tensor] = None,
        attn_mask: Optional[paddle.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerBlock(paddle.nn.Layer):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = paddle.nn.GELU,
        norm_layer: Callable = paddle.nn.LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = paddle.nn.LayerList(
            sublayers=[
                VisualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer, norm_layer=norm_layer)
                for _ in range(layers)
            ]
        )

    def get_cast_dtype(self) -> paddle.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def get_cast_device(self) -> str:
        return self.resblocks[0].mlp.c_fc.weight.device

    def forward(self, x: paddle.Tensor, attn_mask: Optional[paddle.Tensor] = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)

        return x


class VisionTransformer(paddle.nn.Layer):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        n_queries: int = 256,
        output_dim: int = 512,
        **kwargs
    ):
        super().__init__()
        image_height, image_width = self.image_size = image_size, image_size
        patch_height, patch_width = self.patch_size = patch_size, patch_size
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.output_dim = output_dim

        self.conv1 = paddle.nn.Conv2D(
            in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias_attr=False
        )
        scale = width**-0.5
        out_4 = paddle.create_parameter(
            shape=(scale * paddle.randn(shape=[256, width])).shape,
            dtype=(scale * paddle.randn(shape=[256, width])).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(scale * paddle.randn(shape=[256, width])),
        )
        out_4.stop_gradient = not True
        self.positional_embedding = out_4
        norm_layer = partial(paddle.nn.LayerNorm, epsilon=1e-06)
        act_layer = paddle.nn.GELU
        self.ln_pre = norm_layer(width)
        self.transformer = TransformerBlock(
            width, layers, heads, mlp_ratio, act_layer=act_layer, norm_layer=norm_layer
        )
        self.attn_pool = Resampler(
            grid_size=int(math.sqrt(n_queries)),
            embed_dim=output_dim,
            num_heads=output_dim // 128,
            kv_dim=width,
            norm_layer=norm_layer,
        )
        self.ln_post = norm_layer(output_dim)
        self.proj = paddle.create_parameter(
            shape=[output_dim, output_dim],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Assign(
                output_dim**-0.5 * paddle.randn(shape=[output_dim, output_dim])
            ),
        )
        self.proj.stop_gradient = not True

    def image_transform(self, image):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

        image = F.resize(image, size=self.image_size, interpolation="bicubic")
        tensor_normalize = paddle.vision.transforms.Normalize(mean=mean, std=std, data_format="HWC")
        image = tensor_normalize(np.array(image) / 255.0)
        image = F.to_tensor(image)

        return image

    def forward(self, x: paddle.Tensor):
        x = x.astype(dtype=self.conv1.weight.dtype)

        x = self.conv1(x)

        x = x.reshape([x.shape[0], x.shape[1], self.grid_size[0] * self.grid_size[1]])
        x = x.transpose(perm=[0, 2, 1])
        x = x + get_abs_pos(self.positional_embedding, x.shape[1])

        x = self.ln_pre(x)
        x = x.transpose(perm=[1, 0, 2])

        x = self.transformer(x)
        x = x.transpose(perm=[1, 0, 2])

        x = self.attn_pool(x)

        x = self.ln_post(x)
        x = x @ self.proj

        return x

    def prepare(self, image_paths: List[str]):
        images = []
        for image_path in image_paths:
            if image_path.startswith("http://") or image_path.startswith("https://"):
                image = Image.open(requests.get(image_path, stream=True).raw)
            else:
                image = Image.open(image_path)
            image = image.convert("RGB")
            images.append(self.image_transform(image))
        images = paddle.stack(x=images, axis=0)
        return images
