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
from typing import Dict, List

import numpy as np
import paddle
import requests
from paddle.vision.transforms import functional as F
from paddlenlp.utils.initializer import normal_
from PIL import Image

from ..groundingdino.layers import MultiHeadAttention
from .qwen_vit import VisionTransformer, VisionTransformerConfig, get_abs_pos


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


class Vision(paddle.nn.Layer):
    def __init__(
        self,
        config: Dict[str, int],
        n_queries: int = 256,
    ):
        super().__init__()
        image_height, image_width = self.image_size = config["image_size"], config["image_size"]
        patch_height, patch_width = self.patch_size = config["patch_size"], config["patch_size"]
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.output_dim = config["output_dim"]

        vit_config = VisionTransformerConfig(**config)
        self.vit = VisionTransformer(vit_config)
        norm_layer = partial(paddle.nn.LayerNorm, epsilon=1e-06)

        self.attn_pool = Resampler(
            grid_size=int(math.sqrt(n_queries)),
            embed_dim=self.output_dim,
            num_heads=self.output_dim // 128,
            kv_dim=config["width"],
            norm_layer=norm_layer,
        )
        self.ln_post = norm_layer(self.output_dim)
        self.proj = paddle.create_parameter(
            shape=[self.output_dim, self.output_dim],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Assign(
                self.output_dim**-0.5 * paddle.randn(shape=[self.output_dim, self.output_dim])
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
        x = self.vit(x)

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
