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
from functools import partial
from typing import List, Tuple, Union

import paddle
from iopath.common.file_io import g_pathmgr
from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)
from sam2.modeling.sam2_utils import MLP, DropPath


def do_pool(x: paddle.Tensor, pool: paddle.nn.Layer, norm: paddle.nn.Layer = None) -> paddle.Tensor:
    if pool is None:
        return x
    x = x.transpose(perm=[0, 3, 1, 2])
    x = pool(x)
    x = x.transpose(perm=[0, 2, 3, 1])
    if norm:
        x = norm(x)
    return x


class MultiScaleAttention(paddle.nn.Layer):
    def __init__(self, dim: int, dim_out: int, num_heads: int, q_pool: paddle.nn.Layer = None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = paddle.nn.Linear(in_features=dim, out_features=dim_out * 3)
        self.proj = paddle.nn.Linear(in_features=dim_out, out_features=dim_out)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        B, H, W, _ = tuple(x.shape)

        qkv = self.qkv(x).reshape([B, H * W, 3, self.num_heads, -1])
        q, k, v = paddle.unbind(input=qkv, axis=2)
        if self.q_pool:
            q = do_pool(q.reshape([B, H, W, -1]), self.q_pool)
            H, W = tuple(q.shape)[1:3]
            q = q.reshape([B, H * W, self.num_heads, -1])

        x = paddle.nn.functional.scaled_dot_product_attention_(q, k, v)

        x = x.reshape([B, H, W, -1])
        x = self.proj(x)
        return x


class MultiScaleBlock(paddle.nn.Layer):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[paddle.nn.Layer, str] = "LayerNorm",
        q_stride: Tuple[int, int] = None,
        act_layer: paddle.nn.Layer = paddle.nn.GELU,
        window_size: int = 0,
    ):
        super().__init__()
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(paddle.nn, norm_layer), epsilon=1e-06)
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.window_size = window_size
        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = paddle.nn.MaxPool2D(kernel_size=q_stride, stride=q_stride, ceil_mode=False)
        self.attn = MultiScaleAttention(dim, dim_out, num_heads=num_heads, q_pool=self.pool)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()
        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, num_layers=2, activation=act_layer)
        if dim != dim_out:
            self.proj = paddle.nn.Linear(in_features=dim, out_features=dim_out)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)
        window_size = self.window_size
        if window_size > 0:
            H, W = tuple(x.shape)[1], tuple(x.shape)[2]
            x, pad_hw = window_partition(x, window_size)
        x = self.attn(x)
        if self.q_stride:
            window_size = self.window_size // self.q_stride[0]
            H, W = tuple(shortcut.shape)[1:3]
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = H + pad_h, W + pad_w
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Hiera(paddle.nn.Layer):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
        self,
        embed_dim: int = 96,
        num_heads: int = 1,
        drop_path_rate: float = 0.0,
        q_pool: int = 3,
        q_stride: Tuple[int, int] = (2, 2),
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        window_spec: Tuple[int, ...] = (8, 4, 14, 7),
        global_att_blocks: Tuple[int, ...] = (12, 16, 20),
        weights_path=None,
        return_interm_layers=True,
    ):
        super().__init__()
        assert len(stages) == len(window_spec)
        self.window_spec = window_spec
        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [(sum(stages[:i]) - 1) for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [(x + 1) for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.global_att_blocks = global_att_blocks
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        out_16 = paddle.create_parameter(
            shape=paddle.zeros(shape=[1, embed_dim, *self.window_pos_embed_bkg_spatial_size]).shape,
            dtype=paddle.zeros(shape=[1, embed_dim, *self.window_pos_embed_bkg_spatial_size]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.zeros(shape=[1, embed_dim, *self.window_pos_embed_bkg_spatial_size])
            ),
        )
        out_16.stop_gradient = not True
        self.pos_embed = out_16
        out_17 = paddle.create_parameter(
            shape=paddle.zeros(shape=[1, embed_dim, self.window_spec[0], self.window_spec[0]]).shape,
            dtype=paddle.zeros(shape=[1, embed_dim, self.window_spec[0], self.window_spec[0]]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.zeros(shape=[1, embed_dim, self.window_spec[0], self.window_spec[0]])
            ),
        )
        out_17.stop_gradient = not True
        self.pos_embed_window = out_17
        dpr = [x.item() for x in paddle.linspace(start=0, stop=drop_path_rate, num=depth)]
        cur_stage = 1
        self.blocks = paddle.nn.LayerList()
        for i in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]
            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )
            embed_dim = dim_out
            self.blocks.append(block)
        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )
        if weights_path is not None:
            with g_pathmgr.open(weights_path, "rb") as f:
                chkpt = paddle.load(path=f)
            logging.info("loading Hiera", self.set_state_dict(state_dict=chkpt, use_structured_name=False))

    def _get_pos_embed(self, hw: Tuple[int, int]) -> paddle.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = paddle.nn.functional.interpolate(x=self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            repeat_times=[(x // y) for x, y in zip(tuple(pos_embed.shape), tuple(window_embed.shape))]
        )
        pos_embed = pos_embed.transpose(perm=[0, 2, 3, 1])
        return pos_embed

    def forward(self, x: paddle.Tensor) -> List[paddle.Tensor]:
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(tuple(x.shape)[1:3])
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.stage_ends[-1] or i in self.stage_ends and self.return_interm_layers:
                feats = x.transpose(perm=[0, 3, 1, 2])
                outputs.append(feats)
        return outputs

    def get_layer_id(self, layer_name):
        num_layers = self.get_num_layers()
        if layer_name.find("rel_pos") != -1:
            return num_layers + 1
        elif layer_name.find("pos_embed") != -1:
            return 0
        elif layer_name.find("patch_embed") != -1:
            return 0
        elif layer_name.find("blocks") != -1:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        else:
            return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)
