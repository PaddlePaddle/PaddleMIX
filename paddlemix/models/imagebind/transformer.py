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

from functools import partial
from typing import Callable, List, Optional

import paddle
from paddle.nn.layer.transformer import tensor


class Attention(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = paddle.nn.Linear(in_features=dim, out_features=dim * 3, bias_attr=qkv_bias)
        self.attn_drop = paddle.nn.Dropout(p=attn_drop)
        self.proj = paddle.nn.Linear(in_features=dim, out_features=dim)
        self.proj_drop = paddle.nn.Dropout(p=proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = k
        perm_2 = list(range(x.ndim))
        perm_2[-2] = -1
        perm_2[-1] = -2
        attn = q @ x.transpose(perm=perm_2) * self.scale
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        perm_3 = list(range(x.ndim))
        perm_3[1] = 2
        perm_3[2] = 1
        x = x.transpose(perm=perm_3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(paddle.nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=paddle.nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = paddle.nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = act_layer()
        self.fc2 = paddle.nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = paddle.nn.Dropout(p=drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiheadAttention(paddle.nn.MultiHeadAttention):
    def __init__(self, embed_dim, num_heads, *arg, add_bias_kv=None, **kwargs):
        super(MultiheadAttention, self).__init__(embed_dim, num_heads, *arg, **kwargs)
        self.add_bias_kv = add_bias_kv
        self.embed_dim = embed_dim
        if self.add_bias_kv:
            self.bias_k = paddle.create_parameter(
                shape=[1, 1, embed_dim],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(value=0.0),
            )
            self.bias_v = paddle.create_parameter(
                shape=[1, 1, embed_dim],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(value=0.0),
            )

    def compute_kv(self, key, value):
        k = self.k_proj(key)
        v = self.v_proj(value)
        bsz, _, _ = k.shape
        if self.add_bias_kv:
            k = paddle.concat([k, paddle.repeat_interleave(self.bias_k, bsz, axis=0)], axis=1)
            v = paddle.concat([v, paddle.repeat_interleave(self.bias_v, bsz, axis=0)], axis=1)
        k = tensor.reshape(x=k, shape=[0, 0, self.num_heads, self.head_dim])
        k = tensor.transpose(x=k, perm=[0, 2, 1, 3])
        v = tensor.reshape(x=v, shape=[0, 0, self.num_heads, self.head_dim])
        v = tensor.transpose(x=v, perm=[0, 2, 1, 3])
        return k, v

    def forward(self, x: paddle.Tensor, attn_mask: paddle.Tensor):
        # x = paddle.transpose(x, perm=[1,0, 2])
        return super(MultiheadAttention, self).forward(x, x, x, attn_mask=attn_mask)


class ViTAttention(Attention):
    def forward(self, x: paddle.Tensor, attn_mask: paddle.Tensor):
        assert attn_mask is None
        return super(ViTAttention, self).forward(x)


class Identity(paddle.nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class BlockWithMasking(paddle.nn.Layer):
    def __init__(
        self,
        dim: int,
        attn_target: Callable,
        mlp_ratio: int = 4,
        act_layer: Callable = paddle.nn.GELU,
        norm_layer: Callable = paddle.nn.LayerNorm,
        ffn_dropout_rate: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_type: Optional[str] = None,
        layer_scale_init_value: float = 0.0001,
    ):
        super().__init__()
        assert not isinstance(
            attn_target, paddle.nn.Layer
        ), "attn_target should be a Callable. Otherwise attn_target is shared across blocks!"
        self.attn = attn_target()
        if drop_path > 0.0:
            self.drop_path = paddle.nn.Dropout(drop_path)
        else:
            self.drop_path = Identity()
        self.norm_1 = norm_layer(dim)
        mlp_hidden_dim = int(mlp_ratio * dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=ffn_dropout_rate,
        )
        self.norm_2 = norm_layer(dim)
        self.layer_scale_type = layer_scale_type
        if self.layer_scale_type is not None:
            assert self.layer_scale_type in [
                "per_channel",
                "scalar",
            ], f"Found Layer scale type {self.layer_scale_type}"
            if self.layer_scale_type == "per_channel":
                gamma_shape = [1, 1, dim]
            elif self.layer_scale_type == "scalar":
                gamma_shape = [1, 1, 1]

            self.layer_scale_gamma1 = paddle.create_parameter(
                shape=gamma_shape,
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(value=1.0),
            )
            self.layer_scale_gamma2 = paddle.create_parameter(
                shape=gamma_shape,
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(value=1.0),
            )

    def forward(self, x: paddle.Tensor, attn_mask: paddle.Tensor):
        if self.layer_scale_type is None:
            x = x + self.drop_path(self.attn(self.norm_1(x), attn_mask))
            x = x + self.drop_path(self.mlp(self.norm_2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm_1(x), attn_mask)) * self.layer_scale_gamma1
            x = x + self.drop_path(self.mlp(self.norm_2(x))) * self.layer_scale_gamma2
        return x


_LAYER_NORM = partial(paddle.nn.LayerNorm, epsilon=1e-06)


class SimpleTransformer(paddle.nn.Layer):
    def __init__(
        self,
        attn_target: Callable,
        embed_dim: int,
        num_blocks: int,
        block: Callable = BlockWithMasking,
        pre_transformer_layer: Optional[Callable] = None,
        post_transformer_layer: Optional[Callable] = None,
        drop_path_rate: float = 0.0,
        drop_path_type: str = "progressive",
        norm_layer: Callable = _LAYER_NORM,
        mlp_ratio: int = 4,
        ffn_dropout_rate: float = 0.0,
        layer_scale_type: Optional[str] = None,
        layer_scale_init_value: float = 0.0001,
        weight_init_style: str = "jax",
    ):
        """
        Simple Transformer with the following features
        1. Supports masked attention
        2. Supports DropPath
        3. Supports LayerScale
        4. Supports Dropout in Attention and FFN
        5. Makes few assumptions about the input except that it is a Tensor
        """
        super().__init__()
        self.pre_transformer_layer = pre_transformer_layer
        if drop_path_type == "progressive":
            dpr = [x.item() for x in paddle.linspace(start=0, stop=drop_path_rate, num=num_blocks)]
        elif drop_path_type == "uniform":
            dpr = [drop_path_rate for i in range(num_blocks)]
        else:
            raise ValueError(f"Unknown drop_path_type: {drop_path_type}")
        self.blocks = paddle.nn.Sequential(
            *[
                block(
                    dim=embed_dim,
                    attn_target=attn_target,
                    mlp_ratio=mlp_ratio,
                    ffn_dropout_rate=ffn_dropout_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    layer_scale_type=layer_scale_type,
                    layer_scale_init_value=layer_scale_init_value,
                )
                for i in range(num_blocks)
            ]
        )
        self.post_transformer_layer = post_transformer_layer
        self.weight_init_style = weight_init_style
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            if self.weight_init_style == "jax":
                paddle.nn.initializer.XavierUniform()(m.weight)

            elif self.weight_init_style == "pytorch":
                paddle.nn.initializer.TruncatedNormal(std=0.02)(m.weight)

            if m.bias is not None:
                paddle.nn.initializer.Constant(value=0.0)(m.bias)

        elif isinstance(m, paddle.nn.LayerNorm):
            paddle.nn.initializer.Constant(value=0.0)(m.bias)
            paddle.nn.initializer.Constant(value=1.0)(m.weight)

    def forward(
        self,
        tokens: paddle.Tensor,
        attn_mask: paddle.Tensor = None,
        use_checkpoint: bool = False,
        checkpoint_every_n: int = 1,
        checkpoint_blk_ids: Optional[List[int]] = None,
    ):
        """
        Inputs
        - tokens: data of shape N x L x D (or L x N x D depending on the attention implementation)
        - attn: mask of shape L x L

        Output
        - x: data of shape N x L x D (or L x N x D depending on the attention implementation)
        """
        if self.pre_transformer_layer:
            tokens = self.pre_transformer_layer(tokens)
        if use_checkpoint and checkpoint_blk_ids is None:
            checkpoint_blk_ids = [blk_id for blk_id in range(len(self.blocks)) if blk_id % checkpoint_every_n == 0]
        if checkpoint_blk_ids:
            checkpoint_blk_ids = set(checkpoint_blk_ids)
        for blk_id, blk in enumerate(self.blocks):
            if use_checkpoint and blk_id in checkpoint_blk_ids:
                pass
            else:

                tokens = blk(tokens, attn_mask=attn_mask)
        if self.post_transformer_layer:
            tokens = self.post_transformer_layer(tokens)
        return tokens
