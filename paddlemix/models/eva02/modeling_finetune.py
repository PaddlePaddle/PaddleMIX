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
import os
from functools import partial
from typing import Dict, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..clip.modules.rope import VisionRotaryEmbeddingFast
from ..clip.utils import to_2tuple, trunc_normal_

try:
    from ..clip.modules.fusedln import FusedLayerNorm
except:
    from paddle.nn import LayerNorm as FusedLayerNorm

    print("Warning, FusedLn module is not available, use LayerNorm instead.")

import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.incubate.nn.memory_efficient_attention import memory_efficient_attention
from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.utils.log import logger

from paddlemix.models.model_utils import MixPretrainedModel

__all__ = ["EVA02VisionTransformerPretrainedModel", "EVA02VisionTransformer"]


def drop_path(x, drop_prob=0.0, training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob, dtype=x.dtype)
    shape = (paddle.shape(x)[0],) + (1,) * (x.ndim - 1)
    if x.dtype == paddle.float16:
        random_tensor = keep_prob + paddle.rand(shape, dtype=paddle.float32).astype(x.dtype)
    else:
        random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(paddle.nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        with get_rng_state_tracker().rng_state("global_seed"):
            out = drop_path(x, self.drop_prob, self.training)
        return out

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class Mlp(paddle.nn.Layer):
    def __init__(self, config, act_layer=paddle.nn.GELU, norm_layer=paddle.nn.LayerNorm):
        super().__init__()
        in_features = config.embed_dim
        hidden_features = int(config.embed_dim * config.mlp_ratio)
        out_features = in_features
        hidden_features = hidden_features or in_features
        if dist.get_world_size() > 1:
            self.fc1 = fleet.meta_parallel.ColumnParallelLinear(
                in_features, hidden_features, weight_attr=None, has_bias=True, gather_output=True
            )
            self.fc2 = fleet.meta_parallel.ColumnParallelLinear(
                hidden_features, out_features, weight_attr=None, has_bias=True, gather_output=True
            )
        else:
            self.fc1 = paddle.nn.Linear(in_features, hidden_features)
            self.fc2 = paddle.nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if config.subln else paddle.nn.Identity()
        self.drop = paddle.nn.Dropout(p=config.drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.ffn_ln(x)
        x = self.fc2(x)
        with get_rng_state_tracker().rng_state("global_seed"):
            x = self.drop(x)
        return x


class SwiGLU(paddle.nn.Layer):
    def __init__(
        self,
        config,
        drop=0.0,
        act_layer=paddle.nn.Silu,
        norm_layer=paddle.nn.LayerNorm,
    ):
        super().__init__()
        in_features = config.embed_dim
        hidden_features = int(config.embed_dim * config.mlp_ratio)
        out_features = in_features
        if dist.get_world_size() > 1:
            self.w1 = fleet.meta_parallel.ColumnParallelLinear(
                in_features, hidden_features, weight_attr=None, has_bias=True, gather_output=True
            )
            self.w2 = fleet.meta_parallel.ColumnParallelLinear(
                in_features, hidden_features, weight_attr=None, has_bias=True, gather_output=True
            )
            self.w3 = fleet.meta_parallel.ColumnParallelLinear(
                hidden_features, out_features, weight_attr=None, has_bias=True, gather_output=True
            )
        else:
            self.w1 = paddle.nn.Linear(in_features, hidden_features)
            self.w2 = paddle.nn.Linear(in_features, hidden_features)
            self.w3 = paddle.nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        if config.subln and config.naiveswiglu:
            self.ffn_ln = norm_layer(hidden_features)
        else:
            # Note: tiny/s swiglu has no ffn_ln
            self.ffn_ln = paddle.nn.Identity()
        self.drop = paddle.nn.Dropout(p=drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        with get_rng_state_tracker().rng_state("global_seed"):
            x = self.drop(x)
        return x


class Attention(paddle.nn.Layer):
    def __init__(self, config, window_size=None, rope=None, norm_layer=paddle.nn.LayerNorm):
        super().__init__()
        dim = config.embed_dim
        self.xattn_drop = config.attn_drop_rate
        self.xattn = config.xattn
        self.deepnorm = config.deepnorm
        self.subln = config.subln

        self.num_heads = config.num_heads
        head_dim = dim // self.num_heads
        if hasattr(config, "attn_head_dim") and config.attn_head_dim is not None:
            head_dim = config.attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = config.qk_scale or head_dim**-0.5

        # NOTE: only support separate qkv for paddle tensor parallel distributed
        if dist.get_world_size() > 1:
            self.q_proj = fleet.meta_parallel.ColumnParallelLinear(
                dim, all_head_dim, weight_attr=None, has_bias=config.qkv_bias, gather_output=True
            )
            self.k_proj = fleet.meta_parallel.ColumnParallelLinear(
                dim, all_head_dim, weight_attr=None, has_bias=False, gather_output=True
            )
            self.v_proj = fleet.meta_parallel.ColumnParallelLinear(
                dim, all_head_dim, weight_attr=None, has_bias=config.qkv_bias, gather_output=True
            )
        else:
            self.q_proj = paddle.nn.Linear(dim, all_head_dim, bias_attr=config.qkv_bias)
            self.k_proj = paddle.nn.Linear(dim, all_head_dim, bias_attr=False)
            self.v_proj = paddle.nn.Linear(dim, all_head_dim, bias_attr=config.qkv_bias)

        self.rel_pos_bias = None
        self.qk_float = True

        self.window_size = None
        self.relative_position_bias_table = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            init_data = paddle.zeros(shape=[self.num_relative_distance, self.num_heads])
            self.relative_position_bias_table = self.create_parameter(
                shape=[self.num_relative_distance, self.num_heads],
                default_initializer=paddle.nn.initializer.Assign(init_data),
            )
            coords_h = paddle.arange(end=window_size[0])
            coords_w = paddle.arange(end=window_size[1])
            coords = paddle.stack(x=paddle.meshgrid([coords_h, coords_w]))
            coords_flatten = paddle.flatten(x=coords, start_axis=1)
            relative_coords = coords_flatten[:, :, (None)] - coords_flatten[:, (None), :]
            relative_coords = relative_coords.transpose(perm=[1, 2, 0])
            relative_coords[:, :, (0)] += window_size[0] - 1
            relative_coords[:, :, (1)] += window_size[1] - 1
            relative_coords[:, :, (0)] *= 2 * window_size[1] - 1
            relative_position_index = paddle.zeros(
                shape=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
            )
            relative_position_index[1:, 1:] = relative_coords.sum(axis=-1)
            relative_position_index[(0), 0:] = self.num_relative_distance - 3
            relative_position_index[0:, (0)] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1
            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None
        self.attn_drop = paddle.nn.Dropout(p=self.xattn_drop)

        if dist.get_world_size() > 1:
            self.proj = fleet.meta_parallel.ColumnParallelLinear(
                all_head_dim, dim, weight_attr=None, has_bias=True, gather_output=True
            )
        else:
            self.proj = paddle.nn.Linear(all_head_dim, dim)
        self.proj_drop = paddle.nn.Dropout(p=config.drop_rate)
        self.rope = rope

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape

        # NOTE: only support separate qkv for paddle tensor parallel distributed
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape((B, N, self.num_heads, -1)).transpose(perm=[0, 2, 1, 3])
        k = k.reshape((B, N, self.num_heads, -1)).transpose(perm=[0, 2, 1, 3])
        v = v.reshape((B, N, self.num_heads, -1)).transpose(perm=[0, 2, 1, 3])

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
            x = self.proj(x)
            with get_rng_state_tracker().rng_state("global_seed"):
                x = self.proj_drop(x)
        else:
            q = q * self.scale
            if self.qk_float:
                k = k.astype(dtype="float32")
                q = q.astype(dtype="float32")
                attn = q.matmul(k.transpose(perm=[0, 1, 3, 2]))
                # [B, num_heads, N, C] * [B, num_heads, C, N] -> [B, num_heads, N, N]
            else:
                attn = q.matmul(k.transpose(perm=[0, 1, 3, 2]))
            if self.relative_position_bias_table is not None:
                relative_position_bias = self.relative_position_bias_table[
                    self.relative_position_index.reshape((-1))
                ].reshape(
                    (self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1)
                )
                relative_position_bias = relative_position_bias.transpose(perm=[2, 0, 1])
                attn = attn + relative_position_bias.unsqueeze(axis=0).astype(dtype=attn.dtype)

            if self.rel_pos_bias is not None:
                attn = attn + self.rel_pos_bias().astype(dtype=attn.dtype)

            if rel_pos_bias is not None:
                attn = attn + rel_pos_bias.astype(dtype=attn.dtype)
            if attn_mask is not None:
                attn_mask = attn_mask.astype(dtype="bool")
                attn = paddle.where(~attn_mask[:, (None), (None), :], attn, float("-inf"))
            attn = F.softmax(attn, axis=-1)
            with get_rng_state_tracker().rng_state("global_seed"):
                attn = self.attn_drop(attn)
            x = (attn.matmul(v)).transpose(perm=[0, 2, 1, 3]).reshape([B, N, -1])
            x = self.proj(x)
            with get_rng_state_tracker().rng_state("global_seed"):
                x = self.proj_drop(x)
        return x


class Block(paddle.nn.Layer):
    def __init__(
        self,
        config,
        drop_path=0.0,
        window_size=None,
        rope=None,
        act_layer=paddle.nn.GELU,
        norm_layer=paddle.nn.LayerNorm,
    ):
        super().__init__()
        dim = config.embed_dim
        init_values = config.init_values

        self.norm1 = norm_layer(dim)
        self.attn = Attention(config, window_size=window_size, rope=rope)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()
        self.norm2 = norm_layer(dim)

        if config.swiglu or config.naiveswiglu:
            self.mlp = SwiGLU(config, norm_layer=norm_layer)
        else:
            self.mlp = Mlp(config, act_layer=act_layer)

        if init_values is not None and init_values > 0:
            init_data = init_values * paddle.ones(shape=dim)
            self.gamma_1 = self.create_parameter(
                shape=dim, default_initializer=paddle.nn.initializer.Assign(init_data)
            )
            init_data = init_values * paddle.ones(shape=dim)
            self.gamma_2 = self.create_parameter(
                shape=dim, default_initializer=paddle.nn.initializer.Assign(init_data)
            )
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.deepnorm = config.deepnorm
        if self.deepnorm:
            self.alpha = math.pow(2.0 * config.layers, 0.25)
        self.postnorm = config.postnorm

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            elif self.deepnorm:
                residual = x
                x = self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
                x = self.drop_path(x)
                x = residual * self.alpha + x
                x = self.norm1(x)

                residual = x
                x = self.mlp(x)
                x = self.drop_path(x)
                x = residual * self.alpha + x
                x = self.norm2(x)
            else:
                #
                x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.postnorm:
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

    def __init__(self, config):
        super().__init__()
        image_size = to_2tuple(config.image_size)
        patch_size = to_2tuple(config.patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.patch_shape = image_size[0] // patch_size[0], image_size[1] // patch_size[1]
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = paddle.nn.Conv2D(
            in_channels=config.in_chans, out_channels=config.embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        assert (
            H == self.image_size[0] and W == self.image_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
        x = self.proj(x).flatten(2).transpose([0, 2, 1])
        return x


class RelativePositionBias(paddle.nn.Layer):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        init_data = paddle.zeros(shape=[self.num_relative_distance, num_heads])
        self.relative_position_bias_table = self.create_parameter(
            shape=[self.num_relative_distance, num_heads], default_initializer=paddle.nn.initializer.Assign(init_data)
        )
        coords_h = paddle.arange(end=window_size[0])
        coords_w = paddle.arange(end=window_size[1])
        coords = paddle.stack(x=paddle.meshgrid([coords_h, coords_w]))
        coords_flatten = paddle.flatten(x=coords, start_axis=1)
        relative_coords = coords_flatten[:, :, (None)] - coords_flatten[:, (None), :]
        relative_coords = relative_coords.transpose(perm=[1, 2, 0])
        relative_coords[:, :, (0)] += window_size[0] - 1
        relative_coords[:, :, (1)] += window_size[1] - 1
        relative_coords[:, :, (0)] *= 2 * window_size[1] - 1
        relative_position_index = paddle.zeros(
            shape=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(axis=-1)
        relative_position_index[(0), 0:] = self.num_relative_distance - 3
        relative_position_index[0:, (0)] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape((-1))].reshape(
            (self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1)
        )
        return relative_position_bias.transpose(perm=[2, 0, 1])


def _mask_1d_rel_pos_index(seq_len):
    index = paddle.arange(end=seq_len)
    return index.reshape([1, seq_len]) - index.reshape([seq_len, 1]) + seq_len - 1


def _add_cls_to_index_matrix(index, num_tokens, offset):
    index = index.reshape([num_tokens, num_tokens])
    new_index = paddle.zeros(shape=(num_tokens + 1, num_tokens + 1), dtype=index.dtype)
    new_index[1:, 1:] = index
    new_index[0, 0:] = offset
    new_index[0:, 0] = offset + 1
    new_index[0, 0] = offset + 2
    return new_index


class DecoupledRelativePositionBias(nn.Layer):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = 2 * window_size[0] + 2, 2 * window_size[1] + 2

        num_tokens = window_size[0] * window_size[1]

        init_data = paddle.zeros(shape=[self.num_relative_distance[0], num_heads])
        self.relative_position_bias_for_high = self.create_parameter(
            shape=[self.num_relative_distance[0], num_heads],
            default_initializer=paddle.nn.initializer.Assign(init_data),
        )
        init_data = paddle.zeros(shape=[self.num_relative_distance[1], num_heads])
        self.relative_position_bias_for_width = self.create_parameter(
            shape=[self.num_relative_distance[1], num_heads],
            default_initializer=paddle.nn.initializer.Assign(init_data),
        )
        # cls to token & token 2 cls & cls to cls

        h_index = (
            _mask_1d_rel_pos_index(window_size[0])
            .reshape([window_size[0], 1, window_size[0], 1])
            .expand([-1, window_size[1], -1, window_size[1]])
        )
        h_index = _add_cls_to_index_matrix(h_index, num_tokens, 2 * window_size[0] - 1)
        self.register_buffer("relative_position_high_index", h_index)

        w_index = (
            _mask_1d_rel_pos_index(window_size[1])
            .reshape([1, window_size[1], 1, window_size[1]])
            .expand([window_size[0], -1, window_size[0], -1])
        )
        w_index = _add_cls_to_index_matrix(w_index, num_tokens, 2 * window_size[1] - 1)

        self.register_buffer("relative_position_width_index", w_index)

    def forward(self):
        relative_position_bias = F.embedding(
            input=self.relative_position_high_index, weight=self.relative_position_bias_for_high
        ) + F.embedding(input=self.relative_position_width_index, weight=self.relative_position_bias_for_width)
        return relative_position_bias.transpose(perm=[2, 0, 1])


class EVA02VisionTransformerConfig(PretrainedConfig):

    model_type = "eva02_vit_finetune"
    attribute_map: Dict[str, str] = {}

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_values=None,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        use_mean_pooling=True,
        init_scale=0.001,
        enable_recompute=False,
        stop_grad_conv1=True,  #
        postnorm=False,
        deepnorm=False,  #
        subln=True,
        xattn=False,
        swiglu=False,
        naiveswiglu=False,
        rope=True,
        pt_hw_seq_len=16,
        intp_freq=False,
        fusedLN=False,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.layers = layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.init_values = init_values
        self.use_abs_pos_emb = use_abs_pos_emb
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_shared_rel_pos_bias = use_shared_rel_pos_bias
        self.use_mean_pooling = use_mean_pooling
        self.init_scale = init_scale
        self.enable_recompute = enable_recompute
        self.stop_grad_conv1 = stop_grad_conv1
        self.deepnorm = deepnorm
        self.postnorm = postnorm
        self.xattn = xattn
        self.intp_freq = intp_freq
        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu
        self.rope = rope
        self.pt_hw_seq_len = pt_hw_seq_len
        self.subln = subln
        self.fusedLN = fusedLN

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class EVA02VisionTransformerPretrainedModel(MixPretrainedModel):
    """
    See :class:`paddlemix.models.model_utils.MixPretrainedModel` for more details.
    """

    model_config_file = "config.json"
    config_class = EVA02VisionTransformerConfig
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "eva02_vit_finetune"


class EVA02VisionTransformer(EVA02VisionTransformerPretrainedModel):
    def __init__(self, config: EVA02VisionTransformerConfig):
        super(EVA02VisionTransformer, self).__init__(config)
        self.image_size = config.image_size
        self.enable_recompute = config.enable_recompute
        self.num_classes = num_classes = config.num_classes
        self.embed_dim = embed_dim = config.embed_dim
        self.swiglu = config.swiglu
        self.naiveswiglu = config.naiveswiglu
        use_mean_pooling = config.use_mean_pooling
        norm_layer = partial(FusedLayerNorm, epsilon=1e-6)
        self.num_heads = num_heads = config.num_heads

        self.patch_embed = PatchEmbed(config)
        num_patches = self.patch_embed.num_patches

        init_data = paddle.zeros(shape=[1, 1, embed_dim])
        self.cls_token = self.create_parameter(
            shape=[1, 1, embed_dim], default_initializer=paddle.nn.initializer.Assign(init_data)
        )
        if config.use_abs_pos_emb:
            init_data = paddle.zeros(shape=[1, num_patches + 1, embed_dim])
            self.pos_embed = self.create_parameter(
                shape=[1, num_patches + 1, embed_dim], default_initializer=paddle.nn.initializer.Assign(init_data)
            )
        else:
            self.pos_embed = None
        self.pos_drop = paddle.nn.Dropout(p=config.drop_rate)

        # TODO
        self.stop_grad_conv1 = config.stop_grad_conv1
        if self.stop_grad_conv1:
            self.patch_embed.proj.weight.stop_gradient = True
            self.patch_embed.proj.bias.stop_gradient = True

        if config.use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        if config.rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = config.image_size // config.patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim, pt_seq_len=config.pt_hw_seq_len, ft_seq_len=hw_seq_len if config.intp_freq else None
            )
        else:
            self.rope = None

        dpr = [x.item() for x in paddle.linspace(0, config.drop_path_rate, config.layers)]
        self.blocks = paddle.nn.LayerList(
            sublayers=[
                Block(
                    config,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    window_size=self.patch_embed.patch_shape if config.use_rel_pos_bias else None,
                    rope=self.rope,
                )
                for i in range(config.layers)
            ]
        )

        self.norm = paddle.nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        if dist.get_world_size() > 1:
            self.head = (
                fleet.meta_parallel.ColumnParallelLinear(
                    embed_dim, num_classes, weight_attr=None, has_bias=True, gather_output=True
                )
                if num_classes > 0
                else paddle.nn.Identity()
            )
        else:
            self.head = paddle.nn.Linear(embed_dim, num_classes) if num_classes > 0 else paddle.nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        self.apply(self._init_weights)
        self.fix_init_weight()
        if isinstance(self.head, fleet.meta_parallel.ColumnParallelLinear):
            trunc_normal_(self.head.weight, std=0.02)
            with paddle.no_grad():
                self.head.weight.set_value(self.head.weight.scale(scale=config.init_scale))
                self.head.bias.set_value(self.head.bias.scale(scale=config.init_scale))

    def fix_init_weight(self):
        def rescale(param, layer_id):
            origin_dtype = paddle.get_default_dtype()
            paddle.set_default_dtype("float32")
            tmp = paddle.to_tensor(math.sqrt(2.0 * layer_id))
            paddle.set_default_dtype(origin_dtype)
            if origin_dtype != "float32":
                tmp = tmp.astype(origin_dtype)
            param = param.divide(tmp)

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight, layer_id + 1)
            if self.swiglu or self.naiveswiglu:
                rescale(layer.mlp.w3.weight, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight, layer_id + 1)

    def get_cast_dtype(self) -> paddle.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def _init_weights(self, m):
        zeros_params = paddle.nn.initializer.Constant(0.0)
        ones_params = paddle.nn.initializer.Constant(1.0)
        if isinstance(m, (paddle.nn.Linear, fleet.meta_parallel.ColumnParallelLinear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                zeros_params(m.bias)
        elif isinstance(m, paddle.nn.LayerNorm):
            zeros_params(m.bias)
            ones_params(m.weight)

    def get_num_layers(self):
        return len(self.blocks)

    def set_grad_checkpointing(self, enable=True):
        self.enable_recompute = enable

    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        if dist.get_world_size() > 1:
            self.head = (
                fleet.meta_parallel.ColumnParallelLinear(
                    self.embed_dim, num_classes, weight_attr=None, has_bias=True, gather_output=True
                )
                if num_classes > 0
                else paddle.nn.Identity()
            )
        else:
            self.head = paddle.nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else paddle.nn.Identity()

    def forward_features(self, x, return_patch_tokens=False):
        x = self.patch_embed(x)

        # if self.stop_grad_conv1: # TODO
        #     x = x.detach()

        batch_size, seq_len, _ = x.shape
        cls_tokens = self.cls_token.expand(shape=[batch_size, -1, -1])

        x = paddle.concat(x=(cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        with get_rng_state_tracker().rng_state("global_seed"):
            x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        cnt = 0
        for blk in self.blocks:
            cnt += 1
            if self.enable_recompute:
                x = paddle.distributed.fleet.utils.recompute(blk, x, rel_pos_bias, use_reentrant=False)
            else:
                x = blk(x, rel_pos_bias=rel_pos_bias)

        x = self.norm(x)
        if self.fc_norm is not None:
            t = x[:, 1:, :]
            if return_patch_tokens:
                return self.fc_norm(t)
            else:
                return self.fc_norm(t.mean(1))
        else:
            if return_patch_tokens:
                return x[:, 1:]
            else:
                return x[:, 0]

    def forward(self, image, return_patch_tokens=False):
        x = self.forward_features(image, return_patch_tokens=return_patch_tokens)
        x = self.head(x)
        return x
