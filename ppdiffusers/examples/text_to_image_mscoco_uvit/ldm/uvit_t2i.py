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

import math
from typing import Optional

import einops
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker

from ppdiffusers.configuration_utils import ConfigMixin
from ppdiffusers.models.modeling_utils import ModelMixin
from ppdiffusers.utils import is_ppxformers_available


def is_model_parrallel():
    if paddle.distributed.get_world_size() > 1:
        hcg = paddle.distributed.fleet.get_hybrid_communicate_group()
        if hcg.get_model_parallel_world_size() > 1:
            return True
        else:
            return False
    else:
        return False


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if is_model_parrallel():
            self.fc1 = fleet.meta_parallel.ColumnParallelLinear(
                in_features,
                hidden_features,
                weight_attr=None,
                has_bias=True,
                gather_output=True,
            )
            self.fc2 = fleet.meta_parallel.ColumnParallelLinear(
                hidden_features,
                out_features,
                weight_attr=None,
                has_bias=True,
                gather_output=True,
            )
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        with get_rng_state_tracker().rng_state("global_seed"):
            x = self.drop(x)
        x = self.fc2(x)
        with get_rng_state_tracker().rng_state("global_seed"):
            x = self.drop(x)
        return x


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = paddle.exp(-math.log(max_period) * paddle.arange(start=0, end=half, dtype=paddle.float32) / half)
    args = timesteps[:, None].cast("float32") * freqs[None]
    embedding = paddle.concat([paddle.cos(args), paddle.sin(args)], axis=-1)
    if dim % 2:
        embedding = paddle.concat([embedding, paddle.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, "B (h w) (p1 p2 C) -> B C (h p1) (w p2)", h=h, p1=patch_size, p2=patch_size)
    return x


class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_size = head_dim
        self.scale = qk_scale or head_dim**-0.5

        if is_model_parrallel():
            self.qkv = fleet.meta_parallel.ColumnParallelLinear(
                dim, dim * 3, weight_attr=None, has_bias=qkv_bias, gather_output=True
            )
            self.proj = fleet.meta_parallel.ColumnParallelLinear(
                dim, dim, weight_attr=None, has_bias=True, gather_output=True
            )
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
            self.proj = nn.Linear(dim, dim)

        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self._use_memory_efficient_attention_xformers = True
        self._attention_op = None

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

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[str] = None
    ):
        if self.head_size > 128 and attention_op == "flash":
            attention_op = "cutlass"
        if use_memory_efficient_attention_xformers:
            if not is_ppxformers_available():
                raise NotImplementedError(
                    "requires the scaled_dot_product_attention but your PaddlePaddle do not have this. Checkout the instructions on the installation page: https://www.paddlepaddle.org.cn/install/quick and follow the ones that match your environment."
                )
            else:
                try:
                    _ = F.scaled_dot_product_attention_(
                        paddle.ones((1, 1, 2, 40), dtype=paddle.float16),
                        paddle.ones((1, 1, 2, 40), dtype=paddle.float16),
                        paddle.ones((1, 1, 2, 40), dtype=paddle.float16),
                        attention_op=attention_op,
                    )
                except Exception as e:
                    raise e

        self._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
        self._attention_op = attention_op

    def forward(self, x):
        qkv = self.qkv(x)
        if not self._use_memory_efficient_attention_xformers:
            qkv = qkv.cast(paddle.float32)
        query_proj, key_proj, value_proj = qkv.chunk(3, axis=-1)
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
                dropout_p=self.attn_drop_p,
                training=self.training,
                attention_op=self._attention_op,
            )
        else:
            with paddle.amp.auto_cast(enable=False):
                attention_scores = paddle.matmul(query_proj * self.scale, key_proj, transpose_y=True)
                attention_probs = F.softmax(attention_scores, axis=-1)
                with get_rng_state_tracker().rng_state("global_seed"):
                    attention_probs = self.attn_drop(attention_probs)
                hidden_states = paddle.matmul(attention_probs, value_proj).cast(x.dtype)

        x = self.reshape_batch_dim_to_heads(hidden_states, transpose=not self._use_memory_efficient_attention_xformers)
        x = self.proj(x)
        with get_rng_state_tracker().rng_state("global_seed"):
            x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        skip=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, weight_attr=False, bias_attr=False)  #
        self.norm2 = norm_layer(dim, weight_attr=False, bias_attr=False)  #
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        if skip:
            if is_model_parrallel():
                self.skip_linear = fleet.meta_parallel.ColumnParallelLinear(
                    2 * dim, dim, weight_attr=None, has_bias=True, gather_output=True
                )
            else:
                self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

    def forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(paddle.concat([x, skip], axis=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Layer):
    """Image to Patch Embedding"""

    def __init__(self, patch_size, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2D(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = self.proj(x).flatten(2).transpose([0, 2, 1])
        return x


class UViTT2IModel(ModelMixin, ConfigMixin):
    def __init__(
        self,
        sample_size=32,
        patch_size=2,
        in_channels=4,
        out_channels=4,
        num_layers=28,
        num_attention_heads=16,
        attention_head_dim=72,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        pos_drop_rate=0.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        clip_dim=768,
        num_text_tokens=77,
        conv=True,
    ):
        super().__init__()
        embed_dim = num_attention_heads * attention_head_dim
        depth = num_layers
        num_heads = num_attention_heads

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.in_channels = in_channels

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        num_patches = (sample_size // patch_size) ** 2

        if is_model_parrallel():
            self.context_embed = fleet.meta_parallel.ColumnParallelLinear(
                clip_dim, embed_dim, weight_attr=None, has_bias=True, gather_output=True
            )
        else:
            self.context_embed = nn.Linear(clip_dim, embed_dim)
        self.extras = 1 + num_text_tokens
        self.pos_embed = self.create_parameter(
            shape=(1, self.extras + num_patches, embed_dim),
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        dpr = np.linspace(0, drop_rate, depth + 1)
        self.in_blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=dpr[i],
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
                for i in range(depth // 2)
            ]
        )

        self.mid_block = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=dpr[depth // 2],
            attn_drop=attn_drop_rate,
            norm_layer=norm_layer,
        )

        self.out_blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=dpr[i + 1 + depth // 2],
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    skip=True,
                )
                for i in range(depth // 2)
            ]
        )
        self.norm = norm_layer(embed_dim, weight_attr=False, bias_attr=False)
        self.patch_dim = patch_size**2 * in_channels

        if is_model_parrallel():
            self.decoder_pred = fleet.meta_parallel.ColumnParallelLinear(
                embed_dim, self.patch_dim, weight_attr=None, has_bias=True, gather_output=True
            )
        else:
            self.decoder_pred = nn.Linear(embed_dim, self.patch_dim)
        self.final_layer = nn.Conv2D(self.in_channels, self.in_channels, 3, padding=1) if conv else nn.Identity()

        self.gradient_checkpointing = False
        self.fused_attn = False

    def enable_gradient_checkpointing(self, value=True):
        self.gradient_checkpointing = value

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[str] = None):
        self._use_memory_efficient_attention_xformers = True
        self.fused_attn = True

    def no_weight_decay(self):
        return {"pos_embed"}

    def forward(self, x, timesteps, encoder_hidden_states):
        encoder_hidden_states = encoder_hidden_states.cast(x.dtype)
        x = self.patch_embed(x)

        B, L, D = x.shape  # [bs, 256, embed_dim]

        timesteps = timesteps.expand(
            [
                x.shape[0],
            ]
        )
        timesteps = timesteps.cast(x.dtype)
        time_token = timestep_embedding(timesteps, self.embed_dim)
        time_token = time_token.unsqueeze(axis=1)  # [bs, 1, embed_dim]
        context_token = self.context_embed(encoder_hidden_states)  # [bs, 77, 768]->[bs, 77, embed_dim]
        x = paddle.concat((time_token, context_token, x), 1)

        x = x + self.pos_embed
        with get_rng_state_tracker().rng_state("global_seed"):
            x = self.pos_drop(x)

        skips = []
        for i, blk in enumerate(self.in_blocks):
            if self.gradient_checkpointing:
                x = paddle.distributed.fleet.utils.recompute(blk, x)
            else:
                x = blk(x)
            skips.append(x)

        if self.gradient_checkpointing:
            x = paddle.distributed.fleet.utils.recompute(self.mid_block, x)
        else:
            x = self.mid_block(x)

        for i, blk in enumerate(self.out_blocks):
            if self.gradient_checkpointing:
                x = paddle.distributed.fleet.utils.recompute(blk, x, skips.pop())
            else:
                x = blk(x, skips.pop())

        x = self.norm(x)  # eps 1e-05
        x = self.decoder_pred(x)
        assert x.shape[1] == self.extras + L

        x = x[:, self.extras :, :]
        x = unpatchify(x, self.in_channels)
        x = self.final_layer(x)  # conv
        return x
