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

from dataclasses import dataclass
from typing import Optional

import einops
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, is_ppxformers_available
from .embeddings import PatchEmbed, get_timestep_embedding
from .modeling_utils import ModelMixin


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def unpatchify(x, in_chans):
    patch_size = int((x.shape[2] // in_chans) ** 0.5)
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1] and patch_size ** 2 * in_chans == x.shape[2]
    x = einops.rearrange(x, "B (h w) (p1 p2 C) -> B C (h p1) (w p2)", h=h, p1=patch_size, p2=patch_size)
    return x


class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_size = head_dim
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._use_memory_efficient_attention_xformers = is_ppxformers_available()
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
                dropout_p=self.attn_drop,
                training=self.training,
                attention_op=self._attention_op,
            )
        else:
            with paddle.amp.auto_cast(enable=False):
                attention_scores = paddle.matmul(query_proj * self.scale, key_proj, transpose_y=True)
                attention_probs = F.softmax(attention_scores, axis=-1)
                hidden_states = paddle.matmul(attention_probs, value_proj).cast(x.dtype)

        hidden_states = self.reshape_batch_dim_to_heads(
            hidden_states, transpose=not self._use_memory_efficient_attention_xformers
        )

        hidden_states = self.proj_drop(self.proj(hidden_states))
        return hidden_states


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
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        skip=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = nn.Identity()  # infer always be 0.0

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None

    def forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(paddle.concat([x, skip], axis=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class UViTT2IModelOutput(BaseOutput):
    """
    Args:
        sample (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states output. Output of last layer of model.
    """

    sample: paddle.Tensor


class UViTT2IModel(ModelMixin, ConfigMixin):
    r"""
    UViTT2IModel is a simple version of U-ViT used for text to image
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        patch_size=2,
        num_layers=28,
        num_attention_heads=16,
        attention_head_dim=72,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        pos_drop_rate=0.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_type: str = "layer_norm",
        clip_dim=768,
        num_text_tokens=77,
        conv=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size

        embed_dim = num_attention_heads * attention_head_dim
        self.embed_dim = embed_dim
        depth = num_layers
        num_heads = num_attention_heads

        self.img_size = (sample_size, sample_size) if isinstance(sample_size, int) else sample_size
        self.patch_embed = PatchEmbed(
            height=self.img_size[0],
            width=self.img_size[1],
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            add_pos_embed=False,
        )
        num_patches = (self.img_size[0] // patch_size) * (self.img_size[1] // patch_size)

        self.context_embed = nn.Linear(clip_dim, embed_dim)
        self.extras = 1 + num_text_tokens
        self.pos_embed = self.create_parameter(
            shape=(1, self.extras + num_patches, embed_dim),
            default_initializer=nn.initializer.Constant(0.0),
        )
        assert norm_type in ["layer_norm"], f"Invalid norm type: {norm_type}"
        norm_layer = nn.LayerNorm
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        self.in_blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
                for _ in range(depth // 2)
            ]
        )

        self.mid_block = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
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
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    skip=True,
                )
                for _ in range(depth // 2)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size**2 * in_channels
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias_attr=True)
        self.final_layer = nn.Conv2D(self.in_channels, self.in_channels, 3, padding=1) if conv else nn.Identity()
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self, value=True):
        self.gradient_checkpointing = value

    def forward(
        self,
        img: paddle.Tensor,
        timesteps: paddle.Tensor,
        context: paddle.Tensor,
        return_dict=True,
    ):
        img = img.cast(self.dtype)
        context = context.cast(self.dtype)
        x = self.patch_embed(img)
        B, L, D = x.shape

        timesteps = timesteps.expand(
            [
                img.shape[0],
            ]
        )
        time_token = get_timestep_embedding(timesteps, self.embed_dim, True, 0).unsqueeze(axis=1)
        context_token = self.context_embed(context)
        time_token = time_token.cast(self.dtype)
        x = paddle.concat((time_token, context_token, x), 1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        skips = []
        for blk in self.in_blocks:
            if self.gradient_checkpointing:
                x = paddle.distributed.fleet.utils.recompute(blk, x)
            else:
                x = blk(x)
            skips.append(x)

        if self.gradient_checkpointing:
            x = paddle.distributed.fleet.utils.recompute(self.mid_block, x)
        else:
            x = self.mid_block(x)

        for blk in self.out_blocks:
            if self.gradient_checkpointing:
                x = paddle.distributed.fleet.utils.recompute(blk, x, skips.pop())
            else:
                x = blk(x, skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)
        assert x.shape[1] == self.extras + L

        x = x[:, self.extras :, :]
        x = unpatchify(x, self.in_channels)
        x = self.final_layer(x)

        if not return_dict:
            return (x,)

        return UViTT2IModelOutput(sample=x)
