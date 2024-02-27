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

import numpy as np
import math
from itertools import repeat
import collections.abc
from functools import partial
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.incubate.nn.memory_efficient_attention import memory_efficient_attention
from paddle.nn.initializer import Constant, Normal, TruncatedNormal
trunc_normal_ = TruncatedNormal(std=0.02)
normal_ = Normal
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)
import paddle.nn.initializer as initializer


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Layer):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose((0, 2, 1))
        return x


class Mlp(nn.Layer):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2D, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias_attr=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias_attr=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Layer):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            fused_attn: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Layer = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, self.head_dim]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            # x = F.scaled_dot_product_attention(
            #     q, k, v,
            #     dropout_p=self.attn_drop.p if self.training else 0.,
            # )
            q = q.transpose(perm=[0, 2, 1, 3])
            k = k.transpose(perm=[0, 2, 1, 3])
            v = v.transpose(perm=[0, 2, 1, 3])
            x = memory_efficient_attention(
                q, k, v,
                p=self.xattn_drop.p if self.training else 0.,
                scale=self.scale,
            )
            x = x.reshape([B, N, -1])
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            q = q * self.scale
            attn = q @ k.transpose([0, 1, 3, 2])
            attn = F.softmax(attn, axis=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

            x = x.transpose([0, 2, 1, 3]).reshape([B, N, C])
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Layer):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias_attr=True),
            nn.Silu(),
            nn.Linear(hidden_size, hidden_size, bias_attr=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = paddle.exp(
            -math.log(max_period) * paddle.arange(start=0, end=half, dtype=paddle.float32) / half
        )
        args = t[:, None].cast('float32') * freqs[None]
        embedding = paddle.concat([paddle.cos(args), paddle.sin(args)], axis=-1)
        if dim % 2:
            embedding = paddle.concat([embedding, paddle.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Layer):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                paddle.rand(
                    (labels.shape[0],),
                )
                < self.dropout_prob
            )
        else:
            drop_ids = paddle.to_tensor(force_drop_ids == 1)
        labels = paddle.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (self.training and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class DiTBlock(nn.Layer):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, fused_attn=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, epsilon=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, fused_attn=fused_attn, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, epsilon=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate=True) # 'tanh'
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.Silu(),
            nn.Linear(hidden_size, 6 * hidden_size, bias_attr=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, axis=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Layer):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, epsilon=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias_attr=True)
        self.adaLN_modulation = nn.Sequential(
            nn.Silu(),
            nn.Linear(hidden_size, 2 * hidden_size, bias_attr=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SiT(nn.Layer):
    """
    Diffusion model with a Transformer backbone.
    """
    _supports_gradient_checkpointing = True
    _use_memory_efficient_attention_xformers = True

    def __init__(
        self,
        input_size=32, # image_size // 8
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        #enable_recompute=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.learn_sigma = learn_sigma
        #self.enable_recompute = enable_recompute
        self.gradient_checkpointing = False
        self.fused_attn = False

        self.out_channels = in_channels * 2 if learn_sigma else in_channels

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = self.create_parameter(
            shape=(1, num_patches, hidden_size), default_initializer=zeros_
        )
        self.add_parameter("pos_embed", self.pos_embed)

        self.blocks = nn.LayerList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, fused_attn=self.fused_attn) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                initializer.XavierUniform()(module.weight)
                if module.bias is not None:
                    initializer.Constant(value=0)(module.bias)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.set_value(paddle.to_tensor(pos_embed, dtype='float32').unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2D):
        w = self.x_embedder.proj.weight
        initializer.XavierUniform()(w.reshape([w.shape[0], -1]))
        initializer.Constant(value=0)(self.x_embedder.proj.bias)

        # Initialize label embedding table:
        initializer.Normal(std=0.02)(self.y_embedder.embedding_table.weight)

        # Initialize timestep embedding MLP:
        initializer.Normal(std=0.02)(self.t_embedder.mlp[0].weight)
        initializer.Normal(std=0.02)(self.t_embedder.mlp[2].weight)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            initializer.Constant(value=0)(block.adaLN_modulation[-1].weight)
            initializer.Constant(value=0)(block.adaLN_modulation[-1].bias)

        # Zero-out output layers:
        initializer.Constant(value=0)(self.final_layer.adaLN_modulation[-1].weight)
        initializer.Constant(value=0)(self.final_layer.adaLN_modulation[-1].bias)
        initializer.Constant(value=0)(self.final_layer.linear.weight)
        initializer.Constant(value=0)(self.final_layer.linear.bias)

    def enable_gradient_checkpointing(self, enable=True):
        self.gradient_checkpointing = enable

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[str] = None):
        self._use_memory_efficient_attention_xformers = True
        self.fused_attn = True

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = paddle.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y)    # (N, D)
        c = t + y                                # (N, D)

        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing:
                x = paddle.distributed.fleet.utils.recompute(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)       # (N, T, D)

        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        
        ### Note: diff with DiT
        if self.learn_sigma:
            x, _ = x.chunk(2, axis=1)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of SiT, but also batches the unconSiTional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = paddle.concat([half, half], axis=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = paddle.split(eps, 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = paddle.concat([half_eps, half_eps], axis=0)
        return paddle.concat([eps, rest], axis=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
