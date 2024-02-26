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

from functools import partial

import paddle
import paddle.nn as nn

from ..clap_module.htsat_model import SwinTransformerBlock
from ..utils import DropPath, Mlp, to_2tuple


class Attention(nn.Layer):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Layer = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = False

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, self.head_dim]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout=self.attn_drop.p if self.training else 0.0,
            )[0]
        else:
            q = q * self.scale
            k_perm = list(range(k.dim()))
            new_perm = k_perm
            new_perm[-2], new_perm[-1] = k_perm[-1], k_perm[-2]
            attn = q @ k.transpose(new_perm)
            attn = nn.functional.softmax(attn, axis=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x_perm = list(range(x.dim()))
        new_perm = x_perm
        new_perm[1], new_perm[2] = x_perm[2], x_perm[1]
        x = x.transpose(new_perm).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Layer):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        tmp = init_values * paddle.ones(dim)
        self.gamma = paddle.create_parameter(
            shape=tmp.shape, dtype=tmp.dtype, default_initializer=nn.initializer.Assign(tmp)
        )
        self.gamma.stop_gradient = False

    def forward(self, x):
        if self.inplace:
            x = paddle.multiply(x, self.gamma)
            return x
        else:
            return x * self.gamma
        # return paddle.multiply(x, self.gamma) if self.inplace else x * self.gamma


class Block(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class PatchEmbed_org(nn.Layer):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        y = x.flatten(2).transpose([0, 2, 1])
        return y


class MaskedAutoencoderViT(nn.Layer):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=10,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        audio_exp=False,
        alpha=0.0,
        temperature=0.2,
        mode=0,
        contextual_depth=8,
        split_pos=False,
        pos_trainable=False,
        use_nce=False,
        beta=4.0,
        decoder_mode=0,
        mask_t_prob=0.6,
        mask_f_prob=0.5,
        mask_2d=False,
        epoch=0,
        no_shift=False,
        use_custom_patch=False,
    ):
        super().__init__()

        self.audio_exp = audio_exp
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed_org(img_size, patch_size, in_chans, embed_dim)
        self.use_custom_patch = use_custom_patch

        num_patches = self.patch_embed.num_patches
        tmp = paddle.zeros([1, 1, embed_dim])
        self.cls_token = paddle.create_parameter(
            shape=tmp.shape, dtype=tmp.dtype, default_initializer=nn.initializer.Assign(tmp)
        )
        self.cls_token.stop_gradient = False

        # self.split_pos = split_pos # not useful
        tmp = paddle.zeros([1, num_patches + 1, embed_dim])
        self.pos_embed = paddle.create_parameter(
            shape=tmp.shape, dtype=tmp.dtype, default_initializer=nn.initializer.Assign(tmp)
        )  # fixed sin-cos embedding
        self.pos_embed.stop_gradient = not pos_trainable

        self.encoder_depth = depth
        self.contextual_depth = contextual_depth
        self.blocks = nn.LayerList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )  # qk_scale=None
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias_attr=True)

        tmp = paddle.zeros([1, 1, decoder_embed_dim])
        self.mask_token = paddle.create_parameter(
            shape=tmp.shape, dtype=tmp.dtype, default_initializer=nn.initializer.Assign(tmp)
        )
        self.mask_token.stop_gradient = False

        tmp = paddle.zeros([1, num_patches + 1, decoder_embed_dim])
        self.decoder_pos_embed = paddle.create_parameter(
            shape=tmp.shape, dtype=tmp.dtype, default_initializer=nn.initializer.Assign(tmp)
        )  # fixed sin-cos embedding
        self.decoder_pos_embed.stop_gradient = not pos_trainable

        self.no_shift = no_shift

        self.decoder_mode = decoder_mode
        if self.use_custom_patch:  # overlapped patches as in AST. Similar performance yet compute heavy
            window_size = (6, 6)
            feat_size = (102, 12)
        else:
            window_size = (4, 4)
            feat_size = (64, 8)
        if self.decoder_mode == 1:
            decoder_modules = []
            for index in range(16):
                if self.no_shift:
                    shift_size = (0, 0)
                else:
                    if (index % 2) == 0:
                        shift_size = (0, 0)
                    else:
                        shift_size = (2, 0)
                decoder_modules.append(
                    SwinTransformerBlock(
                        dim=decoder_embed_dim,
                        num_heads=16,
                        input_resolution=feat_size,
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=mlp_ratio,
                        drop=0.0,
                        attn_drop=0.0,
                        drop_path=0.0,
                        # extra_norm=False,
                        # sequential_attn=False,
                        norm_layer=norm_layer,  # nn.LayerNorm,
                    )
                )
            self.decoder_blocks = nn.LayerList(decoder_modules)
        else:
            # Transformer
            self.decoder_blocks = nn.LayerList(
                [
                    Block(
                        decoder_embed_dim,
                        decoder_num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                    )  # qk_scale=None,
                    for i in range(decoder_depth)
                ]
            )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias_attr=True
        )  # decoder to patch

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.patch_size = patch_size
        self.stride = stride

        # audio exps
        self.alpha = alpha
        self.T = temperature
        self.mode = mode
        self.use_nce = use_nce
        self.beta = beta

        self.log_softmax = nn.LogSoftmax(axis=-1)

        self.mask_t_prob = mask_t_prob
        self.mask_f_prob = mask_f_prob
        self.mask_2d = mask_2d

        self.epoch = epoch

        # self.initialize_weights()

    def forward_encoder_no_mask(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand([x.shape[0], -1, -1])
        x = paddle.concat((cls_tokens, x), axis=1)

        # apply Transformer blocks
        contextual_embs = []
        for n, blk in enumerate(self.blocks):
            x = blk(x)
            if n > self.contextual_depth:
                contextual_embs.append(self.norm(x))
        contextual_emb = paddle.stack(contextual_embs, axis=0).mean(axis=0)

        return contextual_emb


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        **kwargs,
    )
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
