# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable

import numpy as np
import paddle
import paddle.nn as nn
from paddle import _legacy_C_ops
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddle.nn.functional.flash_attention import flash_attention
from paddle.nn.initializer import Constant, Normal, TruncatedNormal

from paddlemix.models.blip2.configuration import Blip2VisionConfig
from paddlemix.models.blip2.modeling import Blip2PretrainedModel

# Code was based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# reference: https://arxiv.org/abs/2010.11929
from paddlemix.utils.log import logger

trunc_normal_ = TruncatedNormal(std=0.02)
normal_ = Normal
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)
from paddle.distributed.fleet.utils import recompute


def to_2tuple(x):
    return tuple([x] * 2)


def drop_path(x, drop_prob=0.0, training=False):

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob, dtype=x.dtype)
    shape = (paddle.shape(x)[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape).astype(x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        mp_degree=1,
        use_fusedlinear=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if mp_degree > 1:
            self.fc1 = fleet.meta_parallel.ColumnParallelLinear(
                in_features, hidden_features, weight_attr=None, has_bias=True, gather_output=True
            )
            self.fc2 = nn.Linear(hidden_features, out_features)
        else:
            if use_fusedlinear:
                self.use_fusedlinear = True
                self.fc1 = paddle.incubate.nn.FusedLinear(in_features, hidden_features)
                self.fc2 = paddle.incubate.nn.FusedLinear(hidden_features, out_features)
            else:
                self.fc1 = nn.Linear(in_features, hidden_features)
                self.fc2 = nn.Linear(hidden_features, out_features)
        self.mp_degree = mp_degree
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if getattr(self, "use_fusedlinear", False):
            if isinstance(self.act, nn.GELU):
                x = _legacy_C_ops.fused_gemm_epilogue(
                    x, self.fc1.weight, self.fc1.bias, "trans_x", False, "trans_y", False, "activation", "gelu"
                )
            elif isinstance(self.act, nn.ReLU):
                x = _legacy_C_ops.fused_gemm_epilogue(
                    x, self.fc1.weight, self.fc1.bias, "trans_x", False, "trans_y", False, "activation", "relu"
                )
            else:
                ValueError
        else:
            x = self.fc1(x)
            x = self.act(x)
        x = self.fc2(x)
        if self.mp_degree > 1:
            with get_rng_state_tracker().rng_state("global_seed"):
                x = self.drop(x)
        else:
            x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        window_size=None,
        mp_degree=1,
        use_fusedlinear=False,
        use_flash_attn=False,
    ):
        super().__init__()
        self.use_flash_attn = use_flash_attn
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        if mp_degree > 1:
            self.qkv = fleet.meta_parallel.ColumnParallelLinear(
                dim, dim * 3, weight_attr=None, has_bias=True, gather_output=True
            )
        else:
            if use_fusedlinear:
                self.qkv = paddle.incubate.nn.FusedLinear(dim, dim * 3, bias_attr=qkv_bias)
            else:
                self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        if use_fusedlinear:
            self.proj = paddle.incubate.nn.FusedLinear(dim, dim)
        else:
            self.proj = nn.Linear(dim, dim)
        self.mp_degree = mp_degree
        self.proj_drop = nn.Dropout(proj_drop)

    def _register_relative_position_index(
        self,
        window_size,
        num_heads,
    ):
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = self.create_parameter(
            [self.num_relative_distance, num_heads], default_initializer=zeros_
        )  # 2*Wh-1 * 2*Ww-1, nH
        coords_h = paddle.arange(window_size[0])
        coords_w = paddle.arange(window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose([1, 2, 0])  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = paddle.zeros((window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, rel_pos_bias=None):
        N, C = x.shape[1:]
        if self.use_flash_attn:
            qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C // self.num_heads)).transpose((2, 0, 1, 3, 4))
        else:
            qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C // self.num_heads)).transpose((2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_flash_attn:
            x, _ = flash_attention(q, k, v, dropout=self.proj_drop.p, causal=False, return_softmax=False)
            x = paddle.reshape(x, [0, 0, -1])
        else:
            attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
            if hasattr(self, "relative_position_bias_table"):
                relative_position_bias = self.relative_position_bias_table[
                    self.relative_position_index.reshape([-1])
                ].reshape(
                    [self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1]
                )  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.transpose([2, 0, 1])  # nH, Wh*Ww, Wh*Ww
                attn = attn + relative_position_bias.unsqueeze(0)

            attn = nn.functional.softmax(attn, axis=-1)
            if self.mp_degree > 1:
                with get_rng_state_tracker().rng_state("global_seed"):
                    attn = self.attn_drop(attn)
            else:
                attn = self.attn_drop(attn)

            x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))

        x = self.proj(x)
        if self.mp_degree > 1:
            with get_rng_state_tracker().rng_state("global_seed"):
                x = self.proj_drop(x)
        else:
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
        init_values=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer="nn.LayerNorm",
        epsilon=1e-5,
        window_size=None,
        mp_degree=1,
        use_flash_attn=False,
        use_fusedlinear=False,
    ):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        elif isinstance(norm_layer, Callable):
            self.norm1 = norm_layer(dim)
        else:
            raise TypeError("The norm_layer must be str or paddle.nn.layer.Layer class")
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            mp_degree=mp_degree,
            use_flash_attn=use_flash_attn,
            use_fusedlinear=use_fusedlinear,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path)
        self.gamma_1 = None
        self.gamma_2 = None
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        elif isinstance(norm_layer, Callable):
            self.norm2 = norm_layer(dim)
        else:
            raise TypeError("The norm_layer must be str or paddle.nn.layer.Layer class")
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            mp_degree=mp_degree,
            use_fusedlinear=use_fusedlinear,
        )

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is not None:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RelativePositionBias(nn.Layer):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = self.create_parameter(
            [self.num_relative_distance, num_heads], default_initializer=zeros_
        )  # 2*Wh-1 * 2*Ww-1, nH

        coords_h = paddle.arange(window_size[0])
        coords_w = paddle.arange(window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose([1, 2, 0])  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = paddle.zeros((window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape([-1])].reshape(
            [self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1]
        )  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.transpose([2, 0, 1])  # nH, Wh*Ww, Wh*Ww


class PatchEmbed(nn.Layer):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
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


class VisionTransformer(Blip2PretrainedModel):
    """Vision Transformer with support for patch input"""

    main_input_name = "pixel_values"
    config_class = Blip2VisionConfig

    def __init__(self, config: Blip2VisionConfig, **kwargs):
        super().__init__(config)
        mp_degree = getattr(config, "mp_degree", 1)
        use_flash_attn = getattr(config, "use_flash_attn", False)
        use_fusedlinear = getattr(config, "use_fusedlinear", False)
        self.class_num = config.class_num
        self.num_features = self.embed_dim = config.embed_dim
        _img_size = to_2tuple(config.img_size)
        _patch_size = to_2tuple(config.patch_size)
        self.window_size = (_img_size[0] // _patch_size[0], _img_size[1] // _patch_size[1])
        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_chans=config.in_chans,
            embed_dim=config.embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = self.create_parameter(shape=(1, 1, config.embed_dim), default_initializer=zeros_)

        self.pos_embed = self.create_parameter(
            shape=(1, num_patches + 1, config.embed_dim), default_initializer=zeros_
        )

        self.add_parameter("pos_embed", self.pos_embed)

        self.add_parameter("cls_token", self.cls_token)
        self.pos_drop = nn.Dropout(p=config.drop_rate)
        self.gradient_checkpointing = config.gradient_checkpointing
        logger.info("self.gradient_checkpointing:{}".format(self.gradient_checkpointing))
        dpr = np.linspace(0, config.drop_path_rate, config.depth)

        self.blocks = nn.LayerList(
            [
                Block(
                    dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    qk_scale=config.qk_scale,
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=config.norm_layer,
                    epsilon=config.epsilon,
                    window_size=self.window_size,
                    mp_degree=mp_degree,
                    use_flash_attn=use_flash_attn,
                    use_fusedlinear=use_fusedlinear,
                )
                for i in range(config.depth)
            ]
        )

        self.mp_degree = mp_degree
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed)
        trunc_normal_(self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, fleet.meta_parallel.ColumnParallelLinear)):
            trunc_normal_(m.weight)
            if isinstance(m, (nn.Linear, fleet.meta_parallel.ColumnParallelLinear)) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def train(self, mode=True):
        """Overwrite model.train with this function to make sure train/eval mode
        does not change anymore."""
        return self

    def forward_features(self, x):
        # B = x.shape[0]
        B = paddle.shape(x)[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand((B, -1, -1))
        x = paddle.concat((cls_tokens, x), axis=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        if self.mp_degree > 1:
            with get_rng_state_tracker().rng_state("global_seed"):
                x = self.pos_drop(x)
        else:
            x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if hasattr(self, "rel_pos_bias") else None
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:

                x = recompute(blk, x, rel_pos_bias=rel_pos_bias)
            else:
                x = blk(x, rel_pos_bias=rel_pos_bias)
        # x = self.norm(x)
        return x

    def forward(self, pixel_values):
        x = self.forward_features(pixel_values)
        return x


def interpolate_pos_embed(model, checkpoint_model):
    if "visual_encoder.pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["visual_encoder.pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.visual_encoder.patch_embed.num_patches
        num_extra_tokens = model.visual_encoder.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape((-1, orig_size, orig_size, embedding_size)).transpose((0, 3, 1, 2))
            pos_tokens = paddle.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
            )
            pos_tokens = pos_tokens.transpose((0, 2, 3, 1)).flatten(1, 2)
            new_pos_embed = paddle.concat((extra_tokens, pos_tokens), axis=1)
            checkpoint_model["visual_encoder.pos_embed"] = new_pos_embed
    elif "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = (
            model.visual_encoder.patch_embed.num_patches
            if hasattr(model, "visual_encoder")
            else model.patch_embed.num_patches
        )
        num_extra_tokens = (
            model.visual_encoder.pos_embed.shape[-2] - num_patches
            if hasattr(model, "visual_encoder")
            else model.pos_embed.shape[-2] - num_patches
        )
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape((-1, orig_size, orig_size, embedding_size)).transpose((0, 3, 1, 2))
            pos_tokens = paddle.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
            )
            pos_tokens = pos_tokens.transpose((0, 2, 3, 1)).flatten(1, 2)
            new_pos_embed = paddle.concat((extra_tokens, pos_tokens), axis=1)
            checkpoint_model["pos_embed"] = new_pos_embed
