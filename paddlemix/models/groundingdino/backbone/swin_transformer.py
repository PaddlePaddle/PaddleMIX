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

import os
from typing import Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed.fleet.utils import recompute
from paddle.nn.initializer import Constant

from paddlemix.utils.log import logger

from ..layers import DropPath, to_2tuple

trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)

from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.transformers.model_utils import register_base_model

from paddlemix.models.model_utils import MixPretrainedModel

""" swin_transformer model configuration"""
__all__ = ["SwinTransformerConfig"]


class SwinTransformerConfig(PretrainedConfig):

    model_type = "swintransformer"

    def __init__(
        self,
        in_chans=3,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        pretrain_img_size=224,
        patch_size=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        dilation=False,
        use_checkpoint=False,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.pretrain_img_size = pretrain_img_size
        self.patch_size = 4
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.dilation = dilation
        self.use_checkpoint = use_checkpoint

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class SwinTransformerPretrainedModel(MixPretrainedModel):
    """
    See :class:`paddlemix.models.model_utils.MixPretrainedModel` for more details.
    """

    model_config_file = "config.json"
    config_class = SwinTransformerConfig
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "swintransformer"


class Mlp(nn.Layer):
    """Multilayer perceptron."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape([B, H // window_size, window_size, W // window_size, window_size, C])
    windows = x.transpose([0, 1, 3, 2, 4, 5]).reshape([-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / paddle.to_tensor(window_size * window_size)))
    x = windows.reshape([B, H // window_size, W // window_size, window_size, window_size, -1])
    x = x.transpose([0, 1, 3, 2, 4, 5]).reshape([B, H, W, -1])
    return x


class WindowAttention(nn.Layer):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = self.create_parameter(
            shape=[(2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads],
            dtype=paddle.float32,
            default_initializer=Constant(0.0),
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = paddle.arange(self.window_size[0])
        coords_w = paddle.arange(self.window_size[1])
        coords = paddle.stack(paddle.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose([1, 2, 0])  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x, mask=None):
        """Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape([B_, N, 3, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = paddle.mm(q, k.transpose([0, 1, 3, 2]))
        index = self.relative_position_index.flatten()

        relative_position_bias = paddle.index_select(self.relative_position_bias_table, index)

        relative_position_bias = relative_position_bias.reshape(
            [
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            ]
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.transpose([2, 0, 1])  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape([-1, nW, self.num_heads, N, N]) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, N, N])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = paddle.mm(attn, v).transpose([0, 2, 1, 3]).reshape([-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Layer):
    """Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.reshape([B, H, W, C])

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_list = paddle.zeros([4], dtype="int32")
        pad_list[1] = pad_r
        pad_list[3] = pad_b
        x = F.pad(x, pad_list, data_format="NHWC")
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = paddle.roll(x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, C])  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = paddle.roll(shifted_x, shifts=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x

        x = x[:, :H, :W, :]

        x = x.reshape([B, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Layer):
    """Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias_attr=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.reshape([B, H, W, C])

        # padding
        pad_list = paddle.zeros([4], dtype="int32")
        pad_list[3] = H % 2
        pad_list[1] = W % 2
        x = F.pad(x, pad_list, data_format="NHWC")

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = paddle.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.reshape([B, -1, 4 * C])  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Layer):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.LayerList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = (H + self.window_size - 1) // self.window_size * self.window_size
        Wp = (W + self.window_size - 1) // self.window_size * self.window_size
        img_mask = paddle.zeros((1, Hp, Wp, 1), dtype=paddle.float32)  # 1 Hp Wp 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.reshape([-1, self.window_size * self.window_size])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = -100.0 * paddle.ones_like(attn_mask) * (attn_mask != 0).astype(paddle.float32)

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = recompute(blk, x, attn_mask, **{"preserve_rng_state": True})
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Layer):
    """Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.shape
        pad_list = paddle.zeros([4], dtype="int32")
        pad_list[1] = self.patch_size[1] - W % self.patch_size[1]
        pad_list[3] = self.patch_size[0] - H % self.patch_size[0]
        x = F.pad(x, pad_list)

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.shape[2:]
            x = x.flatten(2).transpose([0, 2, 1])
            x = self.norm(x)
            x = x.transpose([0, 2, 1]).reshape([-1, self.embed_dim, Wh, Ww])

        return x


@register_base_model
class SwinTransformerModel(SwinTransformerPretrainedModel):
    """Swin Transformer backbone.
    A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
      https://arxiv.org/pdf/2103.14030
    """

    def __init__(self, config: SwinTransformerConfig):
        super(SwinTransformerModel, self).__init__(config)

        self.pretrain_img_size = config.pretrain_img_size
        self.num_layers = len(config.depths)
        self.in_chans = config.in_chans
        self.embed_dim = config.embed_dim
        self.ape = config.ape
        self.patch_norm = config.patch_norm
        self.norm_layer = nn.LayerNorm
        self.out_indices = config.out_indices
        self.frozen_stages = config.frozen_stages
        self.dilation = config.dilation
        self.patch_size = config.patch_size
        self.patch_norm = config.patch_norm
        self.drop_path_rate = config.drop_path_rate

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer if self.patch_norm else None,
        )

        # absolute position embedding
        if self.ape:
            # patch_size = to_2tuple(self.patch_size)
            patches_resolution = [
                self.pretrain_img_size[0] // self.patch_size[0],
                self.pretrain_img_size[1] // self.patch_size[1],
            ]

            self.absolute_pos_embed = self.create_parameter(
                shape=[1, self.embed_dim, patches_resolution[0], patches_resolution[1]],
                dtype=paddle.float32,
                default_initializer=Constant(0.0),
            )
            trunc_normal_(self.absolute_pos_embed)

        self.pos_drop = nn.Dropout(p=config.drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in paddle.linspace(0, config.drop_path_rate, sum(config.depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.LayerList()
        # prepare downsample list
        downsamplelist = [PatchMerging for i in range(self.num_layers)]
        downsamplelist[-1] = None
        num_features = [int(self.embed_dim * 2**i) for i in range(self.num_layers)]
        if self.dilation:
            downsamplelist[-2] = None
            num_features[-1] = int(self.embed_dim * 2 ** (self.num_layers - 1)) // 2
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=num_features[i_layer],
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                window_size=config.window_size,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                qk_scale=config.qk_scale,
                drop=config.drop_rate,
                attn_drop=config.attn_drop_rate,
                drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                norm_layer=self.norm_layer,
                downsample=downsamplelist[i_layer],
                use_checkpoint=config.use_checkpoint,
            )
            self.layers.append(layer)

        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in self.out_indices:
            layer = self.norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_sublayer(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.stop_gradient = True

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.stop_gradient = True

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.stop_gradient = True

    def forward_raw(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.shape[2:4]
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic")
            x = (x + absolute_pos_embed).flatten(2).transpose([0, 2, 1])  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose([0, 2, 1])
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                out = x_out.reshape((-1, H, W, self.num_features[i])).transpose((0, 3, 1, 2))
                outs.append(out)
        # in:
        #   torch.Size([2, 3, 1024, 1024])
        # outs:
        #   [torch.Size([2, 192, 256, 256]), torch.Size([2, 384, 128, 128]), \
        #       torch.Size([2, 768, 64, 64]), torch.Size([2, 1536, 32, 32])]
        return tuple(outs)

    def forward_with_mask(self, x: paddle.Tensor, m: paddle.Tensor):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.shape[2], x.shape[3]
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode="bicubic")
            x = (x + absolute_pos_embed).flatten(2).transpose([0, 2, 1])  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose([0, 2, 1])
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                out = x_out.reshape((-1, H, W, self.num_features[i])).transpose((0, 3, 1, 2))
                outs.append(out)

        feat_dict = []
        mask_dict = []
        for idx, out_i in enumerate(outs):
            assert m is not None
            mask = F.interpolate(m[None].cast(paddle.float32), size=out_i.shape[-2:]).cast(paddle.bool)[0]
            feat_dict.append(out_i)
            mask_dict.append(mask)

        return feat_dict, mask_dict

    def forward(self, x: paddle.Tensor, m=None):
        if m is not None:
            return self.forward_with_mask(x, m)
        else:
            return self.forward_raw(x)
