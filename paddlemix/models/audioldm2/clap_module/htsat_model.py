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
import random
import warnings

import paddle
import paddle.nn as nn

from ..utils import DropPath, Mlp, to_2tuple
from .feature_fusion import AFF, DAF, iAFF
from .utils import (
    LogmelFilterBank,
    SpecAugmentation,
    Spectrogram,
    do_mixup,
    interpolate,
)


class PatchEmbed(nn.Layer):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        patch_stride=16,
        enable_fusion=False,
        fusion_type="None",
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.grid_size = (
            img_size[0] // patch_stride[0],
            img_size[1] // patch_stride[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type

        padding = (
            (patch_size[0] - patch_stride[0]) // 2,
            (patch_size[1] - patch_stride[1]) // 2,
        )

        if (self.enable_fusion) and (self.fusion_type == "channel_map"):
            self.proj = nn.Conv2D(
                in_chans * 4,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_stride,
                padding=padding,
            )
        else:
            self.proj = nn.Conv2D(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_stride,
                padding=padding,
            )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        if (self.enable_fusion) and (self.fusion_type in ["daf_2d", "aff_2d", "iaff_2d"]):
            self.mel_conv2d = nn.Conv2D(
                in_chans,
                embed_dim,
                kernel_size=(patch_size[0], patch_size[1] * 3),
                stride=(patch_stride[0], patch_stride[1] * 3),
                padding=padding,
            )
            if self.fusion_type == "daf_2d":
                self.fusion_model = DAF()
            elif self.fusion_type == "aff_2d":
                self.fusion_model = AFF(channels=embed_dim, type="2D")
            elif self.fusion_type == "iaff_2d":
                self.fusion_model = iAFF(channels=embed_dim, type="2D")

    def forward(self, x, longer_idx=None):
        if (self.enable_fusion) and (self.fusion_type in ["daf_2d", "aff_2d", "iaff_2d"]):
            global_x = x[:, 0:1, :, :]

            # global processing
            B, C, H, W = global_x.shape
            assert (
                H == self.img_size[0] and W == self.img_size[1]
            ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            global_x = self.proj(global_x)
            TW = global_x.shape[-1]
            if len(longer_idx) > 0:
                # local processing
                local_x = x[longer_idx, 1:, :, :]
                B, C, H, W = local_x.shape
                local_x = local_x.reshape([B * C, 1, H, W])
                local_x = self.mel_conv2d(local_x)
                local_x = local_x.reshape([B, C, local_x.shape[1], local_x.shape[2], local_x.shape[3]])
                local_x = local_x.transpose([0, 2, 3, 1, 4]).flatten(3)
                TB, TC, TH, _ = local_x.shape
                if local_x.shape[-1] < TW:
                    local_x = paddle.concat(
                        [
                            local_x,
                            paddle.zeros([TB, TC, TH, TW - local_x.shape[-1]]),
                        ],
                        axis=-1,
                    )
                else:
                    local_x = local_x[:, :, :, :TW]

                global_x[longer_idx] = self.fusion_model(global_x[longer_idx], local_x)
            x = global_x
        else:
            B, C, H, W = x.shape
            assert (
                H == self.img_size[0] and W == self.img_size[1]
            ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            x = self.proj(x)

        if self.flatten:
            x = x.flatten(2).transpose([0, 2, 1])  # BCHW -> BNC
        x = self.norm(x)
        return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with paddle.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor = paddle.multiply(tensor, paddle.to_tensor(std) * math.sqrt(2.0))
        # tensor.mul_(std * math.sqrt(2.0))
        tensor = paddle.add(tensor, paddle.to_tensor(mean))
        # tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clip_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


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
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
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
        relative_position_bias_table = paddle.zeros([(2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads])
        self.relative_position_bias_table = paddle.create_parameter(
            shape=relative_position_bias_table.shape,
            dtype=str(relative_position_bias_table.numpy().dtype),
            default_initializer=nn.initializer.Assign(relative_position_bias_table),
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
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x, mask=None):
        """
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
        )

        q = q * self.scale
        k_perm_shape = list(range(k.dim()))
        k_new_perm_shape = k_perm_shape
        k_new_perm_shape[-1], k_new_perm_shape[-2] = k_perm_shape[-2], k_perm_shape[-1]
        attn = q @ k.transpose(k_new_perm_shape)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape([-1])].reshape(
            [self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1]
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.transpose([2, 0, 1])  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape([B_ // nW, nW, self.num_heads, N, N]) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, N, N])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        tmp = attn @ v
        tmp_perm_shape = list(range(tmp.dim()))
        new_tmp_perm_shape = tmp_perm_shape
        new_tmp_perm_shape[1], new_tmp_perm_shape[2] = tmp_perm_shape[2], tmp_perm_shape[1]
        x = tmp.transpose(new_tmp_perm_shape).reshape([B_, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

    def extra_repr(self):
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"


# We use the model based on SwinTransformer Block, therefore we can use the swin-transformer pretrained model
class SwinTransformerBlock(nn.Layer):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Layer, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
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
        norm_before_mlp="ln",
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm_before_mlp = norm_before_mlp
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
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
        if self.norm_before_mlp == "ln":
            self.norm2 = nn.LayerNorm(dim)
        elif self.norm_before_mlp == "bn":
            self.bn2 = nn.BatchNorm1D(dim)

            def norm2_fun(x):
                perm_shape = list(range(x.dim()))
                new_perm_shape = perm_shape
                new_perm_shape[1], new_perm_shape[2] = perm_shape[2], perm_shape[1]
                return self.bn2(x.transpose(new_perm_shape)).transpose(new_perm_shape)

            self.norm2 = norm2_fun
        else:
            raise NotImplementedError
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = paddle.zeros([1, H, W, 1])  # 1 H W 1
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
            attn_mask = paddle.where(attn_mask != 0, paddle.ones_like(attn_mask) * float(-100.0), attn_mask)
            attn_mask = paddle.where(attn_mask == 0, paddle.ones_like(attn_mask) * float(0.0), attn_mask)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.reshape([B, H, W, C])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = paddle.roll(x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape([-1, self.window_size * self.window_size, C])  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows, attn = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape([-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = paddle.roll(shifted_x, shifts=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x
        x = x.reshape([B, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn

    def extra_repr(self):
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )


class PatchMerging(nn.Layer):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias_attr=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.reshape([B, H, W, C])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = paddle.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.reshape([B, -1, 4 * C])  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Layer):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Layer, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Layer | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        norm_before_mlp="ln",
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.LayerList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
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
                    norm_before_mlp=norm_before_mlp,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        attns = []
        for blk in self.blocks:
            x, attn = blk(x)
            if not self.training:
                attns.append(attn.unsqueeze(0))
        if self.downsample is not None:
            x = self.downsample(x)
        if not self.training:
            attn = paddle.concat(attns, axis=0)
            attn = paddle.mean(attn, axis=0)
        return x, attn

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


# The Core of HTSAT
class HTSAT_Swin_Transformer(nn.Layer):
    r"""HTSAT based on the Swin Transformer
    Args:
        spec_size (int | tuple(int)): Input Spectrogram size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 4
        path_stride (iot | tuple(int)): Patch Stride for Frequency and Time Axis. Default: 4
        in_chans (int): Number of input image channels. Default: 1 (mono)
        num_classes (int): Number of classes for classification head. Default: 527
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each HTSAT-Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Layer): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        config (module): The configuration Module from config.py
    """

    def __init__(
        self,
        spec_size=256,
        patch_size=4,
        patch_stride=(4, 4),
        in_chans=1,
        num_classes=527,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        norm_before_mlp="ln",
        config=None,
        enable_fusion=False,
        fusion_type="None",
        **kwargs,
    ):
        super(HTSAT_Swin_Transformer, self).__init__()

        self.config = config
        self.spec_size = spec_size
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.ape = ape
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = len(self.depths)
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))

        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        self.qkv_bias = qkv_bias
        self.qk_scale = None

        self.patch_norm = patch_norm
        self.norm_layer = norm_layer if self.patch_norm else None
        self.norm_before_mlp = norm_before_mlp
        self.mlp_ratio = mlp_ratio

        self.use_checkpoint = use_checkpoint

        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type

        #  process mel-spec ; used only once
        self.freq_ratio = self.spec_size // self.config.mel_bins
        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.interpolate_ratio = 32  # Downsampled ratio
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=config.window_size,
            hop_length=config.hop_size,
            win_length=config.window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=config.sample_rate,
            n_fft=config.window_size,
            n_mels=config.mel_bins,
            fmin=config.fmin,
            fmax=config.fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )  # 2 2
        self.bn0 = nn.BatchNorm2D(self.config.mel_bins)

        # split spectrogram into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=self.spec_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer,
            patch_stride=patch_stride,
            enable_fusion=self.enable_fusion,
            fusion_type=self.fusion_type,
        )

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            absolute_pos_embed = paddle.zeros([1, num_patches, self.embed_dim])
            self.absolute_pos_embed = paddle.create_parameter(
                shape=absolute_pos_embed.shape,
                dtype=str(absolute_pos_embed.numpy().dtype),
                default_initializer=nn.initializer.Assign(absolute_pos_embed),
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=self.drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in paddle.linspace(0, self.drop_path_rate, sum(self.depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.LayerList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]) : sum(self.depths[: i_layer + 1])],
                norm_layer=self.norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                norm_before_mlp=self.norm_before_mlp,
            )
            self.layers.append(layer)

        self.norm = self.norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1D(1)
        self.maxpool = nn.AdaptiveMaxPool1D(1)

        SF = self.spec_size // (2 ** (len(self.depths) - 1)) // self.patch_stride[0] // self.freq_ratio
        self.tscam_conv = nn.Conv2D(
            in_channels=self.num_features,
            out_channels=self.num_classes,
            kernel_size=(SF, 3),
            padding=(0, 1),
        )
        self.head = nn.Linear(num_classes, num_classes)

        if (self.enable_fusion) and (self.fusion_type in ["daf_1d", "aff_1d", "iaff_1d"]):
            self.mel_conv1d = nn.Sequential(
                nn.Conv1D(64, 64, kernel_size=5, stride=3, padding=2),
                nn.BatchNorm1D(64),
            )
            if self.fusion_type == "daf_1d":
                self.fusion_model = DAF()
            elif self.fusion_type == "aff_1d":
                self.fusion_model = AFF(channels=64, type="1D")
            elif self.fusion_type == "iaff_1d":
                self.fusion_model = iAFF(channels=64, type="1D")

    @paddle.jit.not_to_static
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @paddle.jit.not_to_static
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x, longer_idx=None):
        # A deprecated optimization for using a hierarchical output from different blocks

        frames_num = x.shape[2]
        x = self.patch_embed(x, longer_idx=longer_idx)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers):
            x, attn = layer(x)
        # for x
        x = self.norm(x)
        B, N, C = x.shape
        SF = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]
        ST = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]
        x = x.transpose([0, 2, 1]).reshape([B, C, SF, ST])
        B, C, F, T = x.shape
        # group 2D CNN
        c_freq_bin = F // self.freq_ratio
        x = x.reshape([B, C, F // c_freq_bin, c_freq_bin, T])
        x = x.transpose([0, 1, 3, 2, 4]).reshape([B, C, c_freq_bin, -1])
        # get latent_output
        fine_grained_latent_output = paddle.mean(x, axis=2)
        fine_grained_latent_output = interpolate(
            fine_grained_latent_output.transpose([0, 2, 1]),
            8 * self.patch_stride[1],
        )

        latent_output = self.avgpool(paddle.flatten(x, 2))
        latent_output = paddle.flatten(latent_output, 1)

        # display the attention map, if needed

        x = self.tscam_conv(x)
        x = paddle.flatten(x, 2)  # B, C, T

        fpx = interpolate(nn.functional.sigmoid(x).transpose([0, 2, 1]), 8 * self.patch_stride[1])

        x = self.avgpool(x)
        x = paddle.flatten(x, 1)

        output_dict = {
            "framewise_output": fpx,  # already sigmoided
            "clipwise_output": nn.functional.sigmoid(x),
            "fine_grained_embedding": fine_grained_latent_output,
            "embedding": latent_output,
        }

        return output_dict

    def crop_wav(self, x, crop_size, spe_pos=None):
        time_steps = x.shape[2]
        tx = paddle.zeros([x.shape[0], x.shape[1], crop_size, x.shape[3]])
        for i in range(len(x)):
            if spe_pos is None:
                crop_pos = random.randint(0, time_steps - crop_size - 1)
            else:
                crop_pos = spe_pos
            tx[i][0] = x[i, 0, crop_pos : crop_pos + crop_size, :]
        return tx

    # Reshape the wavform to a img size, if you want to use the pretrained swin transformer model
    def reshape_wav2img(self, x):
        B, C, T, F = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert T <= target_T and F <= target_F, "the wav size should less than or equal to the swin input size"
        # to avoid bicubic zero error
        if T < target_T:
            x = nn.functional.interpolate(x, (target_T, x.shape[3]), mode="bicubic", align_corners=True)
        if F < target_F:
            x = nn.functional.interpolate(x, (x.shape[2], target_F), mode="bicubic", align_corners=True)
        x = x.transpose([0, 1, 3, 2])
        x = x.reshape([x.shape[0], x.shape[1], x.shape[2], self.freq_ratio, x.shape[3] // self.freq_ratio])
        # print(x.shape)
        x = x.transpose([0, 1, 3, 2, 4])
        x = x.reshape([x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4]])
        return x

    # Repeat the wavform to a img size, if you want to use the pretrained swin transformer model
    def repeat_wat2img(self, x, cur_pos):
        B, C, T, F = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert T <= target_T and F <= target_F, "the wav size should less than or equal to the swin input size"
        # to avoid bicubic zero error
        if T < target_T:
            x = nn.functional.interpolate(x, (target_T, x.shape[3]), mode="bicubic", align_corners=True)
        if F < target_F:
            x = nn.functional.interpolate(x, (x.shape[2], target_F), mode="bicubic", align_corners=True)
        x = x.transpose([0, 1, 3, 2])  # B C F T
        x = x[:, :, :, cur_pos : cur_pos + self.spec_size]
        # x = x.repeat_interleave(repeats=(1, 1, 4, 1))
        x = x.repeat_interleave(repeats=4, axis=2)
        return x

    def forward(
        self, x: paddle.Tensor, mixup_lambda=None, infer_mode=False, device=None
    ):  # out_feat_keys: List[str] = None):
        if self.enable_fusion and x["longer"].sum() == 0:
            # if no audio is longer than 10s, then randomly select one audio to be longer
            x["longer"][paddle.randint(0, x["longer"].shape[0], (1,))] = True

        if not self.enable_fusion:
            x = x["waveform"]
            x = self.spectrogram_extractor(x)  # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
            x = x.transpose([0, 3, 2, 1])
            x = self.bn0(x)
            x = x.transpose([0, 3, 2, 1])
            if self.training:
                x = self.spec_augmenter(x)

            if self.training and mixup_lambda is not None:
                x = do_mixup(x, mixup_lambda)

            x = self.reshape_wav2img(x)
            output_dict = self.forward_features(x)
        else:
            longer_list = x["longer"]
            x = x["mel_fusion"]
            x = x.transpose([0, 3, 2, 1])
            x = self.bn0(x)
            x = x.transpose([0, 3, 2, 1])
            longer_list_idx = paddle.where(longer_list)[0].squeeze()
            if self.fusion_type in ["daf_1d", "aff_1d", "iaff_1d"]:
                new_x = x[:, 0:1, :, :].clone()
                if len(longer_list_idx) > 0:
                    # local processing
                    fusion_x_local = x[longer_list_idx, 1:, :, :].clone()
                    FB, FC, FT, FF = fusion_x_local.shape
                    fusion_x_local = fusion_x_local.reshape([FB * FC, FT, FF])
                    fusion_x_local = paddle.transpose(fusion_x_local, (0, 2, 1))
                    fusion_x_local = self.mel_conv1d(fusion_x_local)
                    fusion_x_local = fusion_x_local.reshape(FB, FC, FF, fusion_x_local.shape[-1])
                    fusion_x_local = paddle.transpose(fusion_x_local, (0, 2, 1, 3)).flatten(2)
                    if fusion_x_local.shape[-1] < FT:
                        fusion_x_local = paddle.concat(
                            [
                                fusion_x_local,
                                paddle.zeros((FB, FF, FT - fusion_x_local.size(-1))),
                            ],
                            axis=-1,
                        )
                    else:
                        fusion_x_local = fusion_x_local[:, :, :FT]
                    # 1D fusion
                    new_x = new_x.squeeze(1).transpose((0, 2, 1))
                    new_x[longer_list_idx] = self.fusion_model(new_x[longer_list_idx], fusion_x_local)
                    x = new_x.transpose((0, 2, 1))[:, None, :, :]
                else:
                    x = new_x

            elif self.fusion_type in ["daf_2d", "aff_2d", "iaff_2d", "channel_map"]:
                x = x  # no change

            if self.training:
                x = self.spec_augmenter(x)
            if self.training and mixup_lambda is not None:
                x = do_mixup(x, mixup_lambda)

            x = self.reshape_wav2img(x)
            output_dict = self.forward_features(x, longer_idx=longer_list_idx)

        return output_dict


def create_htsat_model(audio_cfg, enable_fusion=False, fusion_type="None"):
    try:
        assert audio_cfg.model_name in [
            "base",
        ], "model name for HTS-AT is wrong!"
        if audio_cfg.model_name == "base":
            model = HTSAT_Swin_Transformer(
                spec_size=256,
                patch_size=4,
                patch_stride=(4, 4),
                num_classes=audio_cfg.class_num,
                embed_dim=128,
                depths=[2, 2, 12, 2],
                num_heads=[4, 8, 16, 32],
                window_size=8,
                config=audio_cfg,
                enable_fusion=enable_fusion,
                fusion_type=fusion_type,
            )

        return model
    except:
        raise RuntimeError(
            f"Import Model for {audio_cfg.model_name} not found, or the audio cfg parameters are not enough."
        )
