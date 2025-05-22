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
from typing import Tuple

import paddle
from sam2.modeling.sam2_utils import DropPath, LayerNorm2d, get_clones


class MaskDownSampler(paddle.nn.Layer):
    """
    Progressively downsample a mask by total_stride, each time by stride.
    Note that LayerNorm is applied per *token*, like in ViT.

    With each downsample (by a factor stride**2), channel capacity increases by the same factor.
    In the end, we linearly project to embed_dim channels.
    """

    def __init__(self, embed_dim=256, kernel_size=4, stride=4, padding=0, total_stride=16, activation=paddle.nn.GELU):
        super().__init__()
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride**num_layers == total_stride
        self.encoder = paddle.nn.LayerList()
        mask_in_chans, mask_out_chans = 1, 1
        for _ in range(num_layers):
            mask_out_chans = mask_in_chans * stride**2
            self.encoder.append(
                paddle.nn.Conv2D(
                    in_channels=mask_in_chans,
                    out_channels=mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.encoder.append(LayerNorm2d(mask_out_chans))
            self.encoder.append((activation()))
            mask_in_chans = mask_out_chans
        self.encoder.append(paddle.nn.Conv2D(in_channels=mask_out_chans, out_channels=embed_dim, kernel_size=1))

    def forward(self, x):
        for fn in self.encoder:
            x = fn(x)
        return x


class CXBlock(paddle.nn.Layer):
    """ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, kernel_size=7, padding=3, drop_path=0.0, layer_scale_init_value=1e-06, use_dwconv=True):
        super().__init__()
        self.dwconv = paddle.nn.Conv2D(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim if use_dwconv else 1,
        )
        self.norm = LayerNorm2d(dim, eps=1e-06)
        self.pwconv1 = paddle.nn.Linear(in_features=dim, out_features=4 * dim)
        self.act = paddle.nn.GELU()
        self.pwconv2 = paddle.nn.Linear(in_features=4 * dim, out_features=dim)
        out_18 = paddle.create_parameter(
            shape=(layer_scale_init_value * paddle.ones(shape=dim)).shape,
            dtype=(layer_scale_init_value * paddle.ones(shape=dim)).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(layer_scale_init_value * paddle.ones(shape=dim)),
        )
        out_18.stop_gradient = not True
        self.gamma = out_18 if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.transpose(perm=[0, 2, 3, 1])
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(perm=[0, 3, 1, 2])
        x = input + self.drop_path(x)
        return x


class Fuser(paddle.nn.Layer):
    def __init__(self, layer, num_layers, dim=None, input_projection=False):
        super().__init__()
        self.proj = paddle.nn.Identity()
        self.layers = get_clones(layer, num_layers)
        if input_projection:
            assert dim is not None
            self.proj = paddle.nn.Conv2D(in_channels=dim, out_channels=dim, kernel_size=1)

    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class MemoryEncoder(paddle.nn.Layer):
    def __init__(self, out_dim, mask_downsampler, fuser, position_encoding, in_dim=256):
        super().__init__()
        self.mask_downsampler = mask_downsampler
        self.pix_feat_proj = paddle.nn.Conv2D(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.fuser = fuser
        self.position_encoding = position_encoding
        self.out_proj = paddle.nn.Identity()
        if out_dim != in_dim:
            self.out_proj = paddle.nn.Conv2D(in_channels=in_dim, out_channels=out_dim, kernel_size=1)

    def forward(
        self, pix_feat: paddle.Tensor, masks: paddle.Tensor, skip_mask_sigmoid: bool = False
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        if not skip_mask_sigmoid:
            masks = paddle.nn.functional.sigmoid(x=masks)
        masks = self.mask_downsampler(masks)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)
        pos = self.position_encoding(x).to(x.dtype)
        return {"vision_features": x, "vision_pos_enc": [pos]}
