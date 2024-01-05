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

import numpy as np
import paddle


def silu(x):
    return x * paddle.nn.functional.sigmoid(x=x)


class SiLU(paddle.nn.Layer):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = paddle.mean(x=paddle.nn.functional.relu(x=1.0 - logits_real))
    loss_fake = paddle.mean(x=paddle.nn.functional.relu(x=1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        paddle.mean(x=paddle.nn.functional.softplus(x=-logits_real))
        + paddle.mean(x=paddle.nn.functional.softplus(x=logits_fake))
    )
    return d_loss


def Normalize(in_channels, norm_type="group"):
    assert norm_type in ["group", "batch"]
    if norm_type == "group":
        return paddle.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, epsilon=1e-06, weight_attr=None, bias_attr=None
        )
    elif norm_type == "batch":
        return paddle.nn.SyncBatchNorm(in_channels)


class ResBlock(paddle.nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        norm_type="group",
        padding_type="replicate",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.norm2 = Normalize(in_channels, norm_type)
        self.conv2 = SamePadConv3d(out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)
        return x + h


class SamePadConv3d(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type="replicate"):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3
        total_pad = tuple([(k - s) for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type
        self.conv = paddle.nn.Conv3D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias_attr=bias,
        )
        self.weight = self.conv.weight

    def forward(self, x):
        return self.conv(
            paddle.nn.functional.pad(x=x, pad=self.pad_input, mode=self.padding_type, data_format="NCDHW")
        )


class SamePadConvTranspose3d(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type="replicate"):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3
        total_pad = tuple([(k - s) for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]:
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type
        self.convt = paddle.nn.Conv3DTranspose(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=tuple([(k - 1) for k in kernel_size]),
            bias_attr=bias,
        )

    def forward(self, x):
        return self.convt(
            paddle.nn.functional.pad(x=x, pad=self.pad_input, mode=self.padding_type, data_format="NCDHW")
        )


class Encoder(paddle.nn.Layer):
    def __init__(
        self, n_hiddens, downsample, z_channels, double_z, image_channel=3, norm_type="group", padding_type="replicate"
    ):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = paddle.nn.LayerList()
        max_ds = n_times_downsample.max()
        self.conv_first = SamePadConv3d(image_channel, n_hiddens, kernel_size=3, padding_type=padding_type)
        for i in range(max_ds):
            block = paddle.nn.Layer()
            in_channels = n_hiddens * 2**i
            out_channels = n_hiddens * 2 ** (i + 1)
            stride = tuple([(2 if d > 0 else 1) for d in n_times_downsample])
            block.down = SamePadConv3d(in_channels, out_channels, 4, stride=stride, padding_type=padding_type)
            block.res = ResBlock(out_channels, out_channels, norm_type=norm_type)
            self.conv_blocks.append(block)
            n_times_downsample -= 1
        self.final_block = paddle.nn.Sequential(
            Normalize(out_channels, norm_type),
            SiLU(),
            SamePadConv3d(
                out_channels,
                2 * z_channels if double_z else z_channels,
                kernel_size=3,
                stride=1,
                padding_type=padding_type,
            ),
        )
        self.out_channels = out_channels

    def forward(self, x):
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)
        return h


class Decoder(paddle.nn.Layer):
    def __init__(self, n_hiddens, upsample, z_channels, image_channel, norm_type="group"):
        super().__init__()
        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        in_channels = z_channels
        self.conv_blocks = paddle.nn.LayerList()
        for i in range(max_us):
            block = paddle.nn.Layer()
            in_channels = in_channels if i == 0 else n_hiddens * 2 ** (max_us - i + 1)
            out_channels = n_hiddens * 2 ** (max_us - i)
            us = tuple([(2 if d > 0 else 1) for d in n_times_upsample])
            block.up = SamePadConvTranspose3d(in_channels, out_channels, 4, stride=us)
            block.res1 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            block.res2 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            self.conv_blocks.append(block)
            n_times_upsample -= 1
        self.conv_out = SamePadConv3d(out_channels, image_channel, kernel_size=3)

    def forward(self, x):
        h = x
        for i, block in enumerate(self.conv_blocks):
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.conv_out(h)
        return h
