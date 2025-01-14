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

from typing import Tuple, Union

import paddle
from einops import rearrange, repeat


def nonlinearity(x):
    return x * paddle.nn.functional.sigmoid(x=x)


def Normalize(in_channels):
    return paddle.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, epsilon=1e-06, weight_attr=True, bias_attr=True
    )


class Upsample(paddle.nn.Layer):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = paddle.nn.Conv2D(
                in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = paddle.nn.functional.interpolate(x=x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


# class DepthToSpaceUpsample(paddle.nn.Layer):

#     def __init__(self, in_channels):
#         super().__init__()
#         conv = paddle.nn.Conv2D(in_channels=in_channels, out_channels=
#             in_channels * 4, kernel_size=1)
#         self.net = paddle.nn.Sequential(conv, paddle.nn.Silu(), Rearrange(
#             'b (c p1 p2) h w -> b c (h p1) (w p2)', p1=2, p2=2))
#         self.init_conv_(conv)

#     def init_conv_(self, conv):
#         o, i, h, w = tuple(conv.weight.shape)
#         conv_weight = paddle.empty(shape=[o // 4, i, h, w])
#         init_KaimingUniform = paddle.nn.initializer.KaimingUniform(nonlinearity
#             ='leaky_relu')
#         init_KaimingUniform(conv_weight)
#         conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')
#         paddle.assign(conv_weight, output=conv.weight.data)
#         init_Constant = paddle.nn.initializer.Constant(value=0.0)
#         init_Constant(conv.bias.data)

#     def forward(self, x):
#         out = self.net(x)
#         return out


class Downsample(paddle.nn.Layer):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = paddle.nn.Conv2D(
                in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = 0, 1, 0, 1
            x = paddle.nn.functional.pad(x=x, pad=pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = paddle.nn.functional.avg_pool2d(kernel_size=2, stride=2, x=x, exclusive=False)
        return x


def unpack_time(t, batch):
    _, c, w, h = tuple(t.shape)
    out = paddle.reshape(x=t, shape=[batch, -1, c, w, h])
    out = rearrange(out, "b t c h w -> b c t h w")
    return out


def pack_time(t):
    out = rearrange(t, "b c t h w -> b t c h w")
    _, _, c, w, h = tuple(out.shape)
    return paddle.reshape(x=out, shape=[-1, c, w, h])


class TimeDownsample2x(paddle.nn.Layer):
    def __init__(self, dim, dim_out=None, kernel_size=3):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        self.time_causal_padding = kernel_size - 1, 0
        self.conv = paddle.nn.Conv1D(in_channels=dim, out_channels=dim_out, kernel_size=kernel_size, stride=2)

    def forward(self, x):
        x = rearrange(x, "b c t h w -> b h w c t")
        b, h, w, c, t = tuple(x.shape)
        x = paddle.reshape(x=x, shape=[-1, c, t])
        x = paddle.nn.functional.pad(x=x, pad=self.time_causal_padding)
        out = self.conv(x)
        out = paddle.reshape(x=out, shape=[b, h, w, c, t])
        out = rearrange(out, "b h w c t -> b c t h w")
        out = rearrange(out, "b h w c t -> b c t h w")
        return out


class TimeUpsample2x(paddle.nn.Layer):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        conv = paddle.nn.Conv1D(in_channels=dim, out_channels=dim_out * 2, kernel_size=1)
        self.net = paddle.nn.Sequential(
            paddle.nn.Silu(),
            conv,
            # Rearrange('b (c p) t -> b c (t p)', p=2),
            rearrange("b (c p) t -> b c (t p)", p=2),
        )
        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, t = tuple(conv.weight.shape)
        conv_weight = paddle.empty(shape=[o // 2, i, t])
        init_KaimingUniform = paddle.nn.initializer.KaimingUniform(nonlinearity="leaky_relu")
        init_KaimingUniform(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 2) ...")
        paddle.assign(conv_weight, output=conv.weight.data)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(conv.bias.data)

    def forward(self, x):
        x = rearrange(x, "b c t h w -> b h w c t")
        b, h, w, c, t = tuple(x.shape)
        x = paddle.reshape(x=x, shape=[-1, c, t])
        out = self.net(x)
        out = out[:, :, 1:].contiguous()
        out = paddle.reshape(x=out, shape=[b, h, w, c, t])
        out = rearrange(out, "b h w c t -> b c t h w")
        return out


class AttnBlock(paddle.nn.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = tuple(q.shape)
        q = q.reshape([b, c, h * w])
        q = q.transpose(perm=[0, 2, 1])
        k = k.reshape([b, c, h * w])
        w_ = paddle.bmm(x=q, y=k)
        w_ = w_ * int(c) ** -0.5
        w_ = paddle.nn.functional.softmax(x=w_, axis=2)
        v = v.reshape([b, c, h * w])
        w_ = w_.transpose(perm=[0, 2, 1])
        h_ = paddle.bmm(x=v, y=w_)
        h_ = h_.reshape([b, c, h, w])
        h_ = self.proj_out(h_)
        return x + h_


class TimeAttention(AttnBlock):
    def forward(self, x, *args, **kwargs):
        x = rearrange(x, "b c t h w -> b h w t c")
        b, h, w, t, c = tuple(x.shape)
        x = paddle.reshape(x=x, shape=(-1, t, c))
        x = super().forward(x, *args, **kwargs)
        x = paddle.reshape(x=x, shape=[b, h, w, t, c])
        return rearrange(x, "b h w t c -> b c t h w")


class Residual(paddle.nn.Layer):
    def __init__(self, fn: paddle.nn.Layer):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else (t,) * length


class CausalConv3d(paddle.nn.Layer):
    def __init__(
        self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], pad_mode="constant", **kwargs
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)
        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)
        stride = stride, 1, 1
        dilation = dilation, 1, 1
        self.conv = paddle.nn.Conv3D(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        pad_mode = self.pad_mode if self.time_pad < tuple(x.shape)[2] else "constant"
        x = paddle.nn.functional.pad(x=x, pad=self.time_causal_padding, mode=pad_mode)
        return self.conv(x)


def ResnetBlockCausal3D(dim, kernel_size: Union[int, Tuple[int, int, int]], pad_mode: str = "constant"):
    net = paddle.nn.Sequential(
        Normalize(dim),
        paddle.nn.Silu(),
        CausalConv3d(dim, dim, kernel_size, pad_mode),
        Normalize(dim),
        paddle.nn.Silu(),
        CausalConv3d(dim, dim, kernel_size, pad_mode),
    )
    return Residual(net)


class ResnetBlock(paddle.nn.Layer):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = paddle.nn.Linear(in_features=temb_channels, out_features=out_channels)
        else:
            self.temb_proj = None
        self.norm2 = Normalize(out_channels)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.conv2 = paddle.nn.Conv2D(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = paddle.nn.Conv2D(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = paddle.nn.Conv2D(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h
