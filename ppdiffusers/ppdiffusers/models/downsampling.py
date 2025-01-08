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

from typing import Optional, Tuple

import paddle

from .normalization import RMSNorm
from .upsampling import upfirdn2d_native


class Downsample1D(paddle.nn.Layer):
    """A 1D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 1D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        if use_conv:
            self.conv = paddle.nn.Conv1D(
                in_channels=self.channels,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            self.conv = paddle.nn.AvgPool1D(kernel_size=stride, stride=stride, exclusive=False)

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        assert tuple(inputs.shape)[1] == self.channels
        return self.conv(inputs)


class Downsample2D(paddle.nn.Layer):
    """A 2D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        if norm_type == "ln_norm":
            self.norm = paddle.nn.LayerNorm(
                normalized_shape=channels, epsilon=eps, weight_attr=elementwise_affine, bias_attr=elementwise_affine
            )
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")
        if use_conv:
            conv = paddle.nn.Conv2D(
                in_channels=self.channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias_attr=bias,
            )
        else:
            assert self.channels == self.out_channels
            conv = paddle.nn.AvgPool2D(kernel_size=stride, stride=stride, exclusive=False)
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: paddle.Tensor, *args, **kwargs) -> paddle.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            print("scale", "1.0.0", deprecation_message)
        assert tuple(hidden_states.shape)[1] == self.channels
        if self.norm is not None:
            hidden_states = self.norm(hidden_states.transpose(perm=[0, 2, 3, 1])).transpose(perm=[0, 3, 1, 2])
        if self.use_conv and self.padding == 0:
            pad = 0, 1, 0, 1
            hidden_states = paddle.nn.functional.pad(
                x=hidden_states, pad=pad, mode="constant", value=0, pad_from_left_axis=False
            )
        assert tuple(hidden_states.shape)[1] == self.channels
        hidden_states = self.conv(hidden_states)
        return hidden_states


class FirDownsample2D(paddle.nn.Layer):
    """A 2D FIR downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        fir_kernel (`tuple`, default `(1, 3, 3, 1)`):
            kernel for the FIR filter.
    """

    def __init__(
        self,
        channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        fir_kernel: Tuple[int, int, int, int] = (1, 3, 3, 1),
    ):
        super().__init__()
        out_channels = out_channels if out_channels else channels
        if use_conv:
            self.Conv2d_0 = paddle.nn.Conv2D(
                in_channels=channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
            )
        self.fir_kernel = fir_kernel
        self.use_conv = use_conv
        self.out_channels = out_channels

    def _downsample_2d(
        self,
        hidden_states: paddle.Tensor,
        weight: Optional[paddle.Tensor] = None,
        kernel: Optional[paddle.Tensor] = None,
        factor: int = 2,
        gain: float = 1,
    ) -> paddle.Tensor:
        """Fused `Conv2d()` followed by `downsample_2d()`.
        Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
        efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of
        arbitrary order.

        Args:
            hidden_states (`torch.Tensor`):
                Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
            weight (`torch.Tensor`, *optional*):
                Weight tensor of the shape `[filterH, filterW, inChannels, outChannels]`. Grouped convolution can be
                performed by `inChannels = x.shape[0] // numGroups`.
            kernel (`torch.Tensor`, *optional*):
                FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
                corresponds to average pooling.
            factor (`int`, *optional*, default to `2`):
                Integer downsampling factor.
            gain (`float`, *optional*, default to `1.0`):
                Scaling factor for signal magnitude.

        Returns:
            output (`torch.Tensor`):
                Tensor of the shape `[N, C, H // factor, W // factor]` or `[N, H // factor, W // factor, C]`, and same
                datatype as `x`.
        """
        assert isinstance(factor, int) and factor >= 1
        if kernel is None:
            kernel = [1] * factor
        kernel = paddle.to_tensor(data=kernel, dtype="float32")
        if kernel.ndim == 1:
            kernel = paddle.outer(x=kernel, y=kernel)
        kernel /= paddle.sum(x=kernel)
        kernel = kernel * gain
        if self.use_conv:
            _, _, convH, convW = tuple(weight.shape)
            pad_value = tuple(kernel.shape)[0] - factor + (convW - 1)
            stride_value = [factor, factor]
            upfirdn_input = upfirdn2d_native(
                hidden_states,
                paddle.to_tensor(data=kernel, place=hidden_states.place),
                pad=((pad_value + 1) // 2, pad_value // 2),
            )
            output = paddle.nn.functional.conv2d(x=upfirdn_input, weight=weight, stride=stride_value, padding=0)
        else:
            pad_value = tuple(kernel.shape)[0] - factor
            output = upfirdn2d_native(
                hidden_states,
                paddle.to_tensor(data=kernel, place=hidden_states.place),
                down=factor,
                pad=((pad_value + 1) // 2, pad_value // 2),
            )
        return output

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        if self.use_conv:
            downsample_input = self._downsample_2d(hidden_states, weight=self.Conv2d_0.weight, kernel=self.fir_kernel)
            hidden_states = downsample_input + self.Conv2d_0.bias.reshape(1, -1, 1, 1)
        else:
            hidden_states = self._downsample_2d(hidden_states, kernel=self.fir_kernel, factor=2)
        return hidden_states


class KDownsample2D(paddle.nn.Layer):
    """A 2D K-downsampling layer.

    Parameters:
        pad_mode (`str`, *optional*, default to `"reflect"`): the padding mode to use.
    """

    def __init__(self, pad_mode: str = "reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = paddle.to_tensor(data=[[1 / 8, 3 / 8, 3 / 8, 1 / 8]])
        self.pad = tuple(kernel_1d.shape)[1] // 2 - 1
        self.register_buffer(name="kernel", tensor=kernel_1d.T @ kernel_1d, persistable=False)

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        inputs = paddle.nn.functional.pad(x=inputs, pad=(self.pad,) * 4, mode=self.pad_mode, pad_from_left_axis=False)
        weight = paddle.zeros(
            shape=[
                tuple(inputs.shape)[1],
                tuple(inputs.shape)[1],
                tuple(self.kernel.shape)[0],
                tuple(self.kernel.shape)[1],
            ],
            dtype=inputs.dtype,
        )
        indices = paddle.arange(end=tuple(inputs.shape)[1])
        kernel = self.kernel.to(weight)[None, :].expand(shape=[tuple(inputs.shape)[1], -1, -1])
        weight[indices, indices] = kernel
        return paddle.nn.functional.conv2d(x=inputs, weight=weight, stride=2)


class CogVideoXDownsample3D(paddle.nn.Layer):
    """
    A 3D Downsampling layer using in [CogVideoX]() by Tsinghua University & ZhipuAI

    Args:
        in_channels (`int`):
            Number of channels in the input image.
        out_channels (`int`):
            Number of channels produced by the convolution.
        kernel_size (`int`, defaults to `3`):
            Size of the convolving kernel.
        stride (`int`, defaults to `2`):
            Stride of the convolution.
        padding (`int`, defaults to `0`):
            Padding added to all four sides of the input.
        compress_time (`bool`, defaults to `False`):
            Whether or not to compress the time dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
        compress_time: bool = False,
    ):
        super().__init__()
        self.conv = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.compress_time = compress_time

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.compress_time:
            batch_size, channels, frames, height, width = tuple(x.shape)
            x = x.transpose(perm=[0, 3, 4, 1, 2]).reshape([batch_size * height * width, channels, frames])
            if tuple(x.shape)[-1] % 2 == 1:
                x_first, x_rest = x[..., 0], x[..., 1:]
                if tuple(x_rest.shape)[-1] > 0:
                    x_rest = paddle.nn.functional.avg_pool1d(kernel_size=2, stride=2, x=x_rest, exclusive=False)
                x = paddle.concat(x=[x_first[..., None], x_rest], axis=-1)
                x = x.reshape([batch_size, height, width, channels, tuple(x.shape)[-1]]).transpose(
                    perm=[0, 3, 4, 1, 2]
                )
            else:
                x = paddle.nn.functional.avg_pool1d(kernel_size=2, stride=2, x=x, exclusive=False)
                x = x.reshape([batch_size, height, width, channels, tuple(x.shape)[-1]]).transpose(
                    perm=[0, 3, 4, 1, 2]
                )
        pad = (0, 1, 0, 1, 0, 0)
        x = paddle.nn.functional.pad(x=x, pad=pad, mode="constant", value=0, data_format="NCDHW")
        batch_size, channels, frames, height, width = tuple(x.shape)
        x = x.transpose(perm=[0, 2, 1, 3, 4]).reshape([batch_size * frames, channels, height, width])
        x = self.conv(x)
        x = x.reshape([batch_size, frames, tuple(x.shape)[1], tuple(x.shape)[2], tuple(x.shape)[3]]).transpose(
            perm=[0, 2, 1, 3, 4]
        )
        return x


def downsample_2d(
    hidden_states: paddle.Tensor, kernel: Optional[paddle.Tensor] = None, factor: int = 2, gain: float = 1
) -> paddle.Tensor:
    """Downsample2D a batch of 2D images with the given filter.
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and downsamples each image with the
    given filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the
    specified `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its
    shape is a multiple of the downsampling factor.

    Args:
        hidden_states (`torch.Tensor`)
            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        kernel (`torch.Tensor`, *optional*):
            FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
            corresponds to average pooling.
        factor (`int`, *optional*, default to `2`):
            Integer downsampling factor.
        gain (`float`, *optional*, default to `1.0`):
            Scaling factor for signal magnitude.

    Returns:
        output (`torch.Tensor`):
            Tensor of the shape `[N, C, H // factor, W // factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor
    kernel = paddle.to_tensor(data=kernel, dtype="float32")
    if kernel.ndim == 1:
        kernel = paddle.outer(x=kernel, y=kernel)
    kernel /= paddle.sum(x=kernel)
    kernel = kernel * gain
    pad_value = tuple(kernel.shape)[0] - factor
    output = upfirdn2d_native(
        hidden_states, kernel.to(device=hidden_states.place), down=factor, pad=((pad_value + 1) // 2, pad_value // 2)
    )
    return output
