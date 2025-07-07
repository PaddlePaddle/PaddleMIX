# Copyright 2023 The HuggingFace Team. All rights reserved.
# `TemporalConvLayer` Copyright 2023 Alibaba DAMO-VILAB, The ModelScope Team and The HuggingFace Team. All rights reserved.
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
from typing import Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..utils import USE_PEFT_BACKEND
from .activations import get_activation
from .attention_processor import SpatialNorm
from .lora import LoRACompatibleConv, LoRACompatibleLinear
from .normalization import AdaGroupNorm


class Upsample1D(nn.Layer):
    """A 1D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 1D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.Conv1DTranspose(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1D(self.channels, self.out_channels, 3, padding=1)

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)

        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            outputs = self.conv(outputs)

        return outputs


class Downsample1D(nn.Layer):
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
            self.conv = nn.Conv1D(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.conv = nn.AvgPool1D(kernel_size=stride, stride=stride)

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        assert inputs.shape[1] == self.channels
        return self.conv(inputs)


class Upsample2D(nn.Layer):
    """A 2D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        data_format: str = "NCHW",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.data_format = data_format
        conv_cls = nn.Conv2D if USE_PEFT_BACKEND else LoRACompatibleConv

        conv = None
        if use_conv_transpose:
            conv = nn.Conv2DTranspose(channels, self.out_channels, 4, 2, 1, data_format=data_format)
        elif use_conv:
            conv = conv_cls(self.channels, self.out_channels, 3, padding=1, data_format=data_format)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(
        self,
        hidden_states: paddle.Tensor,
        output_size: Optional[int] = None,
        scale: float = 1.0,
    ) -> paddle.Tensor:
        if self.data_format == "NCHW":
            assert hidden_states.shape[1] == self.channels
        else:
            assert hidden_states.shape[3] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == paddle.bfloat16:
            hidden_states = hidden_states.cast("float32")

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        # if hidden_states.shape[0] >= 64:
        #     hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(
                hidden_states, scale_factor=2.0, mode="nearest", data_format=self.data_format
            )
        else:
            hidden_states = F.interpolate(
                hidden_states, size=output_size, mode="nearest", data_format=self.data_format
            )

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == paddle.bfloat16:
            hidden_states = hidden_states.cast(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                if isinstance(self.conv, LoRACompatibleConv) and not USE_PEFT_BACKEND:
                    hidden_states = self.conv(hidden_states, scale)
                else:
                    hidden_states = self.conv(hidden_states)
            else:
                if isinstance(self.Conv2d_0, LoRACompatibleConv) and not USE_PEFT_BACKEND:
                    hidden_states = self.Conv2d_0(hidden_states, scale)
                else:
                    hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states


class Downsample2D(nn.Layer):
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
        data_format: str = "NCHW",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        self.data_format = data_format
        conv_cls = nn.Conv2D if USE_PEFT_BACKEND else LoRACompatibleConv

        if use_conv:
            conv = conv_cls(
                self.channels, self.out_channels, 3, stride=stride, padding=padding, data_format=data_format
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2D(kernel_size=stride, stride=stride, data_format=data_format)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: paddle.Tensor, scale: float = 1.0) -> paddle.Tensor:
        if self.data_format == "NCHW":
            assert hidden_states.shape[1] == self.channels
        else:
            assert hidden_states.shape[3] == self.channels

        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0, data_format=self.data_format)

        if self.data_format == "NCHW":
            assert hidden_states.shape[1] == self.channels
        else:
            assert hidden_states.shape[3] == self.channels

        if not USE_PEFT_BACKEND:
            if isinstance(self.conv, LoRACompatibleConv):
                hidden_states = self.conv(hidden_states, scale)
            else:
                hidden_states = self.conv(hidden_states)
        else:
            hidden_states = self.conv(hidden_states)

        return hidden_states


class FirUpsample2D(nn.Layer):
    """A 2D FIR upsampling layer with an optional convolution.

    Parameters:
        channels (`int`, optional):
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
            self.Conv2d_0 = nn.Conv2D(channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.use_conv = use_conv
        self.fir_kernel = fir_kernel
        self.out_channels = out_channels

    def _upsample_2d(
        self,
        hidden_states: paddle.Tensor,
        weight: Optional[paddle.Tensor] = None,
        kernel: Optional[paddle.Tensor] = None,
        factor: int = 2,
        gain: float = 1,
    ) -> paddle.Tensor:
        """Fused `upsample_2d()` followed by `Conv2d()`.

        Padding is performed only once at the beginning, not between the operations. The fused op is considerably more
        efficient than performing the same calculation using standard TensorFlow ops. It supports gradients of
        arbitrary order.

        Args:
            hidden_states (`paddle.Tensor`):
                Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
            weight (`paddle.Tensor`, *optional*):
                Weight tensor of the shape `[filterH, filterW, inChannels, outChannels]`. Grouped convolution can be
                performed by `inChannels = x.shape[0] // numGroups`.
            kernel (`paddle.Tensor`, *optional*):
                FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
                corresponds to nearest-neighbor upsampling.
            factor (`int`, *optional*): Integer upsampling factor (default: 2).
            gain (`float`, *optional*): Scaling factor for signal magnitude (default: 1.0).

        Returns:
            output (`paddle.Tensor`):
                Tensor of the shape `[N, C, H * factor, W * factor]` or `[N, H * factor, W * factor, C]`, and same
                datatype as `hidden_states`.
        """

        assert isinstance(factor, int) and factor >= 1

        # Setup filter kernel.
        if kernel is None:
            kernel = [1] * factor

        # setup kernel
        kernel = paddle.to_tensor(kernel, dtype="float32")
        if kernel.ndim == 1:
            kernel = paddle.outer(kernel, kernel)
        kernel /= paddle.sum(kernel)

        kernel = kernel * (gain * (factor**2))

        if self.use_conv:
            convH = weight.shape[2]
            convW = weight.shape[3]
            inC = weight.shape[1]

            pad_value = (kernel.shape[0] - factor) - (convW - 1)

            stride = (factor, factor)
            # Determine data dimensions.
            output_shape = (
                (hidden_states.shape[2] - 1) * factor + convH,
                (hidden_states.shape[3] - 1) * factor + convW,
            )
            output_padding = (
                output_shape[0] - (hidden_states.shape[2] - 1) * stride[0] - convH,
                output_shape[1] - (hidden_states.shape[3] - 1) * stride[1] - convW,
            )
            assert output_padding[0] >= 0 and output_padding[1] >= 0
            num_groups = hidden_states.shape[1] // inC

            # Transpose weights.
            weight = weight.reshape([num_groups, -1, inC, convH, convW])
            weight = paddle.flip(weight, axis=[3, 4]).transpose([0, 2, 1, 3, 4])
            weight = weight.reshape([num_groups * inC, -1, convH, convW])

            inverse_conv = F.conv2d_transpose(
                hidden_states,
                weight,
                stride=stride,
                output_padding=output_padding,
                padding=0,
            )

            output = upfirdn2d_native(
                inverse_conv,
                paddle.to_tensor(kernel),
                pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2 + 1),
            )
        else:
            pad_value = kernel.shape[0] - factor
            output = upfirdn2d_native(
                hidden_states,
                paddle.to_tensor(kernel),
                up=factor,
                pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
            )

        return output

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        if self.use_conv:
            height = self._upsample_2d(hidden_states, self.Conv2d_0.weight, kernel=self.fir_kernel)
            height = height + self.Conv2d_0.bias.reshape([1, -1, 1, 1])
        else:
            height = self._upsample_2d(hidden_states, kernel=self.fir_kernel, factor=2)

        return height


class FirDownsample2D(nn.Layer):
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
            self.Conv2d_0 = nn.Conv2D(channels, out_channels, kernel_size=3, stride=1, padding=1)
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
            hidden_states (`paddle.Tensor`):
                Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
            weight (`paddle.Tensor`, *optional*):
                Weight tensor of the shape `[filterH, filterW, inChannels, outChannels]`. Grouped convolution can be
                performed by `inChannels = x.shape[0] // numGroups`.
            kernel (`paddle.Tensor`, *optional*):
                FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
                corresponds to average pooling.
            factor (`int`, *optional*, default to `2`):
                Integer downsampling factor.
            gain (`float`, *optional*, default to `1.0`):
                Scaling factor for signal magnitude.

        Returns:
            output (`paddle.Tensor`):
                Tensor of the shape `[N, C, H // factor, W // factor]` or `[N, H // factor, W // factor, C]`, and same
                datatype as `x`.
        """

        assert isinstance(factor, int) and factor >= 1
        if kernel is None:
            kernel = [1] * factor

        # setup kernel
        kernel = paddle.to_tensor(kernel, dtype="float32")
        if kernel.ndim == 1:
            kernel = paddle.outer(kernel, kernel)
        kernel /= paddle.sum(kernel)

        kernel = kernel * gain

        if self.use_conv:
            _, _, convH, convW = weight.shape
            pad_value = (kernel.shape[0] - factor) + (convW - 1)
            stride_value = [factor, factor]
            upfirdn_input = upfirdn2d_native(
                hidden_states,
                paddle.to_tensor(kernel),
                pad=((pad_value + 1) // 2, pad_value // 2),
            )
            output = F.conv2d(upfirdn_input, weight, stride=stride_value, padding=0)
        else:
            pad_value = kernel.shape[0] - factor
            output = upfirdn2d_native(
                hidden_states,
                paddle.to_tensor(kernel),
                down=factor,
                pad=((pad_value + 1) // 2, pad_value // 2),
            )

        return output

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        if self.use_conv:
            downsample_input = self._downsample_2d(hidden_states, weight=self.Conv2d_0.weight, kernel=self.fir_kernel)
            hidden_states = downsample_input + self.Conv2d_0.bias.reshape([1, -1, 1, 1])
        else:
            hidden_states = self._downsample_2d(hidden_states, kernel=self.fir_kernel, factor=2)

        return hidden_states


# downsample/upsample layer used in k-upscaler, might be able to use FirDownsample2D/DirUpsample2D instead
class KDownsample2D(nn.Layer):
    r"""A 2D K-downsampling layer.

    Parameters:
        pad_mode (`str`, *optional*, default to `"reflect"`): the padding mode to use.
    """

    def __init__(self, pad_mode: str = "reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = paddle.to_tensor([[1 / 8, 3 / 8, 3 / 8, 1 / 8]])
        self.pad = kernel_1d.shape[1] // 2 - 1
        self.register_buffer("kernel", paddle.matmul(kernel_1d, kernel_1d, transpose_x=True), persistable=False)

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        inputs = F.pad(inputs, (self.pad,) * 4, self.pad_mode)
        weight = paddle.zeros(
            [inputs.shape[1], inputs.shape[1], self.kernel.shape[0], self.kernel.shape[1]], dtype=inputs.dtype
        )
        indices = paddle.arange(inputs.shape[1])
        # TODO verify this method
        weight[indices, indices] = self.kernel.cast(weight.dtype).expand([inputs.shape[1], -1, -1])
        return F.conv2d(inputs, weight, stride=2)


class KUpsample2D(nn.Layer):
    r"""A 2D K-upsampling layer.

    Parameters:
        pad_mode (`str`, *optional*, default to `"reflect"`): the padding mode to use.
    """

    def __init__(self, pad_mode: str = "reflect"):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = paddle.to_tensor([[1 / 8, 3 / 8, 3 / 8, 1 / 8]]) * 2
        self.pad = kernel_1d.shape[1] // 2 - 1
        self.register_buffer("kernel", paddle.matmul(kernel_1d, kernel_1d, transpose_x=True), persistable=False)

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        inputs = F.pad(inputs, ((self.pad + 1) // 2,) * 4, self.pad_mode)
        weight = paddle.zeros(
            [inputs.shape[1], inputs.shape[1], self.kernel.shape[0], self.kernel.shape[1]], dtype=inputs.dtype
        )
        indices = paddle.arange(inputs.shape[1])
        # TODO verify this method
        weight[indices, indices] = self.kernel.cast(weight.dtype).expand([inputs.shape[1], -1, -1])
        return F.conv2d_transpose(inputs, weight, stride=2, padding=self.pad * 2 + 1)


class ResnetBlock2D(nn.Layer):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"default"` ): Time scale shift config.
            By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift" or
            "ada_group" for a stronger conditioning with scale and shift.
        kernel (`paddle.Tensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_2d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # default, scale_shift, ada_group, spatial
        kernel: Optional[paddle.Tensor] = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
        data_format: str = "NCHW",
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act
        self.data_format = data_format

        linear_cls = nn.Linear if USE_PEFT_BACKEND else LoRACompatibleLinear
        conv_cls = nn.Conv2D if USE_PEFT_BACKEND else LoRACompatibleConv

        if groups_out is None:
            groups_out = groups

        if self.time_embedding_norm == "ada_group":
            self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm1 = SpatialNorm(in_channels, temb_channels)
        else:
            self.norm1 = nn.GroupNorm(
                num_groups=groups, num_channels=in_channels, epsilon=eps, data_format=data_format
            )

        self.conv1 = conv_cls(in_channels, out_channels, kernel_size=3, stride=1, padding=1, data_format=data_format)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = linear_cls(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = linear_cls(temb_channels, 2 * out_channels)
            elif self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                self.time_emb_proj = None
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm2 = SpatialNorm(out_channels, temb_channels)
        else:
            self.norm2 = nn.GroupNorm(
                num_groups=groups_out, num_channels=out_channels, epsilon=eps, data_format=data_format
            )

        self.dropout = nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = conv_cls(
            out_channels, conv_2d_out_channels, kernel_size=3, stride=1, padding=1, data_format=data_format
        )

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                self.upsample = Upsample2D(in_channels, use_conv=False, data_format=data_format)
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample2D(
                    in_channels, use_conv=False, padding=1, name="op", data_format=data_format
                )

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = conv_cls(
                in_channels,
                conv_2d_out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=conv_shortcut_bias,
                data_format=data_format,
            )

    def forward(
        self,
        input_tensor: paddle.Tensor,
        temb: paddle.Tensor,
        scale: float = 1.0,
    ) -> paddle.Tensor:
        hidden_states = input_tensor

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = (
                self.upsample(input_tensor, scale=scale)
                if isinstance(self.upsample, Upsample2D)
                else self.upsample(input_tensor)
            )
            hidden_states = (
                self.upsample(hidden_states, scale=scale)
                if isinstance(self.upsample, Upsample2D)
                else self.upsample(hidden_states)
            )
        elif self.downsample is not None:
            input_tensor = (
                self.downsample(input_tensor, scale=scale)
                if isinstance(self.downsample, Downsample2D)
                else self.downsample(input_tensor)
            )
            hidden_states = (
                self.downsample(hidden_states, scale=scale)
                if isinstance(self.downsample, Downsample2D)
                else self.downsample(hidden_states)
            )

        hidden_states = self.conv1(hidden_states, scale) if not USE_PEFT_BACKEND else self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = (
                self.time_emb_proj(temb, scale)[:, :, None, None]
                if not USE_PEFT_BACKEND
                else self.time_emb_proj(temb)[:, :, None, None]
            )

        if temb is not None and self.time_embedding_norm == "default":
            if self.data_format == "NHWC":
                temb = temb.transpose([0, 2, 3, 1])
            hidden_states = hidden_states + temb

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = paddle.chunk(temb, 2, axis=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, scale) if not USE_PEFT_BACKEND else self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = (
                self.conv_shortcut(input_tensor, scale) if not USE_PEFT_BACKEND else self.conv_shortcut(input_tensor)
            )

        # TODO this maybe result -inf, input_tensor's min value -57644  hidden_states's min value -10000
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


# unet_rl.py
def rearrange_dims(tensor: paddle.Tensor) -> paddle.Tensor:
    if len(tensor.shape) == 2:
        return tensor[:, :, None]
    if len(tensor.shape) == 3:
        return tensor[:, :, None, :]
    elif len(tensor.shape) == 4:
        return tensor[:, :, 0, :]
    else:
        raise ValueError(f"`len(tensor)`: {len(tensor)} has to be 2, 3 or 4.")


class Conv1dBlock(nn.Layer):
    """
    Conv1d --> GroupNorm --> Mish

    Parameters:
        inp_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        kernel_size (`int` or `tuple`): Size of the convolving kernel.
        n_groups (`int`, default `8`): Number of groups to separate the channels into.
        activation (`str`, defaults to `mish`): Name of the activation function.
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        n_groups: int = 8,
        activation: str = "mish",
    ):
        super().__init__()

        self.conv1d = nn.Conv1D(inp_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.group_norm = nn.GroupNorm(n_groups, out_channels)
        self.mish = get_activation(activation)

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        intermediate_repr = self.conv1d(inputs)
        intermediate_repr = rearrange_dims(intermediate_repr)
        intermediate_repr = self.group_norm(intermediate_repr)
        intermediate_repr = rearrange_dims(intermediate_repr)
        output = self.mish(intermediate_repr)
        return output


# unet_rl.py
class ResidualTemporalBlock1D(nn.Layer):
    """
    Residual 1D block with temporal convolutions.

    Parameters:
        inp_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        embed_dim (`int`): Embedding dimension.
        kernel_size (`int` or `tuple`): Size of the convolving kernel.
        activation (`str`, defaults `mish`): It is possible to choose the right activation function.
    """

    def __init__(
        self,
        inp_channels: int,
        out_channels: int,
        embed_dim: int,
        kernel_size: Union[int, Tuple[int, int]] = 5,
        activation: str = "mish",
    ):
        super().__init__()
        self.conv_in = Conv1dBlock(inp_channels, out_channels, kernel_size)
        self.conv_out = Conv1dBlock(out_channels, out_channels, kernel_size)

        self.time_emb_act = get_activation(activation)
        self.time_emb = nn.Linear(embed_dim, out_channels)

        self.residual_conv = (
            nn.Conv1D(inp_channels, out_channels, 1) if inp_channels != out_channels else nn.Identity()
        )

    def forward(self, inputs: paddle.Tensor, t: paddle.Tensor) -> paddle.Tensor:
        """
        Args:
            inputs : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]

        returns:
            out : [ batch_size x out_channels x horizon ]
        """
        t = self.time_emb_act(t)
        t = self.time_emb(t)
        out = self.conv_in(inputs) + rearrange_dims(t)
        out = self.conv_out(out)
        return out + self.residual_conv(inputs)


def upsample_2d(
    hidden_states: paddle.Tensor,
    kernel: Optional[paddle.Tensor] = None,
    factor: int = 2,
    gain: float = 1,
) -> paddle.Tensor:
    r"""Upsample2D a batch of 2D images with the given filter.
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and upsamples each image with the given
    filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the specified
    `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its shape is
    a: multiple of the upsampling factor.

    Args:
        hidden_states (`paddle.Tensor`):
            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        kernel (`paddle.Tensor`, *optional*):
            FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
            corresponds to nearest-neighbor upsampling.
        factor (`int`, *optional*, default to `2`):
            Integer upsampling factor.
        gain (`float`, *optional*, default to `1.0`):
            Scaling factor for signal magnitude (default: 1.0).

    Returns:
        output (`paddle.Tensor`):
            Tensor of the shape `[N, C, H * factor, W * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = paddle.to_tensor(kernel, dtype=paddle.float32)
    if kernel.ndim == 1:
        kernel = paddle.outer(kernel, kernel)
    kernel /= paddle.sum(kernel)

    kernel = kernel * (gain * (factor**2))
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d_native(
        hidden_states,
        kernel,
        up=factor,
        pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
    )
    return output


def downsample_2d(
    hidden_states: paddle.Tensor,
    kernel: Optional[paddle.Tensor] = None,
    factor: int = 2,
    gain: float = 1,
) -> paddle.Tensor:
    r"""Downsample2D a batch of 2D images with the given filter.
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and downsamples each image with the
    given filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the
    specified `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its
    shape is a multiple of the downsampling factor.

    Args:
        hidden_states (`paddle.Tensor`)
            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        kernel (`paddle.Tensor`, *optional*):
            FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
            corresponds to average pooling.
        factor (`int`, *optional*, default to `2`):
            Integer downsampling factor.
        gain (`float`, *optional*, default to `1.0`):
            Scaling factor for signal magnitude.

    Returns:
        output (`paddle.Tensor`):
            Tensor of the shape `[N, C, H // factor, W // factor]`
    """

    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = paddle.to_tensor(kernel, dtype=paddle.float32)
    if kernel.ndim == 1:
        kernel = paddle.outer(kernel, kernel)
    kernel /= paddle.sum(kernel)

    kernel = kernel * gain
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d_native(
        hidden_states,
        kernel,
        down=factor,
        pad=((pad_value + 1) // 2, pad_value // 2),
    )
    return output


def dummy_pad(tensor, up_x=0, up_y=0):
    if up_x > 0:
        tensor = paddle.concat(
            [
                tensor,
                paddle.zeros(
                    [tensor.shape[0], tensor.shape[1], tensor.shape[2], tensor.shape[3], up_x, tensor.shape[5]],
                    dtype=tensor.dtype,
                ),
            ],
            axis=4,
        )
    if up_y > 0:
        tensor = paddle.concat(
            [
                tensor,
                paddle.zeros(
                    [tensor.shape[0], tensor.shape[1], up_y, tensor.shape[3], tensor.shape[4], tensor.shape[5]],
                    dtype=tensor.dtype,
                ),
            ],
            axis=2,
        )
    return tensor


def upfirdn2d_native(
    tensor: paddle.Tensor,
    kernel: paddle.Tensor,
    up: int = 1,
    down: int = 1,
    pad: Tuple[int, int] = (0, 0),
) -> paddle.Tensor:
    up_x = up_y = up
    down_x = down_y = down
    pad_x0 = pad_y0 = pad[0]
    pad_x1 = pad_y1 = pad[1]

    _, channel, in_h, in_w = tensor.shape
    tensor = tensor.reshape([-1, in_h, in_w, 1])

    _, in_h, in_w, minor = tensor.shape
    kernel_h, kernel_w = kernel.shape

    out = tensor.reshape([-1, in_h, 1, in_w, 1, minor])
    # (TODO, junnyu F.pad bug)
    # F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = dummy_pad(out, up_x - 1, up_y - 1)
    out = out.reshape([-1, in_h * up_y, in_w * up_x, minor])

    # (TODO, junnyu F.pad bug)
    # out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out.unsqueeze(0)
    out = F.pad(out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0), 0, 0], data_format="NDHWC")
    out = out.squeeze(0)

    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.transpose([0, 3, 1, 2])
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = paddle.flip(kernel, [0, 1]).reshape([1, 1, kernel_h, kernel_w])
    out = F.conv2d(out, w)
    out = out.reshape(
        [-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1]
    )
    out = out.transpose([0, 2, 3, 1])
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.reshape([-1, channel, out_h, out_w])


class TemporalConvLayer(nn.Layer):
    """
    Temporal convolutional layer that can be used for video (sequence of images) input Code mostly copied from:
    https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/models/multi_modal/video_synthesis/unet_sd.py#L1016

    Parameters:
        in_dim (`int`): Number of input channels.
        out_dim (`int`): Number of output channels.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        # conv layers
        self.conv1 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, in_dim),
            nn.Silu(),
            nn.Conv3D(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.Silu(),
            nn.Dropout(dropout),
            nn.Conv3D(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.Silu(),
            nn.Dropout(dropout),
            nn.Conv3D(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv4 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.Silu(),
            nn.Dropout(dropout),
            nn.Conv3D(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )

        # zero out the last layer params,so the conv block is identity
        with paddle.no_grad():
            self.conv4[-1].weight.zero_()
            self.conv4[-1].bias.zero_()

    def forward(self, hidden_states: paddle.Tensor, num_frames: int = 1) -> paddle.Tensor:
        hidden_states = (
            hidden_states[None, :].reshape([-1, num_frames] + hidden_states.shape[1:]).transpose([0, 2, 1, 3, 4])
        )

        identity = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = self.conv4(hidden_states)

        hidden_states = identity + hidden_states

        hidden_states = hidden_states.transpose([0, 2, 1, 3, 4]).reshape(
            [hidden_states.shape[0] * hidden_states.shape[2], -1] + hidden_states.shape[3:]
        )
        return hidden_states


class TemporalResnetBlock(nn.Layer):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        kernel_size = (3, 1, 1)
        padding = [k // 2 for k in kernel_size]

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, epsilon=eps)
        self.conv1 = nn.Conv3D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, epsilon=eps)

        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv3D(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        self.nonlinearity = get_activation("silu")

        self.use_in_shortcut = self.in_channels != out_channels

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv3D(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )

    def forward(self, input_tensor: paddle.Tensor, temb: paddle.Tensor) -> paddle.Tensor:
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, :, None, None]
            temb = temb.transpose([0, 2, 1, 3, 4])
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor


# VideoResBlock
class SpatioTemporalResBlock(nn.Layer):
    r"""
    A SpatioTemporal Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the spatial resenet.
        temporal_eps (`float`, *optional*, defaults to `eps`): The epsilon to use for the temporal resnet.
        merge_factor (`float`, *optional*, defaults to `0.5`): The merge factor to use for the temporal mixing.
        merge_strategy (`str`, *optional*, defaults to `learned_with_images`):
            The merge strategy to use for the temporal mixing.
        switch_spatial_to_temporal_mix (`bool`, *optional*, defaults to `False`):
            If `True`, switch the spatial and temporal mixing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        temb_channels: int = 512,
        eps: float = 1e-6,
        temporal_eps: Optional[float] = None,
        merge_factor: float = 0.5,
        merge_strategy="learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
    ):
        super().__init__()

        self.spatial_res_block = ResnetBlock2D(
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            eps=eps,
        )

        self.temporal_res_block = TemporalResnetBlock(
            in_channels=out_channels if out_channels is not None else in_channels,
            out_channels=out_channels if out_channels is not None else in_channels,
            temb_channels=temb_channels,
            eps=temporal_eps if temporal_eps is not None else eps,
        )

        self.time_mixer = AlphaBlender(
            alpha=merge_factor,
            merge_strategy=merge_strategy,
            switch_spatial_to_temporal_mix=switch_spatial_to_temporal_mix,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        temb: Optional[paddle.Tensor] = None,
        image_only_indicator: Optional[paddle.Tensor] = None,
    ):
        num_frames = image_only_indicator.shape[-1]
        hidden_states = self.spatial_res_block(hidden_states, temb)

        batch_frames, channels, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states_mix = (
            hidden_states[None, :]
            .reshape([batch_size, num_frames, channels, height, width])
            .transpose([0, 2, 1, 3, 4])
        )
        hidden_states = (
            hidden_states[None, :]
            .reshape([batch_size, num_frames, channels, height, width])
            .transpose([0, 2, 1, 3, 4])
        )

        if temb is not None:
            temb = temb.reshape([batch_size, num_frames, -1])

        hidden_states = self.temporal_res_block(hidden_states, temb)
        hidden_states = self.time_mixer(
            x_spatial=hidden_states_mix,
            x_temporal=hidden_states,
            image_only_indicator=image_only_indicator,
        )

        hidden_states = hidden_states.transpose([0, 2, 1, 3, 4]).reshape([batch_frames, channels, height, width])
        return hidden_states


class AlphaBlender(nn.Layer):
    r"""
    A module to blend spatial and temporal features.

    Parameters:
        alpha (`float`): The initial value of the blending factor.
        merge_strategy (`str`, *optional*, defaults to `learned_with_images`):
            The merge strategy to use for the temporal mixing.
        switch_spatial_to_temporal_mix (`bool`, *optional*, defaults to `False`):
            If `True`, switch the spatial and temporal mixing.
    """

    strategies = ["learned", "fixed", "learned_with_images"]

    def __init__(
        self,
        alpha: float,
        merge_strategy: str = "learned_with_images",
        switch_spatial_to_temporal_mix: bool = False,
    ):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.switch_spatial_to_temporal_mix = switch_spatial_to_temporal_mix  # For TemporalVAE

        if merge_strategy not in self.strategies:
            raise ValueError(f"merge_strategy needs to be in {self.strategies}")

        if self.merge_strategy == "fixed":
            self.register_buffer("mix_factor", paddle.to_tensor([alpha]))
        elif self.merge_strategy == "learned" or self.merge_strategy == "learned_with_images":
            self.mix_factor = nn.Parameter(paddle.to_tensor([alpha]))
        else:
            raise ValueError(f"Unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, image_only_indicator: paddle.Tensor, ndims: int) -> paddle.Tensor:
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor

        elif self.merge_strategy == "learned":
            alpha = F.sigmoid(self.mix_factor)

        elif self.merge_strategy == "learned_with_images":
            if image_only_indicator is None:
                raise ValueError("Please provide image_only_indicator to use learned_with_images merge strategy")

            alpha = paddle.where(
                image_only_indicator.cast("bool"),
                paddle.ones([1, 1]),
                F.sigmoid(self.mix_factor)[..., None],
            )

            # (batch, channel, frames, height, width)
            if ndims == 5:
                alpha = alpha[:, None, :, None, None]
            # (batch*frames, height*width, channels)
            elif ndims == 3:
                alpha = alpha.reshape(
                    [
                        -1,
                    ]
                )[:, None, None]
            else:
                raise ValueError(f"Unexpected ndims {ndims}. Dimensions should be 3 or 5")

        else:
            raise NotImplementedError

        return alpha

    def forward(
        self,
        x_spatial: paddle.Tensor,
        x_temporal: paddle.Tensor,
        image_only_indicator: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        alpha = self.get_alpha(image_only_indicator, x_spatial.ndim)
        alpha = alpha.cast(x_spatial.dtype)

        if self.switch_spatial_to_temporal_mix:
            alpha = 1.0 - alpha

        x = alpha * x_spatial + (1.0 - alpha) * x_temporal
        return x
