import paddle
from typing import Optional, Tuple
from .normalization import RMSNorm


class Upsample1D(paddle.nn.Layer):
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

    def __init__(self, channels: int, use_conv: bool=False,
        use_conv_transpose: bool=False, out_channels: Optional[int]=None,
        name: str='conv'):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.conv = None
        if use_conv_transpose:
            self.conv = paddle.nn.Conv1DTranspose(in_channels=channels,
                out_channels=self.out_channels, kernel_size=4, stride=2,
                padding=1)
        elif use_conv:
            self.conv = paddle.nn.Conv1D(in_channels=self.channels,
                out_channels=self.out_channels, kernel_size=3, padding=1)

    def forward(self, inputs: paddle.Tensor) ->paddle.Tensor:
        assert tuple(inputs.shape)[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)
        outputs = paddle.nn.functional.interpolate(x=inputs, scale_factor=
            2.0, mode='nearest')
        if self.use_conv:
            outputs = self.conv(outputs)
        return outputs


class Upsample2D(paddle.nn.Layer):
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

    def __init__(self, channels: int, use_conv: bool=False,
        use_conv_transpose: bool=False, out_channels: Optional[int]=None,
        name: str='conv', kernel_size: Optional[int]=None, padding=1,
        norm_type=None, eps=None, elementwise_affine=None, bias=True,
        interpolate=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate
        if norm_type == 'ln_norm':
            self.norm = paddle.nn.LayerNorm(normalized_shape=channels,
                epsilon=eps, weight_attr=elementwise_affine, bias_attr=
                elementwise_affine)
        elif norm_type == 'rms_norm':
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f'unknown norm_type: {norm_type}')
        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = paddle.nn.Conv2DTranspose(in_channels=channels,
                out_channels=self.out_channels, kernel_size=kernel_size,
                stride=2, padding=padding, bias_attr=bias)
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = paddle.nn.Conv2D(in_channels=self.channels, out_channels
                =self.out_channels, kernel_size=kernel_size, padding=
                padding, bias_attr=bias)
        if name == 'conv':
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states: paddle.Tensor, output_size: Optional[
        int]=None, *args, **kwargs) ->paddle.Tensor:
        if len(args) > 0 or kwargs.get('scale', None) is not None:
            deprecation_message = (
                'The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`.'
                )
            print('scale', '1.0.0', deprecation_message)
        assert tuple(hidden_states.shape)[1] == self.channels
        if self.norm is not None:
            hidden_states = self.norm(hidden_states.transpose(perm=[0, 2, 3,
                1])).transpose(perm=[0, 3, 1, 2])
        if self.use_conv_transpose:
            return self.conv(hidden_states)
        dtype = hidden_states.dtype
        if dtype == 'bfloat16':
            hidden_states = hidden_states.cast('float32')
        if tuple(hidden_states.shape)[0] >= 64:
            hidden_states = hidden_states.contiguous()
        if self.interpolate:
            if output_size is None:
                hidden_states = paddle.nn.functional.interpolate(x=
                    hidden_states, scale_factor=2.0, mode='nearest')
            else:
                hidden_states = paddle.nn.functional.interpolate(x=
                    hidden_states, size=output_size, mode='nearest')
        if dtype == 'bfloat16':
            hidden_states = hidden_states.cast(dtype)
        if self.use_conv:
            if self.name == 'conv':
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)
        return hidden_states


class FirUpsample2D(paddle.nn.Layer):
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

    def __init__(self, channels: Optional[int]=None, out_channels: Optional
        [int]=None, use_conv: bool=False, fir_kernel: Tuple[int, int, int,
        int]=(1, 3, 3, 1)):
        super().__init__()
        out_channels = out_channels if out_channels else channels
        if use_conv:
            self.Conv2d_0 = paddle.nn.Conv2D(in_channels=channels,
                out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.use_conv = use_conv
        self.fir_kernel = fir_kernel
        self.out_channels = out_channels

    def _upsample_2d(self, hidden_states: paddle.Tensor, weight: Optional[
        paddle.Tensor]=None, kernel: Optional[paddle.Tensor]=None, factor:
        int=2, gain: float=1) ->paddle.Tensor:
        """Fused `upsample_2d()` followed by `Conv2d()`.

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
                corresponds to nearest-neighbor upsampling.
            factor (`int`, *optional*): Integer upsampling factor (default: 2).
            gain (`float`, *optional*): Scaling factor for signal magnitude (default: 1.0).

        Returns:
            output (`torch.Tensor`):
                Tensor of the shape `[N, C, H * factor, W * factor]` or `[N, H * factor, W * factor, C]`, and same
                datatype as `hidden_states`.
        """
        assert isinstance(factor, int) and factor >= 1
        if kernel is None:
            kernel = [1] * factor
        kernel = paddle.to_tensor(data=kernel, dtype='float32')
        if kernel.ndim == 1:
            kernel = paddle.outer(x=kernel, y=kernel)
        kernel /= paddle.sum(x=kernel)
        kernel = kernel * (gain * factor ** 2)
        if self.use_conv:
            convH = tuple(weight.shape)[2]
            convW = tuple(weight.shape)[3]
            inC = tuple(weight.shape)[1]
            pad_value = tuple(kernel.shape)[0] - factor - (convW - 1)
            stride = factor, factor
            output_shape = (tuple(hidden_states.shape)[2] - 1
                ) * factor + convH, (tuple(hidden_states.shape)[3] - 1
                ) * factor + convW
            output_padding = output_shape[0] - (tuple(hidden_states.shape)[
                2] - 1) * stride[0] - convH, output_shape[1] - (tuple(
                hidden_states.shape)[3] - 1) * stride[1] - convW
            assert output_padding[0] >= 0 and output_padding[1] >= 0
            num_groups = tuple(hidden_states.shape)[1] // inC
            weight = paddle.reshape(x=weight, shape=(num_groups, -1, inC,
                convH, convW))
            weight = paddle.flip(x=weight, axis=[3, 4]).transpose(perm=[0, 
                2, 1, 3, 4])
            weight = paddle.reshape(x=weight, shape=(num_groups * inC, -1,
                convH, convW))
            inverse_conv = paddle.nn.functional.conv2d_transpose(x=
                hidden_states, weight=weight, stride=stride, output_padding
                =output_padding, padding=0)
            output = upfirdn2d_native(inverse_conv, paddle.to_tensor(data=
                kernel), pad=((pad_value + 1) // 
                2 + factor - 1, pad_value // 2 + 1))
        else:
            pad_value = tuple(kernel.shape)[0] - factor
            output = upfirdn2d_native(hidden_states, paddle.to_tensor(data=
                kernel), up=factor, pad=((
                pad_value + 1) // 2 + factor - 1, pad_value // 2))
        return output

    def forward(self, hidden_states: paddle.Tensor) ->paddle.Tensor:
        if self.use_conv:
            height = self._upsample_2d(hidden_states, self.Conv2d_0.weight,
                kernel=self.fir_kernel)
            height = height + self.Conv2d_0.bias.reshape(1, -1, 1, 1)
        else:
            height = self._upsample_2d(hidden_states, kernel=self.
                fir_kernel, factor=2)
        return height


class KUpsample2D(paddle.nn.Layer):
    """A 2D K-upsampling layer.

    Parameters:
        pad_mode (`str`, *optional*, default to `"reflect"`): the padding mode to use.
    """

    def __init__(self, pad_mode: str='reflect'):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = paddle.to_tensor(data=[[1 / 8, 3 / 8, 3 / 8, 1 / 8]]) * 2
        self.pad = tuple(kernel_1d.shape)[1] // 2 - 1
        self.register_buffer(name='kernel', tensor=kernel_1d.T @ kernel_1d,
            persistable=False)

    def forward(self, inputs: paddle.Tensor) ->paddle.Tensor:
        inputs = paddle.nn.functional.pad(x=inputs, pad=((self.pad + 1) // 
            2,) * 4, mode=self.pad_mode, pad_from_left_axis=False)
        weight = paddle.zeros(shape=[tuple(inputs.shape)[1], tuple(inputs.
            shape)[1], tuple(self.kernel.shape)[0], tuple(self.kernel.shape
            )[1]], dtype=inputs.dtype)
        indices = paddle.arange(end=tuple(inputs.shape)[1])
        kernel = self.kernel.cast(weight.dtype)[None, :].expand(shape=[tuple(inputs
            .shape)[1], -1, -1])
        weight[indices, indices] = kernel
        return paddle.nn.functional.conv2d_transpose(x=inputs, weight=
            weight, stride=2, padding=self.pad * 2 + 1)


class CogVideoXUpsample3D(paddle.nn.Layer):
    """
    A 3D Upsample layer using in CogVideoX by Tsinghua University & ZhipuAI # Todo: Wait for paper relase.

    Args:
        in_channels (`int`):
            Number of channels in the input image.
        out_channels (`int`):
            Number of channels produced by the convolution.
        kernel_size (`int`, defaults to `3`):
            Size of the convolving kernel.
        stride (`int`, defaults to `1`):
            Stride of the convolution.
        padding (`int`, defaults to `1`):
            Padding added to all four sides of the input.
        compress_time (`bool`, defaults to `False`):
            Whether or not to compress the time dimension.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size:
        int=3, stride: int=1, padding: int=1, compress_time: bool=False
        ) ->None:
        super().__init__()
        self.conv = paddle.nn.Conv2D(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding)
        self.compress_time = compress_time

    def forward(self, inputs: paddle.Tensor) ->paddle.Tensor:
        if self.compress_time:
            if tuple(inputs.shape)[2] > 1 and tuple(inputs.shape)[2] % 2 == 1:
                x_first, x_rest = inputs[:, :, 0], inputs[:, :, 1:]
                x_first = paddle.nn.functional.interpolate(x=x_first,
                    scale_factor=2.0)
                x_rest = paddle.nn.functional.interpolate(x=x_rest,
                    scale_factor=2.0)
                x_first = x_first[:, :, None, :, :]
                inputs = paddle.concat(x=[x_first, x_rest], axis=2)
            elif tuple(inputs.shape)[2] > 1:
                inputs = paddle.nn.functional.interpolate(x=inputs,
                    scale_factor=2.0)
            else:
                inputs = inputs.squeeze(axis=2)
                inputs = paddle.nn.functional.interpolate(x=inputs,
                    scale_factor=2.0)
                inputs = inputs[:, :, None, :, :]
        else:
            b, c, t, h, w = tuple(inputs.shape)
            inputs = inputs.transpose(perm=[0, 2, 1, 3, 4]).reshape([b * t,
                c, h, w])
            inputs = paddle.nn.functional.interpolate(x=inputs,
                scale_factor=2.0)
            inputs = inputs.reshape([b, t, c] + inputs.shape[2:]).transpose(perm=[0, 2, 1, 3, 4])
        b, c, t, h, w = inputs.shape
        inputs = inputs.transpose(perm=[0, 2, 1, 3, 4]).reshape([b * t, c, h, w])
        inputs = self.conv(inputs)
        inputs = inputs.reshape([b, t] + inputs.shape[1:]).transpose(perm
            =[0, 2, 1, 3, 4])
        return inputs


def upfirdn2d_native(tensor: paddle.Tensor, kernel: paddle.Tensor, up: int=
    1, down: int=1, pad: Tuple[int, int]=(0, 0)) ->paddle.Tensor:
    up_x = up_y = up
    down_x = down_y = down
    pad_x0 = pad_y0 = pad[0]
    pad_x1 = pad_y1 = pad[1]
    _, channel, in_h, in_w = tuple(tensor.shape)
    tensor = tensor.reshape(-1, in_h, in_w, 1)
    _, in_h, in_w, minor = tuple(tensor.shape)
    kernel_h, kernel_w = tuple(kernel.shape)
    out = tensor.view(-1, in_h, 1, in_w, 1, minor)
    out = paddle.nn.functional.pad(x=out, pad=[0, 0, 0, up_x - 1, 0, 0, 0, 
        up_y - 1], pad_from_left_axis=False)
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)
    out = paddle.nn.functional.pad(x=out, pad=[0, 0, max(pad_x0, 0), max(
        pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)], pad_from_left_axis=False)
    out = out[:, max(-pad_y0, 0):tuple(out.shape)[1] - max(-pad_y1, 0), max
        (-pad_x0, 0):tuple(out.shape)[2] - max(-pad_x1, 0), :]
    out = out.transpose(perm=[0, 3, 1, 2])
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x +
        pad_x0 + pad_x1])
    w = paddle.flip(x=kernel, axis=[0, 1]).view(1, 1, kernel_h, kernel_w)
    out = paddle.nn.functional.conv2d(x=out, weight=w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h +
        1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1)
    out = out.transpose(perm=[0, 2, 3, 1])
    out = out[:, ::down_y, ::down_x, :]
    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1
    return out.view(-1, channel, out_h, out_w)


def upsample_2d(hidden_states: paddle.Tensor, kernel: Optional[paddle.
    Tensor]=None, factor: int=2, gain: float=1) ->paddle.Tensor:
    """Upsample2D a batch of 2D images with the given filter.
    Accepts a batch of 2D images of the shape `[N, C, H, W]` or `[N, H, W, C]` and upsamples each image with the given
    filter. The filter is normalized so that if the input pixels are constant, they will be scaled by the specified
    `gain`. Pixels outside the image are assumed to be zero, and the filter is padded with zeros so that its shape is
    a: multiple of the upsampling factor.

    Args:
        hidden_states (`torch.Tensor`):
            Input tensor of the shape `[N, C, H, W]` or `[N, H, W, C]`.
        kernel (`torch.Tensor`, *optional*):
            FIR filter of the shape `[firH, firW]` or `[firN]` (separable). The default is `[1] * factor`, which
            corresponds to nearest-neighbor upsampling.
        factor (`int`, *optional*, default to `2`):
            Integer upsampling factor.
        gain (`float`, *optional*, default to `1.0`):
            Scaling factor for signal magnitude (default: 1.0).

    Returns:
        output (`torch.Tensor`):
            Tensor of the shape `[N, C, H * factor, W * factor]`
    """
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor
    kernel = paddle.to_tensor(data=kernel, dtype='float32')
    if kernel.ndim == 1:
        kernel = paddle.outer(x=kernel, y=kernel)
    kernel /= paddle.sum(x=kernel)
    kernel = kernel * (gain * factor ** 2)
    pad_value = tuple(kernel.shape)[0] - factor
    output = upfirdn2d_native(hidden_states, kernel, up=factor, pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2))
    return output
