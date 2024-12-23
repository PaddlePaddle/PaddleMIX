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
from typing import Optional, Tuple

import paddle

from ppdiffusers.models.embeddings import TimestepEmbedding, Timesteps

__all__ = ["ShallowUViTEncoder", "ShallowUViTDecoder"]


class LlamaRMSNorm(paddle.nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = paddle.create_parameter(
            shape=[self.hidden_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=paddle.nn.initializer.Constant(1.0),
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if paddle.in_dynamic_mode():
            with paddle.amp.auto_cast(False):
                # hidden_states = hidden_states.astype("float32")
                # variance = hidden_states.pow(2).mean(-1, keepdim=True)
                variance = hidden_states.astype("float32").pow(2).mean(-1, keepdim=True)
                hidden_states = paddle.rsqrt(variance + self.variance_epsilon) * hidden_states
        else:
            hidden_states = hidden_states.astype("float32")
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = paddle.rsqrt(variance + self.variance_epsilon) * hidden_states

        if self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = paddle.cast(hidden_states, self.weight.dtype)
        return hidden_states * self.weight

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class GlobalResponseNorm(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.zeros(shape=[1, 1, 1, dim]))
        self.bias = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.zeros(shape=[1, 1, 1, dim]))

    def forward(self, x):
        gx = paddle.linalg.norm(x=x, p=2, axis=(1, 2), keepdim=True)
        nx = gx / (gx.mean(axis=-1, keepdim=True) + 1e-06)
        return paddle.add(self.bias, 1 * (self.weight * nx + 1) * x)


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
        stride=2,
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
        self.name = name
        if norm_type == "ln_norm":
            self.norm = paddle.nn.LayerNorm(
                normalized_shape=channels, epsilon=eps, weight_attr=elementwise_affine, bias_attr=elementwise_affine
            )
        elif norm_type == "rms_norm":
            self.norm = LlamaRMSNorm(channels, eps)
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

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        stride=2,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate
        self.stride = stride
        if norm_type == "ln_norm":
            self.norm = paddle.nn.LayerNorm(
                normalized_shape=channels, epsilon=eps, weight_attr=elementwise_affine, bias_attr=elementwise_affine
            )
        elif norm_type == "rms_norm":
            self.norm = LlamaRMSNorm(channels, eps)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")
        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            conv = paddle.nn.Conv2DTranspose(
                in_channels=channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias_attr=bias,
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = paddle.nn.Conv2D(
                in_channels=self.channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias_attr=bias,
            )
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(
        self, hidden_states: paddle.Tensor, output_size: Optional[int] = None, *args, **kwargs
    ) -> paddle.Tensor:
        assert tuple(hidden_states.shape)[1] == self.channels
        if self.norm is not None:
            hidden_states = self.norm(hidden_states.transpose(perm=[0, 2, 3, 1])).transpose(perm=[0, 3, 1, 2])
        if self.use_conv_transpose:
            return self.conv(hidden_states)
        dtype = hidden_states.dtype
        if dtype == paddle.bfloat16:
            hidden_states = hidden_states.astype("float32")
        if tuple(hidden_states.shape)[0] >= 64:
            hidden_states = hidden_states.contiguous()
        if self.interpolate:
            if output_size is None:
                hidden_states = paddle.nn.functional.interpolate(
                    x=hidden_states, scale_factor=self.stride, mode="nearest"
                )
            else:
                hidden_states = paddle.nn.functional.interpolate(x=hidden_states, size=output_size, mode="nearest")
        if dtype == paddle.bfloat16:
            hidden_states = hidden_states.astype(dtype)
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)
        return hidden_states


class ConvNextBlock(paddle.nn.Layer):
    def __init__(
        self, channels, norm_eps, elementwise_affine, use_bias, hidden_dropout, hidden_size, res_ffn_factor: int = 4
    ):
        super().__init__()
        self.depthwise = paddle.nn.Conv2D(
            in_channels=channels, out_channels=channels, kernel_size=7, padding=3, groups=channels, bias_attr=use_bias
        )
        self.norm = LlamaRMSNorm(channels, norm_eps)
        self.channelwise_linear_1 = paddle.nn.Linear(
            in_features=channels, out_features=int(channels * res_ffn_factor), bias_attr=use_bias
        )
        self.channelwise_act = paddle.nn.GELU()
        self.channelwise_norm = GlobalResponseNorm(int(channels * res_ffn_factor))
        self.channelwise_linear_2 = paddle.nn.Linear(
            in_features=int(channels * res_ffn_factor), out_features=channels, bias_attr=use_bias
        )
        self.channelwise_dropout = paddle.nn.Dropout(p=hidden_dropout)
        self.cond_embeds_mapper = paddle.nn.Linear(
            in_features=hidden_size, out_features=channels * 2, bias_attr=use_bias
        )

    def forward(self, x, cond_embeds):
        x_res = x
        x = self.depthwise(x)
        x = x.transpose(perm=[0, 2, 3, 1])
        x = self.norm(x)
        x = self.channelwise_linear_1(x)
        x = self.channelwise_act(x)
        x = paddle.cast(self.channelwise_norm(x.astype(paddle.float32)), dtype=x.dtype)
        x = self.channelwise_linear_2(x)
        x = self.channelwise_dropout(x)
        x = x.transpose(perm=[0, 3, 1, 2])
        x = x + x_res
        scale, shift = self.cond_embeds_mapper(paddle.nn.functional.silu(x=cond_embeds)).chunk(chunks=2, axis=1)
        x = paddle.add(shift[:, :, None, None], 1 * x * (1 + scale)[:, :, None, None])
        return x


class Patchify(paddle.nn.Layer):
    def __init__(self, in_channels, block_out_channels, patch_size, bias, elementwise_affine, eps, kernel_size=None):
        super().__init__()
        if kernel_size is None:
            kernel_size = patch_size
        self.patch_conv = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=block_out_channels,
            kernel_size=kernel_size,
            stride=patch_size,
            bias_attr=bias,
        )
        self.norm = LlamaRMSNorm(block_out_channels, eps)

    def forward(self, x):
        embeddings = self.patch_conv(x)
        embeddings = embeddings.transpose(perm=[0, 2, 3, 1])
        embeddings = self.norm(embeddings)
        embeddings = embeddings.transpose(perm=[0, 3, 1, 2])
        return embeddings


class Unpatchify(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, patch_size, bias, elementwise_affine, eps):
        super().__init__()
        self.norm = LlamaRMSNorm(in_channels, eps)
        self.unpatch_conv = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=out_channels * patch_size * patch_size, kernel_size=1, bias_attr=bias
        )
        self.pixel_shuffle = paddle.nn.PixelShuffle(upscale_factor=patch_size)
        self.patch_size = patch_size

    def forward(self, x):
        x = x.transpose(perm=[0, 2, 3, 1])
        x = self.norm(x)
        x = x.transpose(perm=[0, 3, 1, 2])
        x = self.unpatch_conv(x)
        x = self.pixel_shuffle(x)
        return x


class UVitBlock(paddle.nn.Layer):
    def __init__(
        self,
        channels,
        out_channels,
        num_res_blocks,
        stride,
        hidden_size,
        hidden_dropout,
        elementwise_affine,
        norm_eps,
        use_bias,
        downsample: bool,
        upsample: bool,
        res_ffn_factor: int = 4,
        seq_len=None,
        concat_input=False,
        original_input_channels=None,
        use_zero=True,
        norm_type="RMS",
    ):
        super().__init__()
        self.res_blocks = paddle.nn.LayerList()
        for i in range(num_res_blocks):
            conv_block = ConvNextBlock(
                channels,
                norm_eps,
                elementwise_affine,
                use_bias,
                hidden_dropout,
                hidden_size,
                res_ffn_factor=res_ffn_factor,
            )
            self.res_blocks.append(conv_block)
        if downsample:
            self.downsample = Downsample2D(
                channels=channels,
                out_channels=out_channels,
                use_conv=True,
                name="Conv2d_0",
                kernel_size=3,
                padding=1,
                stride=stride,
                norm_type="rms_norm",
                eps=norm_eps,
                elementwise_affine=elementwise_affine,
                bias=use_bias,
            )
        else:
            self.downsample = None
        if upsample:
            self.upsample = Upsample2D(
                channels=channels,
                out_channels=out_channels,
                use_conv_transpose=False,
                use_conv=True,
                kernel_size=3,
                padding=1,
                stride=stride,
                name="conv",
                norm_type="rms_norm",
                eps=norm_eps,
                elementwise_affine=elementwise_affine,
                bias=use_bias,
                interpolate=True,
            )
        else:
            self.upsample = None

    def forward(self, x, emb, recompute=False):
        for res_block in self.res_blocks:
            x = res_block(x, emb)
        if self.downsample is not None:
            x = self.downsample(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class ShallowUViTEncoder(paddle.nn.Layer):
    def __init__(
        self,
        input_channels=3,
        stride=4,
        kernel_size=7,
        padding=None,
        block_out_channels=(768,),
        layers_in_middle=2,
        hidden_size=2048,
        elementwise_affine=True,
        use_bias=True,
        norm_eps=1e-06,
        dropout=0.0,
        use_mid_block=True,
        **kwargs
    ):
        super().__init__()
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embed = TimestepEmbedding(block_out_channels[0], hidden_size)
        if padding is None:
            padding = math.ceil(kernel_size - stride)
        self.in_conv = paddle.nn.Conv2D(
            in_channels=input_channels,
            out_channels=block_out_channels[0],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        if use_mid_block:
            self.mid_block = UVitBlock(
                block_out_channels[-1],
                block_out_channels[-1],
                num_res_blocks=layers_in_middle,
                hidden_size=hidden_size,
                hidden_dropout=dropout,
                elementwise_affine=elementwise_affine,
                norm_eps=norm_eps,
                use_bias=use_bias,
                downsample=False,
                upsample=False,
                stride=1,
                res_ffn_factor=4,
            )
        else:
            self.mid_block = None

    def get_num_extra_tensors(self):
        return 2

    def forward(self, x, timesteps):
        bs = tuple(x.shape)[0]
        dtype = x.dtype
        t_emb = paddle.cast(self.time_proj(timesteps.flatten()).reshape([bs, -1]), dtype=dtype)
        t_emb = self.time_embed(t_emb)
        x_emb = self.in_conv(x)
        if self.mid_block is not None:
            x_emb = self.mid_block(x_emb, t_emb)
        hs = [x_emb]
        return x_emb, t_emb, hs


class ShallowUViTDecoder(paddle.nn.Layer):
    def __init__(
        self,
        in_channels=768,
        out_channels=3,
        block_out_channels: Tuple[int] = (768,),
        upsamples=2,
        layers_in_middle=2,
        hidden_size=2048,
        elementwise_affine=True,
        norm_eps=1e-06,
        use_bias=True,
        dropout=0.0,
        use_mid_block=True,
        **kwargs
    ):
        super().__init__()
        if use_mid_block:
            self.mid_block = UVitBlock(
                in_channels + block_out_channels[-1],
                block_out_channels[-1],
                num_res_blocks=layers_in_middle,
                hidden_size=hidden_size,
                hidden_dropout=dropout,
                elementwise_affine=elementwise_affine,
                norm_eps=norm_eps,
                use_bias=use_bias,
                downsample=False,
                upsample=False,
                stride=1,
                res_ffn_factor=4,
            )
        else:
            self.mid_block = None
        self.out_convs = paddle.nn.LayerList()
        for rank in range(upsamples):
            if rank == upsamples - 1:
                curr_out_channels = out_channels
            else:
                curr_out_channels = block_out_channels[-1]
            if rank == 0:
                curr_in_channels = block_out_channels[-1] + in_channels
            else:
                curr_in_channels = block_out_channels[-1]
            self.out_convs.append(
                Unpatchify(
                    curr_in_channels,
                    curr_out_channels,
                    patch_size=2,
                    bias=use_bias,
                    elementwise_affine=elementwise_affine,
                    eps=norm_eps,
                )
            )
        self.input_norm = LlamaRMSNorm(in_channels, norm_eps)

    def forward(self, x, hs, t_emb):
        x = x.transpose(perm=[0, 2, 3, 1])
        x = self.input_norm(x)
        x = x.transpose(perm=[0, 3, 1, 2])
        x = paddle.concat(x=[x, hs.pop()], axis=1)
        if self.mid_block is not None:
            x = self.mid_block(x, t_emb)
        for out_conv in self.out_convs:
            x = out_conv(x)
        assert len(hs) == 0
        return x
