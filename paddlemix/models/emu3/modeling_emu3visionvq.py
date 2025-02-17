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

import paddle
import paddle.nn as nn
import paddlenlp

""" Emu3VisionVQ model """
import math
from typing import Optional, Tuple, Union

from .configuration_emu3 import Emu3VisionVQConfig


class Emu3VisionVQActivation(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, x: paddle.Tensor):
        return x * paddle.nn.functional.sigmoid(x=x)


class Emu3VisionVQUpsample(paddle.nn.Layer):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: paddle.Tensor):
        x = paddle.nn.functional.interpolate(x=x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Emu3VisionVQDownsample(paddle.nn.Layer):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=0
        )

    def forward(self, x: paddle.Tensor):
        pad = 0, 1, 0, 1
        x = paddle.nn.functional.pad(x=x, pad=pad, mode="constant", value=0, pad_from_left_axis=False)
        x = self.conv(x)
        return x


class Emu3VisionVQCausalConv3d(paddle.nn.Layer):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Union[int, Tuple[int, ...]] = (3, 1, 1),
        stride: Union[int, Tuple[int, ...]] = (1, 1, 1),
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3
        hw_pad = [(k - s) for k, s in zip(kernel_size[1:], stride[1:])]
        self.padding = tuple()
        for p in hw_pad[::-1]:
            self.padding += p // 2 + p % 2, p // 2
        self.padding += 2, 0
        self.conv = paddle.nn.Conv3D(
            in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride
        )

    def forward(self, x: paddle.Tensor):
        x = paddle.nn.functional.pad(x=x, pad=self.padding, pad_from_left_axis=False)
        x = self.conv(x)
        return x


class Emu3VisionVQResnetTemporalBlock(paddle.nn.Layer):
    def __init__(
        self, in_channels: int, out_channels: Optional[int] = None, conv_shortcut: bool = False, dropout: float = 0.0
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        stride = 1, 1, 1
        kernel_size = 3, 3, 3
        self.norm1 = paddle.nn.BatchNorm3D(num_features=in_channels)
        self.conv1 = Emu3VisionVQCausalConv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.norm2 = paddle.nn.BatchNorm3D(num_features=out_channels)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.conv2 = Emu3VisionVQCausalConv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.act = Emu3VisionVQActivation()
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = Emu3VisionVQCausalConv3d(
                    in_channels, out_channels, kernel_size=kernel_size, stride=stride
                )
            else:
                self.nin_shortcut = paddle.nn.Conv3D(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x: paddle.Tensor):
        origin_dtype = x.dtype
        if self.norm1.weight.dtype != paddle.float32:
            self.norm1.to(dtype="float32")
        h = self.norm1(x.astype("float32")).astype(origin_dtype)
        h = self.act(h)
        h = self.conv1(h)
        if self.norm2.weight.dtype != paddle.float32:
            self.norm2.to(dtype="float32")
        h = self.norm2(h.astype("float32")).astype(origin_dtype)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class Emu3VisionVQSpatialNorm(paddle.nn.Layer):
    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        norm_layer: paddle.nn.Layer = paddle.nn.GroupNorm,
        add_conv: bool = False,
        num_groups: int = 32,
        epsilon: float = 1e-06,
    ):
        super().__init__()
        self.norm_layer = norm_layer(num_channels=f_channels, num_groups=num_groups, epsilon=epsilon)
        self.add_conv = add_conv
        if self.add_conv:
            self.conv = paddle.nn.Conv2D(
                in_channels=zq_channels, out_channels=zq_channels, kernel_size=3, stride=1, padding=1
            )
        self.conv_y = paddle.nn.Conv2D(
            in_channels=zq_channels, out_channels=f_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv_b = paddle.nn.Conv2D(
            in_channels=zq_channels, out_channels=f_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: paddle.Tensor, zq: paddle.Tensor):
        zq = paddle.nn.functional.interpolate(x=zq, size=tuple(x.shape)[-2:], mode="nearest")
        if self.add_conv:
            zq = self.conv(zq)
        x = self.norm_layer(x)
        x = x * self.conv_y(zq) + self.conv_b(zq)
        return x


class Emu3VisionVQResnetBlock(paddle.nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        zq_ch: Optional[int] = None,
        add_conv: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.zq_ch = zq_ch
        if zq_ch is None:
            norm_kwargs = dict(num_groups=32, epsilon=1e-06)  # affine=True
            self.norm1 = paddle.nn.GroupNorm(num_channels=in_channels, **norm_kwargs)
            self.norm2 = paddle.nn.GroupNorm(num_channels=out_channels, **norm_kwargs)
        else:
            self.norm1 = Emu3VisionVQSpatialNorm(in_channels, zq_ch, add_conv=add_conv)
            self.norm2 = Emu3VisionVQSpatialNorm(out_channels, zq_ch, add_conv=add_conv)
        self.conv1 = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.conv2 = paddle.nn.Conv2D(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )
        self.act = Emu3VisionVQActivation()
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = paddle.nn.Conv2D(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = paddle.nn.Conv2D(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x: paddle.Tensor, zq: Optional[paddle.Tensor] = None):
        norm_args = tuple() if self.zq_ch is None else (zq,)
        h = self.norm1(x, *norm_args)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h, *norm_args)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class Emu3VisionVQAttnBlock(paddle.nn.Layer):
    def __init__(self, in_channels: int, zq_ch: Optional[int] = None, add_conv: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.zq_ch = zq_ch
        if zq_ch is None:
            norm_kwargs = dict(num_groups=32, epsilon=1e-06)
            self.norm = paddle.nn.GroupNorm(num_channels=in_channels, **norm_kwargs)
        else:
            self.norm = Emu3VisionVQSpatialNorm(in_channels, zq_ch, add_conv=add_conv)
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

    def forward(self, x: paddle.Tensor, zq: Optional[paddle.Tensor] = None):
        norm_args = tuple() if self.zq_ch is None else (zq,)
        nx = self.norm(x, *norm_args)
        q = self.q(nx)
        k = self.k(nx)
        v = self.v(nx)
        b, c, h, w = tuple(q.shape)
        q = q.reshape([b, c, h * w])
        k = k.reshape([b, c, h * w])
        score = paddle.bmm(x=q.transpose(perm=[0, 2, 1]), y=k)
        score = score / c**0.5
        score = paddle.nn.functional.softmax(x=score, axis=2)
        v = v.reshape([b, c, h * w])
        v = paddle.bmm(x=v, y=score.transpose(perm=[0, 2, 1]))
        v = v.reshape([b, c, h, w])
        v = self.proj_out(v)
        return x + v


class Emu3VisionVQTemporalUpsample(paddle.nn.Layer):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Tuple[int, ...] = (3, 3, 3),
        stride: Tuple[int, ...] = (1, 1, 1),
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = Emu3VisionVQCausalConv3d(in_channel, out_channel, kernel_size, stride=stride)

    def forward(self, x: paddle.Tensor):
        b, c, t, h, w = tuple(x.shape)
        x = x.transpose(perm=[0, 1, 3, 4, 2]).reshape([b, -1, t])
        x = paddle.nn.functional.interpolate(x=x, scale_factor=2.0, mode="linear")
        x = x.reshape([b, c, h, w, -1]).transpose(perm=[0, 1, 4, 2, 3])
        x = self.conv(x)
        return x


class Emu3VisionVQTemporalDownsample(paddle.nn.Layer):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Tuple[int, ...] = (4, 3, 3),
        stride: Tuple[int, ...] = (2, 1, 1),
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.conv = Emu3VisionVQCausalConv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)

    def forward(self, x: paddle.Tensor):
        x = self.conv(x)
        return x


class Emu3VisionVQVectorQuantizer(paddle.nn.Layer):
    def __init__(self, config: Emu3VisionVQConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.codebook_size, config.embed_dim)
        # self.embedding.weight.data.uniform_(min=-1.0 / config.codebook_size,
        #     max=1.0 / config.codebook_size)

    def forward(self, x: paddle.Tensor):
        b, t, c, h, w = tuple(x.shape)
        x = x.transpose(perm=[0, 1, 3, 4, 2])
        x_flattened = x.reshape([-1, c])
        codebook = self.embedding.weight
        d = (
            paddle.sum(x=x_flattened**2, axis=1, keepdim=True)
            + paddle.sum(x=codebook**2, axis=1)
            - 2 * paddle.einsum("bd,dn->bn", x_flattened, codebook.transpose(perm=[1, 0]))
        )
        indices = paddle.argmin(x=d, axis=1)
        indices = indices.reshape([b, t, h, w])
        return indices


class Emu3VisionVQEncoder(paddle.nn.Layer):
    def __init__(self, config: Emu3VisionVQConfig):
        super().__init__()
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        self.in_channels = config.in_channels
        self.conv_in = paddle.nn.Conv2D(
            in_channels=self.in_channels, out_channels=self.ch, kernel_size=3, stride=1, padding=1
        )
        in_ch_mult = (1,) + tuple(config.ch_mult)
        self.down = paddle.nn.LayerList()
        for i_level in range(self.num_resolutions):
            block = paddle.nn.LayerList()
            attn = paddle.nn.LayerList()
            block_in = config.ch * in_ch_mult[i_level]
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    Emu3VisionVQResnetBlock(in_channels=block_in, out_channels=block_out, dropout=config.dropout)
                )
                block_in = block_out
                if i_level in config.attn_resolutions:
                    attn.append(Emu3VisionVQAttnBlock(block_in))
            down = paddle.nn.Layer()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Emu3VisionVQDownsample(block_in)
            self.down.append(down)
        self.mid = paddle.nn.Layer()
        self.mid.block_1 = Emu3VisionVQResnetBlock(in_channels=block_in, out_channels=block_in, dropout=config.dropout)
        self.mid.attn_1 = Emu3VisionVQAttnBlock(block_in)
        self.mid.block_2 = Emu3VisionVQResnetBlock(in_channels=block_in, out_channels=block_in, dropout=config.dropout)
        self.norm_out = paddle.nn.GroupNorm(
            num_channels=block_in, num_groups=32, epsilon=1e-06, weight_attr=True, bias_attr=True
        )
        out_z_channels = 2 * config.z_channels if config.double_z else config.z_channels
        self.conv_out = paddle.nn.Conv2D(
            in_channels=block_in, out_channels=out_z_channels, kernel_size=3, stride=1, padding=1
        )
        temporal_down_blocks = int(math.log2(config.temporal_downsample_factor))
        self.time_conv = paddle.nn.LayerList()
        for i in range(temporal_down_blocks):
            conv = Emu3VisionVQTemporalDownsample(out_z_channels, out_z_channels)
            self.time_conv.append(conv)
        self.time_res_stack = paddle.nn.Sequential(
            *[
                Emu3VisionVQResnetTemporalBlock(
                    in_channels=out_z_channels, out_channels=out_z_channels, dropout=config.dropout
                )
                for _ in range(self.num_res_blocks)
            ]
        )
        self.act = Emu3VisionVQActivation()

    def forward(self, x: paddle.Tensor):
        t = tuple(x.shape)[1]
        x = x.reshape([-1, *tuple(x.shape)[2:]])
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = self.act(h)
        h = self.conv_out(h)
        h = h.reshape([-1, t, *tuple(h.shape)[1:]])
        h = h.transpose(perm=[0, 2, 1, 3, 4])
        for conv in self.time_conv:
            h = self.act(conv(h))
        h = self.time_res_stack(h)
        h = h.transpose(perm=[0, 2, 1, 3, 4])
        return h


class Emu3VisionVQDecoder(paddle.nn.Layer):
    def __init__(self, config: Emu3VisionVQConfig):
        super().__init__()
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        # in_ch_mult = (1,) + tuple(config.ch_mult)
        zq_ch = config.embed_dim
        block_in = config.ch * config.ch_mult[-1]
        self.time_res_stack = paddle.nn.Sequential(
            *[
                Emu3VisionVQResnetTemporalBlock(
                    in_channels=config.z_channels, out_channels=config.z_channels, dropout=config.dropout
                )
                for _ in range(config.num_res_blocks)
            ]
        )
        tempo_upsample_block_num = int(math.log2(config.temporal_downsample_factor))
        self.time_conv = paddle.nn.LayerList()
        for i in range(tempo_upsample_block_num):
            conv = Emu3VisionVQTemporalUpsample(config.z_channels, config.z_channels)
            self.time_conv.append(conv)
        self.conv_in = paddle.nn.Conv2D(
            in_channels=config.z_channels, out_channels=block_in, kernel_size=3, stride=1, padding=1
        )
        self.mid = paddle.nn.Layer()
        self.mid.block_1 = Emu3VisionVQResnetBlock(
            in_channels=block_in, out_channels=block_in, dropout=config.dropout, zq_ch=zq_ch
        )
        self.mid.attn_1 = Emu3VisionVQAttnBlock(block_in, zq_ch)
        self.mid.block_2 = Emu3VisionVQResnetBlock(
            in_channels=block_in, out_channels=block_in, dropout=config.dropout, zq_ch=zq_ch
        )
        self.up = paddle.nn.LayerList()
        for i_level in reversed(range(self.num_resolutions)):
            block = paddle.nn.LayerList()
            attn = paddle.nn.LayerList()
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    Emu3VisionVQResnetBlock(
                        in_channels=block_in, out_channels=block_out, dropout=config.dropout, zq_ch=zq_ch
                    )
                )
                block_in = block_out
                if i_level in config.attn_resolutions:
                    attn.append(Emu3VisionVQAttnBlock(block_in, zq_ch))
            up = paddle.nn.Layer()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Emu3VisionVQUpsample(block_in)
            # self.up.insert(0, up)
            self.up.append(up)

        self.up = self.up[::-1]

        self.act = Emu3VisionVQActivation()
        self.norm_out = Emu3VisionVQSpatialNorm(block_in, zq_ch)
        self.conv_out = paddle.nn.Conv2D(
            in_channels=block_in, out_channels=config.out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z: paddle.Tensor, zq: paddle.Tensor):
        z_zq = paddle.concat(x=(z, zq), axis=0)
        z_zq = z_zq.transpose(perm=[0, 2, 1, 3, 4])
        z_zq = self.time_res_stack(z_zq)
        for conv in self.time_conv:
            z_zq = self.act(conv(z_zq))
        z_zq = z_zq.transpose(perm=[0, 2, 1, 3, 4])
        h, zq = paddle.chunk(x=z_zq, chunks=2, axis=0)
        h = h.reshape([-1, *tuple(h.shape)[2:]])
        zq = zq.reshape([-1, *tuple(zq.shape)[2:]])
        h = self.conv_in(h)
        h = self.mid.block_1(h, zq)
        h = self.mid.attn_1(h, zq)
        h = self.mid.block_2(h, zq)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, zq)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h, zq)
        h = self.act(h)
        h = self.conv_out(h)
        return h


class Emu3VisionVQPretrainedModel(paddlenlp.transformers.PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Emu3VisionVQConfig
    base_model_prefix = "emuvideovq"
    main_input_name = "pixel_values"
    _no_split_modules = ["Emu3VisionVQResnetBlock", "Emu3VisionVQAttnBlock", "Emu3VisionVQResnetTemporalBlock"]


#     def _init_weights(self, module):
#         if isinstance(module, (paddle.nn.Conv2D, paddle.nn.Conv3D)):
# >>>>>>            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out',
#                 nonlinearity='relu')
#         elif isinstance(module, paddle.nn.Linear):
#             init_KaimingUniform = paddle.nn.initializer.KaimingUniform(
#                 negative_slope=math.sqrt(5), nonlinearity='leaky_relu')
#             init_KaimingUniform(module.weight)
#             if module.bias is not None:
# >>>>>>                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module
#                     .weight)
#                 bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#                 init_Uniform = paddle.nn.initializer.Uniform(low=-bound,
#                     high=bound)
#                 init_Uniform(module.bias)
#         elif isinstance(module, (paddle.nn.BatchNorm2D, paddle.nn.
#             BatchNorm3D, paddle.nn.GroupNorm)):
#             init_Constant = paddle.nn.initializer.Constant(value=1)
#             init_Constant(module.weight)
#             init_Constant = paddle.nn.initializer.Constant(value=0)
#             init_Constant(module.bias)


class Emu3VisionVQModel(Emu3VisionVQPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = Emu3VisionVQEncoder(config)
        self.decoder = Emu3VisionVQDecoder(config)
        self.quantize = Emu3VisionVQVectorQuantizer(config)
        self.quant_conv = Emu3VisionVQCausalConv3d(config.z_channels, config.embed_dim)
        self.post_quant_conv = Emu3VisionVQCausalConv3d(config.embed_dim, config.z_channels)
        self.spatial_scale_factor = 2 ** (len(config.ch_mult) - 1)
        # self.post_init()

    def encode(self, x: paddle.Tensor):
        ndim = x.ndim
        if ndim == 4:
            t = self.config.temporal_downsample_factor
            b, c, h, w = tuple(x.shape)
            x = x.unsqueeze(axis=1).tile(repeat_times=[1, t, 1, 1, 1])
        elif ndim == 5:
            b, t, c, h, w = tuple(x.shape)
        h = self.encoder(x)
        h = h.transpose(perm=[0, 2, 1, 3, 4])
        h = self.quant_conv(h)
        h = h.transpose(perm=[0, 2, 1, 3, 4])
        codes = self.quantize(h)
        if ndim == 4:
            codes = codes.squeeze(axis=1)
        return codes

    def decode(self, x: paddle.Tensor):
        ndim = x.ndim
        if ndim == 3:
            x = x.unsqueeze(axis=1)
        b, t, h, w = tuple(x.shape)
        quant = self.quantize.embedding(x.flatten())
        c = tuple(quant.shape)[-1]
        quant = quant.reshape([b, t, h, w, c]).transpose(perm=[0, 4, 1, 2, 3])
        quant2 = self.post_quant_conv(quant)
        quant = quant.transpose(perm=[0, 2, 1, 3, 4])
        quant2 = quant2.transpose(perm=[0, 2, 1, 3, 4])
        video = self.decoder(quant2, quant)
        video = video.reshape(
            [
                b,
                t * self.config.temporal_downsample_factor,
                self.config.out_channels,
                h * self.spatial_scale_factor,
                w * self.spatial_scale_factor,
            ]
        )
        if ndim == 3:
            return video[:, 0]
        return video

    # @property
    # def device(self):
    #     return next(self.parameters()).place

    @property
    def dtype(self):
        return self.parameters()[0].dtype
