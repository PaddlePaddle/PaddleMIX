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

from .constants import N_MELS


class ConvBlockRes(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=out_channels, momentum=1 - momentum),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=out_channels, momentum=1 - momentum),
            paddle.nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = paddle.nn.Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

    def forward(self, x):
        if self.is_shortcut:
            return self.conv(x) + self.shortcut(x)
        else:
            return self.conv(x) + x


class ResEncoderBlock(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = paddle.nn.LayerList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = paddle.nn.AvgPool2D(kernel_size=kernel_size, exclusive=False)

    def forward(self, x):
        for i in range(self.n_blocks):
            x = self.conv[i](x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class ResDecoderBlock(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = paddle.nn.Sequential(
            paddle.nn.Conv2DTranspose(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=out_channels, momentum=1 - momentum),
            paddle.nn.ReLU(),
        )
        self.conv2 = paddle.nn.LayerList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = paddle.concat(x=(x, concat_tensor), axis=1)
        for i in range(self.n_blocks):
            x = self.conv2[i](x)
        return x


class Encoder(paddle.nn.Layer):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = paddle.nn.BatchNorm2D(num_features=in_channels, momentum=1 - momentum)
        self.layers = paddle.nn.LayerList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for i in range(self.n_encoders):
            _, x = self.layers[i](x)
            concat_tensors.append(_)
        return x, concat_tensors


class Intermediate(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = paddle.nn.LayerList()
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        for i in range(self.n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for i in range(self.n_inters):
            x = self.layers[i](x)
        return x


class Decoder(paddle.nn.Layer):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()
        self.layers = paddle.nn.LayerList()
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for i in range(self.n_decoders):
            x = self.layers[i](x, concat_tensors[-1 - i])
        return x


class TimbreFilter(paddle.nn.Layer):
    def __init__(self, latent_rep_channels):
        super(TimbreFilter, self).__init__()
        self.layers = paddle.nn.LayerList()
        for latent_rep in latent_rep_channels:
            self.layers.append(ConvBlockRes(latent_rep[0], latent_rep[0]))

    def forward(self, x_tensors):
        out_tensors = []
        for i, layer in enumerate(self.layers):
            out_tensors.append(layer(x_tensors[i]))
        return out_tensors


class DeepUnet0(paddle.nn.Layer):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super(DeepUnet0, self).__init__()
        self.encoder = Encoder(in_channels, N_MELS, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks
        )
        self.tf = TimbreFilter(self.encoder.latent_channels)
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x
