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
import paddle.nn.functional as F
from paddle.nn import Conv1D, Conv1DTranspose
from paddle.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def get_vocoder_config():
    return {
        "resblock": "1",
        "num_gpus": 6,
        "batch_size": 16,
        "learning_rate": 0.0002,
        "adam_b1": 0.8,
        "adam_b2": 0.99,
        "lr_decay": 0.999,
        "seed": 1234,
        "upsample_rates": [5, 4, 2, 2, 2],
        "upsample_kernel_sizes": [16, 16, 8, 4, 4],
        "upsample_initial_channel": 1024,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "segment_size": 8192,
        "num_mels": 64,
        "num_freq": 1025,
        "n_fft": 1024,
        "hop_size": 160,
        "win_size": 1024,
        "sampling_rate": 16000,
        "fmin": 0,
        "fmax": 8000,
        "fmax_for_loss": None,
        "num_workers": 4,
        "dist_config": {
            "dist_backend": "nccl",
            "dist_url": "tcp://localhost:54321",
            "world_size": 1,
        },
    }


def get_vocoder_config_48k():
    return {
        "resblock": "1",
        "num_gpus": 8,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "adam_b1": 0.8,
        "adam_b2": 0.99,
        "lr_decay": 0.999,
        "seed": 1234,
        "upsample_rates": [6, 5, 4, 2, 2],
        "upsample_kernel_sizes": [12, 10, 8, 4, 4],
        "upsample_initial_channel": 1536,
        "resblock_kernel_sizes": [3, 7, 11, 15],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "segment_size": 15360,
        "num_mels": 256,
        "n_fft": 2048,
        "hop_size": 480,
        "win_size": 2048,
        "sampling_rate": 48000,
        "fmin": 20,
        "fmax": 24000,
        "fmax_for_loss": None,
        "num_workers": 8,
        "dist_config": {"dist_backend": "nccl", "dist_url": "tcp://localhost:18273", "world_size": 1},
    }


class ResBlock(nn.Layer):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.h = h
        weight_attr1 = paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=0.01))
        weight_attr2 = paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=0.01))
        weight_attr3 = paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=0.01))
        self.convs1 = nn.LayerList(
            [
                weight_norm(
                    Conv1D(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        padding=get_padding(kernel_size, dilation[0]),
                        dilation=dilation[0],
                        weight_attr=weight_attr1,
                    )
                ),
                weight_norm(
                    Conv1D(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        padding=get_padding(kernel_size, dilation[1]),
                        dilation=dilation[1],
                        weight_attr=weight_attr2,
                    )
                ),
                weight_norm(
                    Conv1D(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        padding=get_padding(kernel_size, dilation[2]),
                        dilation=dilation[2],
                        weight_attr=weight_attr3,
                    )
                ),
            ]
        )

        weight_attr4 = paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=0.01))
        weight_attr5 = paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=0.01))
        weight_attr6 = paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=0.01))
        self.convs2 = nn.LayerList(
            [
                weight_norm(
                    Conv1D(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        padding=get_padding(kernel_size, 1),
                        dilation=1,
                        weight_attr=weight_attr4,
                    )
                ),
                weight_norm(
                    Conv1D(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        padding=get_padding(kernel_size, 1),
                        dilation=1,
                        weight_attr=weight_attr5,
                    )
                ),
                weight_norm(
                    Conv1D(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        padding=get_padding(kernel_size, 1),
                        dilation=1,
                        weight_attr=weight_attr6,
                    )
                ),
            ]
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Generator(nn.Layer):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1D(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock

        self.ups = nn.LayerList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            weight_attr_tmp = paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=0.01))
            self.ups.append(
                weight_norm(
                    Conv1DTranspose(
                        h.upsample_initial_channel // (2**i),
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                        weight_attr=weight_attr_tmp,
                    )
                )
            )

        self.resblocks = nn.LayerList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(mean=0.0, std=0.01))
        self.conv_post = weight_norm(Conv1D(ch, 1, 7, 1, padding=3, weight_attr=weight_attr))

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = paddle.tanh(x)

        return x

    def remove_weight_norm(self):
        # print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def get_vocoder(config, mel_bins):
    if mel_bins == 64:
        config = get_vocoder_config()
        config = AttrDict(config)
        vocoder = Generator(config)
        vocoder.eval()
        vocoder.remove_weight_norm()
    else:
        config = get_vocoder_config_48k()
        config = AttrDict(config)
        vocoder = Generator(config)
        vocoder.eval()
        vocoder.remove_weight_norm()

    return vocoder


def vocoder_infer(mels, vocoder, lengths=None):
    with paddle.no_grad():
        wavs = vocoder(mels).squeeze(1)

    wavs = (wavs.numpy() * 32768).astype("int16")

    if lengths is not None:
        wavs = wavs[:, :lengths]

    return wavs


def synth_one_sample(mel_input, mel_prediction, labels, vocoder):
    if vocoder is not None:

        wav_reconstruction = vocoder_infer(
            mel_input.transpose([0, 2, 1]),
            vocoder,
        )
        wav_prediction = vocoder_infer(
            mel_prediction.transpose([0, 2, 1]),
            vocoder,
        )
    else:
        wav_reconstruction = wav_prediction = None

    return wav_reconstruction, wav_prediction
