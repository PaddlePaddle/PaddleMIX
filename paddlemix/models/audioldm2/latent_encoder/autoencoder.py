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

import numpy as np
import paddle

from ppdiffusers import AutoencoderKL

from ..hifigan.model import get_vocoder


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = paddle.chunk(parameters, 2, axis=1)
        self.logvar = paddle.clip(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = paddle.exp(0.5 * self.logvar)
        self.var = paddle.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = paddle.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * paddle.randn(self.mean.shape)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return paddle.to_tensor([0.0])
        else:
            if other is None:
                return 0.5 * paddle.mean(
                    paddle.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * paddle.mean(
                    paddle.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return paddle.to_tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * paddle.sum(
            logtwopi + self.logvar + paddle.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class AudioLDMAutoencoderKL(AutoencoderKL):
    def __init__(
        self,
        ddconfig=None,
        lossconfig=None,
        batchsize=None,
        embed_dim=None,
        time_shuffle=1,
        subband=1,
        sampling_rate=16000,
        reload_from_ckpt=None,
        ignore_keys=[],
        image_key="fbank",
        colorize_nlabels=None,
        monitor=None,
        base_learning_rate=1e-5,
    ):
        super().__init__(
            in_channels=ddconfig["in_channels"],
            out_channels=ddconfig["out_ch"],
            down_block_types=("DownEncoderBlock2D",) * len(ddconfig["ch_mult"]),
            up_block_types=("UpDecoderBlock2D",) * len(ddconfig["ch_mult"]),
            block_out_channels=tuple([ddconfig["ch"] * i for i in ddconfig["ch_mult"]]),
            layers_per_block=ddconfig["num_res_blocks"],
            latent_channels=ddconfig["z_channels"],
        )
        self.automatic_optimization = False
        assert "mel_bins" in ddconfig.keys(), "mel_bins is not specified in the Autoencoder config"
        num_mel = ddconfig["mel_bins"]
        self.image_key = image_key
        self.sampling_rate = sampling_rate

        self.loss = None
        self.subband = int(subband)

        if self.subband > 1:
            print("Use subband decomposition %s" % self.subband)

        if self.image_key == "fbank":
            self.vocoder = get_vocoder(None, num_mel)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", paddle.randn([3, colorize_nlabels, 1, 1]))
        if monitor is not None:
            self.monitor = monitor
        self.learning_rate = float(base_learning_rate)
        # print("Initial learning rate %s" % self.learning_rate)

        self.time_shuffle = time_shuffle
        self.reload_from_ckpt = reload_from_ckpt
        self.reloaded = False
        self.mean, self.std = None, None

        self.feature_cache = None
        self.flag_first_run = True
        self.train_step = 0

        self.logger_save_dir = None
        self.logger_exp_name = None

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
