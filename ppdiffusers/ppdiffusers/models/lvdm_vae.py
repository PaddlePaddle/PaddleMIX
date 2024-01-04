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

from dataclasses import dataclass

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .lvdm_aemodules3d import Decoder, Encoder, SamePadConv3d
from .lvdm_distributions import DiagonalGaussianDistribution
from .modeling_utils import ModelMixin
from .vae import DecoderOutput


def conv3d(in_channels, out_channels, kernel_size, conv3d_type="SamePadConv3d"):
    if conv3d_type == "SamePadConv3d":
        return SamePadConv3d(in_channels, out_channels, kernel_size=kernel_size, padding_type="replicate")
    else:
        raise NotImplementedError


@dataclass
class AutoencoderKLOutput(BaseOutput):
    """
    Output of AutoencoderKL encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent_dist: DiagonalGaussianDistribution


class LVDMAutoencoderKL(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        n_hiddens=32,
        downsample=[4, 8, 8],
        z_channels=4,
        double_z=True,
        image_channel=3,
        norm_type="group",
        padding_type="replicate",
        upsample=[4, 8, 8],
        embed_dim=4,
        # ckpt_path=None,
        # ignore_keys=[],
        image_key="image",
        monitor=None,
        std=1.0,
        mean=0.0,
        prob=0.2,
    ):
        super().__init__()
        self.image_key = image_key
        # pass init params to Encoder
        self.encoder = Encoder(
            n_hiddens=n_hiddens,
            downsample=downsample,
            z_channels=z_channels,
            double_z=double_z,
            image_channel=image_channel,
            norm_type=norm_type,
            padding_type=padding_type,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            n_hiddens=n_hiddens,
            upsample=upsample,
            z_channels=z_channels,
            image_channel=image_channel,
            norm_type="group",
        )

        self.quant_conv = conv3d(2 * z_channels, 2 * embed_dim, 1)
        self.post_quant_conv = conv3d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim
        self.std = std
        self.mean = mean
        self.prob = prob
        if monitor is not None:
            self.monitor = monitor

    def encode(self, x, **kwargs):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        # return posterior
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z, **kwargs):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        # return dec
        return DecoderOutput(sample=dec)

    def forward(self, input, sample_posterior=True, **kwargs):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        # return dec, posterior
        return DecoderOutput(sample=dec)
