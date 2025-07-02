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
from dataclasses import dataclass, field
from typing import List

import numpy as np
import paddle

from ppdiffusers.configuration_utils import ConfigMixin
from ppdiffusers.models.modeling_utils import ModelMixin  # , register_to_config

from .common_modules import (
    AttnBlock,
    Downsample,
    Normalize,
    ResnetBlock,
    Upsample,
    nonlinearity,
)


class Updateable:
    def do_update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        for attr in self.__dir__():
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except:
                continue
            if isinstance(module, Updateable):
                module.do_update_step(epoch, global_step, on_load_weights=on_load_weights)
        self.update_step(epoch, global_step, on_load_weights=on_load_weights)

    def do_update_step_end(self, epoch: int, global_step: int):
        for attr in self.__dir__():
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except:
                continue
            if isinstance(module, Updateable):
                module.do_update_step_end(epoch, global_step)
        self.update_step_end(epoch, global_step)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        pass

    def update_step_end(self, epoch: int, global_step: int):
        pass


class VQGANEncoder(ModelMixin, ConfigMixin):
    @dataclass
    class Config:
        ch: int = 128
        ch_mult: List[int] = field(default_factory=lambda: [1, 2, 2, 4, 4])
        num_res_blocks: List[int] = field(default_factory=lambda: [4, 3, 4, 3, 4])
        attn_resolutions: List[int] = field(default_factory=lambda: [5])
        dropout: float = 0.0
        in_ch: int = 3
        out_ch: int = 3
        resolution: int = 256
        z_channels: int = 13
        double_z: bool = False

    def __init__(
        self,
        ch: int = 128,
        ch_mult: List[int] = [1, 2, 2, 4, 4],
        num_res_blocks: List[int] = [4, 3, 4, 3, 4],
        attn_resolutions: List[int] = [5],
        dropout: float = 0.0,
        in_ch: int = 3,
        out_ch: int = 3,
        resolution: int = 256,
        z_channels: int = 13,
        double_z: bool = False,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_ch = in_ch
        self.conv_in = paddle.nn.Conv2D(
            in_channels=self.in_ch, out_channels=self.ch, kernel_size=3, stride=1, padding=1
        )
        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = paddle.nn.LayerList()
        for i_level in range(self.num_resolutions):
            block = paddle.nn.LayerList()
            attn = paddle.nn.LayerList()
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = paddle.nn.Layer()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, True)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = paddle.nn.Layer()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.norm_out = Normalize(block_in)
        self.conv_out = paddle.nn.Conv2D(
            in_channels=block_in,
            out_channels=2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.quant_conv = paddle.nn.Conv2D(in_channels=z_channels, out_channels=z_channels, kernel_size=1)

    def forward(self, x):
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class LFQuantizer(paddle.nn.Layer):
    def __init__(
        self,
        num_codebook_entry: int = -1,
        codebook_dim: int = 13,
        beta: float = 0.25,
        entropy_multiplier: float = 0.1,
        commit_loss_multiplier: float = 0.1,
    ):
        super().__init__()
        self.codebook_size = 2**codebook_dim
        print(f"Look-up free quantizer with codebook size: {self.codebook_size}")
        self.e_dim = codebook_dim
        self.beta = beta

        indices = paddle.arange(end=self.codebook_size)

        binary = (
            indices.unsqueeze(1).numpy() >> paddle.arange(codebook_dim - 1, -1, -1, dtype=paddle.int64).numpy()
        ) & 1
        binary = paddle.to_tensor(binary)
        # binary = indices.unsqueeze(axis=1) >> paddle.arange(start=
        #     codebook_dim - 1, end=-1, step=-1, dtype='int64') & 1

        embedding = binary.astype(dtype="float32") * 2 - 1
        self.register_buffer(name="embedding", tensor=embedding)
        self.register_buffer(name="power_vals", tensor=2 ** paddle.arange(start=codebook_dim - 1, end=-1, step=-1))
        self.commit_loss_multiplier = commit_loss_multiplier
        self.entropy_multiplier = entropy_multiplier

    def get_indices(self, z_q):
        return (self.power_vals.reshape([1, -1, 1, 1]) * (z_q > 0).astype(dtype="int64")).sum(axis=1, keepdim=True)

    def get_codebook_entry(self, indices, shape=None):
        if shape is None:
            h, w = int(math.sqrt(indices.shape[-1])), int(math.sqrt(indices.shape[-1]))
        else:
            h, w = shape
        b, _ = tuple(indices.shape)
        indices = indices.reshape([-1])
        z_q = self.embedding[indices]
        z_q = z_q.reshape([b, h, w, -1])
        z_q = z_q.transpose(perm=[0, 3, 1, 2])
        return z_q

    def forward(self, z, get_code=False):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        if get_code:
            return self.get_codebook_entry(z)
        z = z.transpose(perm=[0, 2, 3, 1])
        z_flattened = z.reshape([-1, self.e_dim])
        ge_zero = (z_flattened > 0).astype(dtype="float32")
        ones = paddle.ones_like(x=z_flattened)
        z_q = ones * ge_zero + -ones * (1 - ge_zero)
        z_q = z_flattened + (z_q - z_flattened).detach()

        CatDist = paddle.distribution.Categorical
        logit = paddle.stack(
            [
                -(z_flattened - paddle.ones_like(z_q)).pow(2),
                -(z_flattened - paddle.ones_like(z_q) * -1).pow(2),
            ],
            axis=-1,
        )
        cat_dist = CatDist(logits=logit)
        entropy = cat_dist.entropy().mean()
        # mean_prob = cat_dist.probs.mean(0)
        # mean_entropy = CatDist(probs=mean_prob).entropy().mean() # TODO
        mean_entropy = CatDist(logits=logit).entropy().mean()

        # compute loss for embedding
        commit_loss = paddle.mean((z_q.detach() - z_flattened) ** 2) + self.beta * paddle.mean(
            (z_q - z_flattened.detach()) ** 2
        )

        # reshape back to match original input shape
        z_q = z_q.reshape(tuple(z.shape))
        z_q = z_q.transpose(perm=[0, 3, 1, 2])

        return {
            "z": z_q,
            "quantizer_loss": commit_loss * self.commit_loss_multiplier,
            "entropy_loss": (entropy - mean_entropy) * self.entropy_multiplier,
            "indices": self.get_indices(z_q),
        }


class VQGANDecoder(ModelMixin, ConfigMixin):
    def __init__(
        self,
        ch: int = 128,
        ch_mult: List[int] = [1, 1, 2, 2, 4],
        num_res_blocks: List[int] = [4, 4, 3, 4, 3],
        attn_resolutions: List[int] = [5],
        dropout: float = 0.0,
        in_ch: int = 3,
        out_ch: int = 3,
        resolution: int = 256,
        z_channels: int = 13,
        double_z: bool = False,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_ch = in_ch
        self.give_pre_end = False
        self.z_channels = z_channels
        # in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = self.resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = 1, z_channels, curr_res, curr_res
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))
        self.conv_in = paddle.nn.Conv2D(
            in_channels=z_channels, out_channels=block_in, kernel_size=3, stride=1, padding=1
        )
        self.mid = paddle.nn.Layer()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout
        )
        self.up = paddle.nn.LayerList()
        for i_level in reversed(range(self.num_resolutions)):
            block = paddle.nn.LayerList()
            attn = paddle.nn.LayerList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = paddle.nn.Layer()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, True)
                curr_res = curr_res * 2
            # self.up.insert(0, up)
            self.up.append(up)
        self.up = self.up[::-1]
        self.norm_out = Normalize(block_in)
        self.conv_out = paddle.nn.Conv2D(in_channels=block_in, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.post_quant_conv = paddle.nn.Conv2D(in_channels=z_channels, out_channels=z_channels, kernel_size=1)

    def forward(self, z):
        self.last_z_shape = tuple(z.shape)
        temb = None
        output = dict()
        z = self.post_quant_conv(z)
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks[i_level]):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        output["output"] = h
        if self.give_pre_end:
            return output
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        output["output"] = h
        return output


class MAGVITv2(ModelMixin, ConfigMixin):

    # @register_to_config
    def __init__(self):
        super().__init__()
        self.encoder = VQGANEncoder()
        self.decoder = VQGANDecoder()
        self.quantize = LFQuantizer()

    def forward(self, pixel_values, return_loss=False):
        pass

    def encode(self, pixel_values, return_loss=False):
        hidden_states = self.encoder(pixel_values)
        quantized_states = self.quantize(hidden_states)["z"]
        codebook_indices = self.quantize.get_indices(quantized_states).reshape(
            [pixel_values.shape[0], -1],
        )
        output = quantized_states, codebook_indices
        return output

    def get_code(self, pixel_values):
        hidden_states = self.encoder(pixel_values)
        codebook_indices = self.quantize.get_indices(self.quantize(hidden_states)["z"]).reshape(
            [pixel_values.shape[0], -1]
        )
        return codebook_indices

    def decode_code(self, codebook_indices, shape=None):
        z_q = self.quantize.get_codebook_entry(codebook_indices, shape=shape)
        reconstructed_pixel_values = self.decoder(z_q)["output"]
        return reconstructed_pixel_values
