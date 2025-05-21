# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


def nonlinearity(x):
    return x * paddle.nn.functional.sigmoid(x=x)


def Normalize(in_channels, num_groups=32):
    return paddle.nn.GroupNorm(
        num_groups=num_groups,
        num_channels=in_channels,
        epsilon=1e-06,
        weight_attr=True,
        bias_attr=True,
    )


class Upsample(paddle.nn.Layer):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )

    def forward(self, x):
        x = paddle.nn.functional.interpolate(x=x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(paddle.nn.Layer):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = paddle.nn.Conv2D(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=2,
                padding=0,
            )

    def forward(self, x):
        if self.with_conv:
            pad = 0, 1, 0, 1
            x = paddle.nn.functional.pad(x=x, pad=pad, mode="constant", value=0, pad_from_left_axis=False)
            x = self.conv(x)
        else:
            x = paddle.nn.functional.avg_pool2d(x=x, kernel_size=2, stride=2, exclusive=False)
        return x


class ResnetBlock(paddle.nn.Layer):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if temb_channels > 0:
            self.temb_proj = paddle.nn.Linear(in_features=temb_channels, out_features=out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.conv2 = paddle.nn.Conv2D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = paddle.nn.Conv2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                self.nin_shortcut = paddle.nn.Conv2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(paddle.nn.Layer):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.k = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.v = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.proj_out = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = tuple(q.shape)
        q = q.reshape((b, c, h * w))
        q = q.transpose(perm=[0, 2, 1])
        k = k.reshape((b, c, h * w))
        w_ = paddle.bmm(x=q, y=k)
        w_ = w_ * int(c) ** -0.5
        w_ = paddle.nn.functional.softmax(x=w_, axis=2)
        v = v.reshape((b, c, h * w))
        w_ = w_.transpose(perm=[0, 2, 1])
        h_ = paddle.bmm(x=v, y=w_)
        h_ = h_.reshape((b, c, h, w))
        h_ = self.proj_out(h_)
        return x + h_


class Encoder(paddle.nn.Layer):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=self.ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = paddle.nn.LayerList()
        for i_level in range(self.num_resolutions):
            block = paddle.nn.LayerList()
            attn = paddle.nn.LayerList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = paddle.nn.Layer()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
        self.mid = paddle.nn.Layer()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.norm_out = Normalize(block_in)
        self.conv_out = paddle.nn.Conv2D(
            in_channels=block_in,
            out_channels=2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
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
        return h


class Decoder(paddle.nn.Layer):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        give_pre_end=False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = 1, z_channels, curr_res, curr_res
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))
        self.conv_in = paddle.nn.Conv2D(
            in_channels=z_channels,
            out_channels=block_in,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.mid = paddle.nn.Layer()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.up = paddle.nn.LayerList()
        for i_level in reversed(range(self.num_resolutions)):
            block = paddle.nn.LayerList()
            attn = paddle.nn.LayerList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = paddle.nn.Layer()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.append(up)
        self.up = self.up[::-1]
        self.norm_out = Normalize(block_in)
        self.conv_out = paddle.nn.Conv2D(
            in_channels=block_in,
            out_channels=out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, z):
        self.last_z_shape = tuple(z.shape)
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = paddle.chunk(x=parameters, chunks=2, axis=1)
        self.logvar = paddle.clip(x=self.logvar, min=-30.0, max=20.0)
        self.deterministic = deterministic
        self.std = paddle.exp(x=0.5 * self.logvar)
        self.var = paddle.exp(x=self.logvar)
        if self.deterministic:
            self.var = self.std = paddle.zeros_like(x=self.mean).to(device=self.parameters.place)

    def sample(self):
        x = self.mean + self.std * paddle.randn(shape=tuple(self.mean.shape)).to(device=self.parameters.place)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return paddle.to_tensor(data=[0.0], dtype="float32")
        elif other is None:
            return 0.5 * paddle.sum(
                x=paddle.pow(x=self.mean, y=2) + self.var - 1.0 - self.logvar,
                axis=[1, 2, 3],
            )
        else:
            return 0.5 * paddle.sum(
                x=paddle.pow(x=self.mean - other.mean, y=2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                axis=[1, 2, 3],
            )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return paddle.to_tensor(data=[0.0], dtype="float32")
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * paddle.sum(
            x=logtwopi + self.logvar + paddle.pow(x=sample - self.mean, y=2) / self.var,
            axis=dims,
        )

    def mode(self):
        return self.mean


class AutoencoderKL(paddle.nn.Layer):
    def __init__(self, embed_dim, ch_mult, use_variational=True, ckpt_path=None):
        super().__init__()
        self.encoder = Encoder(ch_mult=ch_mult, z_channels=embed_dim)
        self.decoder = Decoder(ch_mult=ch_mult, z_channels=embed_dim)
        self.use_variational = use_variational
        mult = 2 if self.use_variational else 1
        self.quant_conv = paddle.nn.Conv2D(in_channels=2 * embed_dim, out_channels=mult * embed_dim, kernel_size=1)
        self.post_quant_conv = paddle.nn.Conv2D(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1)
        self.embed_dim = embed_dim
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)

    def init_from_ckpt(self, path):

        sd = paddle.load(path)
        msg = self.set_state_dict(state_dict=sd, use_structured_name=True)
        print("Loading pre-trained KL-VAE")
        print("Missing keys:")
        print(msg[0])
        print("Unexpected keys:")
        print(msg[1])
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        if not self.use_variational:
            moments = paddle.concat(x=(moments, paddle.ones_like(x=moments)), axis=1)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        # z = z.sample()
        # z = z.mode()
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, inputs, disable=True, train=True, optimizer_idx=0):
        if train:
            return self.training_step(inputs, disable, optimizer_idx)
        else:
            return self.validation_step(inputs, disable)
