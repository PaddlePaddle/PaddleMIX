# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import Optional

import numpy as np
import paddle
import paddle.nn as nn
from paddle.distributed.fleet.utils import recompute

from ..utils import BaseOutput, randn_tensor
from .attention_processor import SpatialNorm
from .unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block

try:
    from paddle.amp.auto_cast import amp_state
except ImportError:
    from paddle.fluid.dygraph.amp.auto_cast import amp_state


def finfo(dtype):
    if dtype == paddle.float32:
        return np.finfo(np.float32)
    if dtype == paddle.float16:
        return np.finfo(np.float16)
    if dtype == paddle.float64:
        return np.finfo(np.float64)


@dataclass
class DecoderOutput(BaseOutput):
    """
    Output of decoding method.

    Args:
        sample (`paddle.Tensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: paddle.Tensor


class Encoder(nn.Layer):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        double_z=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2D(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        self.down_blocks = nn.LayerList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, epsilon=1e-6
        )
        self.conv_act = nn.Silu()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2D(block_out_channels[-1], conv_out_channels, 3, padding=1)
        self.gradient_checkpointing = False

    def forward(self, x):
        sample = x
        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing and not sample.stop_gradient:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            for down_block in self.down_blocks:
                sample = recompute(create_custom_forward(down_block), sample)

            # middle
            sample = recompute(create_custom_forward(self.mid_block), sample)

        else:
            # down
            for down_block in self.down_blocks:
                sample = down_block(sample)

            # middle
            sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class Decoder(nn.Layer):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        norm_type="group",  # group, spatial
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2D(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        self.up_blocks = nn.LayerList([])
        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, epsilon=1e-6
            )
        self.conv_act = nn.Silu()
        self.conv_out = nn.Conv2D(block_out_channels[0], out_channels, 3, padding=1)
        self.gradient_checkpointing = False

    def forward(self, z, latent_embeds=None):
        sample = z
        sample = self.conv_in(sample)

        upscale_dtype = self.up_blocks.dtype
        if self.training and self.gradient_checkpointing and not sample.stop_gradient:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # middle
            sample = recompute(create_custom_forward(self.mid_block), sample, latent_embeds)
            if upscale_dtype != sample.dtype:
                sample = sample.cast(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = recompute(create_custom_forward(up_block), sample, latent_embeds)
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            if upscale_dtype != sample.dtype:
                sample = sample.cast(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = up_block(sample, latent_embeds)

        # (TODO, junnyu) check nan
        # clamp inf values to enable fp16 training
        if (amp_state() or sample.dtype == paddle.float16) and paddle.isinf(sample).any():
            clamp_value = finfo(sample.dtype).max - 1000
            sample = paddle.clip(sample, min=-clamp_value, max=clamp_value)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class UpSample(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deconv = nn.Conv2DTranspose(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = paddle.nn.functional.relu(x)
        x = self.deconv(x)
        return x


class MaskConditionEncoder(nn.Layer):
    """
    used in AsymmetricAutoencoderKL
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int = 192,
        res_ch: int = 768,
        stride: int = 16,
    ) -> None:
        super().__init__()

        channels = []
        while stride > 1:
            stride = stride // 2
            in_ch_ = out_ch * 2
            if out_ch > res_ch:
                out_ch = res_ch
            if stride == 1:
                in_ch_ = res_ch
            channels.append((in_ch_, out_ch))
            out_ch *= 2

        out_channels = []
        for _in_ch, _out_ch in channels:
            out_channels.append(_out_ch)
        out_channels.append(channels[-1][0])

        layers = []
        in_ch_ = in_ch
        for l in range(len(out_channels)):
            out_ch_ = out_channels[l]
            if l == 0 or l == 1:
                layers.append(nn.Conv2D(in_ch_, out_ch_, kernel_size=3, stride=1, padding=1))
            else:
                layers.append(nn.Conv2D(in_ch_, out_ch_, kernel_size=4, stride=2, padding=1))
            in_ch_ = out_ch_

        self.layers = nn.Sequential(*layers)

    def forward(self, x: paddle.Tensor, mask=None) -> paddle.Tensor:
        out = {}
        for l in range(len(self.layers)):
            layer = self.layers[l]
            x = layer(x)
            out[str(tuple(x.shape))] = x
            x = paddle.nn.functional.relu(x)
        return out


class MaskConditionDecoder(nn.Layer):
    """The `MaskConditionDecoder` should be used in combination with [`AsymmetricAutoencoderKL`] to enhance the model's
    decoder with a conditioner on the mask and masked image."""

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        norm_type="group",  # group, spatial
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2D(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.mid_block = None
        self.up_blocks = nn.LayerList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # condition encoder
        self.condition_encoder = MaskConditionEncoder(
            in_ch=out_channels,
            out_ch=block_out_channels[0],
            res_ch=block_out_channels[-1],
        )

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, epsilon=1e-6
            )
        self.conv_act = nn.Silu()
        self.conv_out = nn.Conv2D(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, z, image=None, mask=None, latent_embeds=None):
        sample = z
        sample = self.conv_in(sample)

        upscale_dtype = self.up_blocks.dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # middle
            sample = recompute(create_custom_forward(self.mid_block), sample, latent_embeds)
            sample = sample.cast(upscale_dtype)

            # condition encoder
            if image is not None and mask is not None:
                masked_image = (1 - mask) * image
                im_x = recompute(create_custom_forward(self.condition_encoder), masked_image, mask)

            # up
            for up_block in self.up_blocks:
                if image is not None and mask is not None:
                    sample_ = im_x[str(tuple(sample.shape))]
                    mask_ = nn.functional.interpolate(mask, size=sample.shape[-2:], mode="nearest")
                    sample = sample * mask_ + sample_ * (1 - mask_)
                sample = recompute(create_custom_forward(up_block), sample, latent_embeds)
            if image is not None and mask is not None:
                sample = sample * mask + im_x[str(tuple(sample.shape))] * (1 - mask)
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            sample = sample.cast(upscale_dtype)

            # condition encoder
            if image is not None and mask is not None:
                masked_image = (1 - mask) * image
                im_x = self.condition_encoder(masked_image, mask)

            # up
            for up_block in self.up_blocks:
                if image is not None and mask is not None:
                    sample_ = im_x[str(tuple(sample.shape))]
                    mask_ = nn.functional.interpolate(mask, size=sample.shape[-2:], mode="nearest")
                    sample = sample * mask_ + sample_ * (1 - mask_)
                sample = up_block(sample, latent_embeds)
            if image is not None and mask is not None:
                sample = sample * mask + im_x[str(tuple(sample.shape))] * (1 - mask)

        # post-process
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class VectorQuantizer(nn.Layer):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly avoids costly matrix
    multiplications and allows for post-hoc remapping of indices.
    """

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self, n_e, vq_embed_dim, beta, remap=None, unknown_index="random", sane_index_shape=False, legacy=True
    ):
        super().__init__()
        self.n_e = n_e
        self.vq_embed_dim = vq_embed_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(
            self.n_e, self.vq_embed_dim, weight_attr=nn.initializer.Uniform(-1.0 / self.n_e, 1.0 / self.n_e)
        )

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", paddle.to_tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape([ishape[0], -1])
        used = self.used.cast(inds.dtype)
        match = (inds[:, :, None] == used[None, None, ...]).cast("int64")
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = paddle.randint(0, self.re_embed, shape=new[unknown].shape)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape([ishape[0], -1])
        used = self.used.cast(inds.dtype)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = paddle.take_along_axis(used[None, :][inds.shape[0] * [0], :], inds, axis=1)
        return back.reshape(ishape)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.transpose([0, 2, 3, 1])
        z_flattened = z.reshape([-1, self.vq_embed_dim])
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = (
            paddle.sum(z_flattened**2, axis=1, keepdim=True)
            + paddle.sum(self.embedding.weight**2, axis=1)
            - 2 * paddle.matmul(z_flattened, self.embedding.weight, transpose_y=True)
        )

        min_encoding_indices = paddle.argmin(d, axis=1)
        z_q = self.embedding(min_encoding_indices).reshape(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            loss = self.beta * paddle.mean((z_q.detach() - z) ** 2) + paddle.mean((z_q - z.detach()) ** 2)
        else:
            loss = paddle.mean((z_q.detach() - z) ** 2) + self.beta * paddle.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = z_q.transpose([0, 3, 1, 2])

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape([z.shape[0], -1])  # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape([-1, 1])  # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape([z_q.shape[0], z_q.shape[2], z_q.shape[3]])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape([shape[0], -1])  # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(
                [
                    -1,
                ]
            )  # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.reshape(shape)
            # reshape back to match original input shape
            z_q = z_q.transpose([0, 3, 1, 2])

        return z_q


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = paddle.chunk(parameters, 2, axis=1)
        self.logvar = paddle.clip(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = paddle.exp(0.5 * self.logvar)
        self.var = paddle.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = paddle.zeros_like(self.mean, dtype=self.parameters.dtype)

    def sample(self, generator: Optional[paddle.Generator] = None) -> paddle.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(self.mean.shape, generator=generator, dtype=self.parameters.dtype)
        x = self.mean + self.std * sample
        return x

    def kl(self, other=None):
        if self.deterministic:
            return paddle.to_tensor([0.0])
        else:
            if other is None:
                return 0.5 * paddle.sum(paddle.pow(self.mean, 2) + self.var - 1.0 - self.logvar, axis=[1, 2, 3])
            else:
                return 0.5 * paddle.sum(
                    paddle.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    axis=[1, 2, 3],
                )

    def nll(self, sample, axis=[1, 2, 3]):
        if self.deterministic:
            return paddle.to_tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * paddle.sum(logtwopi + self.logvar + paddle.pow(sample - self.mean, 2) / self.var, axis=axis)

    def mode(self):
        return self.mean
