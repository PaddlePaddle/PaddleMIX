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

import functools
from typing import Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn as nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import logging
from ..utils.accelerate_utils import apply_forward_hook
from .activations import get_activation
from .attention_processor import Attention, MochiVaeAttnProcessor2_0
from .autoencoder_kl_cogvideox import CogVideoXCausalConv3d
from .modeling_outputs import AutoencoderKLOutput
from .modeling_utils import ModelMixin
from .vae import DecoderOutput, DiagonalGaussianDistribution

logger = logging.get_logger(__name__)


class MochiChunkedGroupNorm3D(nn.Layer):
    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        affine: bool = True,
        chunk_size: int = 8,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(
            num_channels=num_channels, num_groups=num_groups, weight_attr=affine, bias_attr=affine
        )
        self.chunk_size = chunk_size

    def forward(self, x: paddle.Tensor = None) -> paddle.Tensor:
        batch_size = x.shape[0]

        x = x.transpose([0, 2, 1, 3, 4]).flatten(0, 1)

        if x.shape[0] <= self.chunk_size:
            output = self.norm_layer(x)
        elif x.shape[0] % self.chunk_size == 0:
            output = paddle.concat(
                [self.norm_layer(chunk) for chunk in paddle.split(x, num_or_sections=self.chunk_size, axis=0)], axis=0
            )
        else:
            chunks = []
            num_full_chunks = x.shape[0] // self.chunk_size

            # Process the evenly divisible portion
            for i in range(num_full_chunks):
                start_idx = i * self.chunk_size
                end_idx = start_idx + self.chunk_size
                chunk = x[start_idx:end_idx]
                chunks.append(self.norm_layer(chunk))

            # Process the remaining portion
            remainder = x.shape[0] % self.chunk_size
            if remainder > 0:
                last_chunk = x[-remainder:]
                chunks.append(self.norm_layer(last_chunk))

            output = paddle.concat(chunks, axis=0)

        output = output.reshape([batch_size, -1] + list(output.shape[1:])).transpose([0, 2, 1, 3, 4])

        return output


class MochiResnetBlock3D(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        act_fn: str = "swish",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = get_activation(act_fn)

        self.norm1 = MochiChunkedGroupNorm3D(num_channels=in_channels)
        self.conv1 = CogVideoXCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            pad_mode="replicate",
            is_mochi=True,
        )
        self.norm2 = MochiChunkedGroupNorm3D(num_channels=out_channels)
        self.conv2 = CogVideoXCausalConv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            pad_mode="replicate",
            is_mochi=True,
        )

    def forward(
        self,
        inputs: paddle.Tensor,
    ) -> paddle.Tensor:

        hidden_states = inputs

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        hidden_states = hidden_states + inputs
        return hidden_states


class MochiDownBlock3D(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        temporal_expansion: int = 2,
        spatial_expansion: int = 2,
        add_attention: bool = True,
    ):
        super().__init__()
        self.temporal_expansion = temporal_expansion
        self.spatial_expansion = spatial_expansion

        self.conv_in = CogVideoXCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(temporal_expansion, spatial_expansion, spatial_expansion),
            stride=(temporal_expansion, spatial_expansion, spatial_expansion),
            pad_mode="replicate",
            is_mochi=True,
        )

        resnets = []
        norms = []
        attentions = []
        for _ in range(num_layers):
            resnets.append(MochiResnetBlock3D(in_channels=out_channels))
            if add_attention:
                norms.append(MochiChunkedGroupNorm3D(num_channels=out_channels))
                attentions.append(
                    Attention(
                        query_dim=out_channels,
                        heads=out_channels // 32,
                        dim_head=32,
                        qk_norm="l2",
                        is_causal=True,
                        processor=MochiVaeAttnProcessor2_0(),
                    )
                )
            else:
                norms.append(None)
                attentions.append(None)

        self.resnets = nn.LayerList(resnets)
        self.norms = nn.LayerList(norms)
        self.attentions = nn.LayerList(attentions)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: paddle.Tensor,
        chunk_size: int = 2**15,
    ) -> paddle.Tensor:

        hidden_states = self.conv_in(hidden_states)

        for i, (resnet, norm, attn) in enumerate(zip(self.resnets, self.norms, self.attentions)):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states = paddle.distributed.fleet.utils.recompute(
                    create_custom_forward(resnet),
                    hidden_states,
                )
            else:
                hidden_states = resnet(hidden_states)

            if attn is not None:
                residual = hidden_states
                hidden_states = norm(hidden_states)

                batch_size, num_channels, num_frames, height, width = hidden_states.shape
                hidden_states = hidden_states.transpose([0, 3, 4, 2, 1]).flatten(0, 2)

                if hidden_states.shape[0] <= chunk_size:
                    hidden_states = attn(hidden_states)
                else:
                    hidden_states_chunks = []
                    for i in range(0, hidden_states.shape[0], chunk_size):
                        hidden_states_chunk = hidden_states[i : i + chunk_size]
                        hidden_states_chunk = attn(hidden_states_chunk)
                        hidden_states_chunks.append(hidden_states_chunk)
                    hidden_states = paddle.concat(hidden_states_chunks)

                hidden_states = hidden_states.reshape([batch_size, height, width, num_frames, num_channels]).transpose(
                    [0, 4, 3, 1, 2]
                )

                hidden_states = residual + hidden_states

        return hidden_states


class MochiMidBlock3D(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 3,
        add_attention: bool = True,
    ):
        super().__init__()

        resnets = []
        norms = []
        attentions = []

        for _ in range(num_layers):
            resnets.append(MochiResnetBlock3D(in_channels=in_channels))

            if add_attention:
                norms.append(MochiChunkedGroupNorm3D(num_channels=in_channels))
                attentions.append(
                    Attention(
                        query_dim=in_channels,
                        heads=in_channels // 32,
                        dim_head=32,
                        qk_norm="l2",
                        is_causal=True,
                        processor=MochiVaeAttnProcessor2_0(),
                    )
                )
            else:
                norms.append(None)
                attentions.append(None)

        self.resnets = nn.LayerList(resnets)
        self.norms = nn.LayerList(norms)
        self.attentions = nn.LayerList(attentions)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: paddle.Tensor,
    ) -> paddle.Tensor:

        for i, (resnet, norm, attn) in enumerate(zip(self.resnets, self.norms, self.attentions)):

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states = paddle.distributed.fleet.utils.recompute(create_custom_forward(resnet), hidden_states)
            else:
                hidden_states = resnet(hidden_states)

            if attn is not None:
                residual = hidden_states
                hidden_states = norm(hidden_states)

                batch_size, num_channels, num_frames, height, width = hidden_states.shape
                hidden_states = hidden_states.transpose([0, 3, 4, 2, 1]).reshape(
                    [batch_size * height * width, num_frames, num_channels]
                )
                hidden_states = attn(hidden_states)
                hidden_states = hidden_states.reshape([batch_size, height, width, num_frames, num_channels]).transpose(
                    [0, 4, 3, 1, 2]
                )

                hidden_states = residual + hidden_states

        return hidden_states


class MochiUpBlock3D(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        temporal_expansion: int = 2,
        spatial_expansion: int = 2,
    ):
        super().__init__()
        self.temporal_expansion = temporal_expansion
        self.spatial_expansion = spatial_expansion

        resnets = []
        for _ in range(num_layers):
            resnets.append(MochiResnetBlock3D(in_channels=in_channels))
        self.resnets = nn.LayerList(resnets)

        self.proj = nn.Linear(in_channels, out_channels * temporal_expansion * spatial_expansion**2)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: paddle.Tensor,
    ) -> paddle.Tensor:

        for i, resnet in enumerate(self.resnets):

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def create_forward(*inputs):
                        return module(*inputs)

                    return create_forward

                hidden_states = paddle.distributed.fleet.utils.recompute(create_custom_forward(resnet), hidden_states)
            else:
                hidden_states = resnet(hidden_states)

        hidden_states = hidden_states.transpose([0, 2, 3, 4, 1])
        hidden_states = self.proj(hidden_states)
        hidden_states = hidden_states.transpose([0, 4, 1, 2, 3])

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        st = self.temporal_expansion
        sh = self.spatial_expansion
        sw = self.spatial_expansion

        hidden_states = hidden_states.reshape([batch_size, -1, st, sh, sw, num_frames, height, width])
        hidden_states = hidden_states.transpose([0, 1, 5, 2, 6, 3, 7, 4])
        hidden_states = hidden_states.reshape([batch_size, -1, num_frames * st, height * sh, width * sw])

        return hidden_states


class FourierFeatures(nn.Layer):
    def __init__(self, start: int = 6, stop: int = 8, step: int = 1):
        super().__init__()

        self.start = start
        self.stop = stop
        self.step = step

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        original_dtype = inputs.dtype
        inputs = inputs.astype("float32")
        num_channels = inputs.shape[1]
        num_freqs = (self.stop - self.start) // self.step

        freqs = paddle.arange(self.start, self.stop, self.step, dtype=inputs.dtype)
        w = paddle.pow(paddle.to_tensor(2.0), freqs) * (2 * np.pi)
        w = w.tile([num_channels]).unsqueeze([0, 2, 3, 4])

        h = inputs.tile([1, num_freqs, 1, 1, 1])
        h = w * h

        return paddle.concat([inputs, paddle.sin(h), paddle.cos(h)], axis=1).astype(original_dtype)


class MochiEncoder3D(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 768),
        layers_per_block: Tuple[int, ...] = (3, 3, 4, 6, 3),
        temporal_expansions: Tuple[int, ...] = (1, 2, 3),
        spatial_expansions: Tuple[int, ...] = (2, 2, 2),
        add_attention_block: Tuple[bool, ...] = (False, True, True, True, True),
        act_fn: str = "swish",
    ):
        super().__init__()

        self.nonlinearity = get_activation(act_fn)

        self.fourier_features = FourierFeatures()
        self.proj_in = nn.Linear(in_channels, block_out_channels[0])
        self.block_in = MochiMidBlock3D(
            in_channels=block_out_channels[0], num_layers=layers_per_block[0], add_attention=add_attention_block[0]
        )

        down_blocks = []
        for i in range(len(block_out_channels) - 1):
            down_block = MochiDownBlock3D(
                in_channels=block_out_channels[i],
                out_channels=block_out_channels[i + 1],
                num_layers=layers_per_block[i + 1],
                temporal_expansion=temporal_expansions[i],
                spatial_expansion=spatial_expansions[i],
                add_attention=add_attention_block[i + 1],
            )
            down_blocks.append(down_block)
        self.down_blocks = nn.LayerList(down_blocks)

        self.block_out = MochiMidBlock3D(
            in_channels=block_out_channels[-1], num_layers=layers_per_block[-1], add_attention=add_attention_block[-1]
        )
        self.norm_out = MochiChunkedGroupNorm3D(block_out_channels[-1])
        self.proj_out = nn.Linear(block_out_channels[-1], 2 * out_channels, bias_attr=False)

    def forward(
        self,
        hidden_states: paddle.Tensor,
    ) -> paddle.Tensor:

        hidden_states = self.fourier_features(hidden_states)

        hidden_states = hidden_states.transpose([0, 2, 3, 4, 1])
        hidden_states = self.proj_in(hidden_states)
        hidden_states = hidden_states.transpose([0, 4, 1, 2, 3])

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            hidden_states = paddle.distributed.fleet.utils.recompute(
                create_custom_forward(self.block_in), hidden_states
            )

            for i, down_block in enumerate(self.down_blocks):
                hidden_states = paddle.distributed.fleet.utils.recompute(
                    create_custom_forward(down_block), hidden_states
                )
        else:
            hidden_states = self.block_in(hidden_states)

            for i, down_block in enumerate(self.down_blocks):
                hidden_states = down_block(hidden_states)

        hidden_states = self.block_out(hidden_states)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = hidden_states.transpose([0, 2, 3, 4, 1])
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.transpose([0, 4, 1, 2, 3])

        return hidden_states


class MochiDecoder3D(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 768),
        layers_per_block: Tuple[int, ...] = (3, 3, 4, 6, 3),
        temporal_expansions: Tuple[int, ...] = (1, 2, 3),
        spatial_expansions: Tuple[int, ...] = (2, 2, 2),
        act_fn: str = "swish",
    ):
        super().__init__()

        self.nonlinearity = get_activation(act_fn)

        self.conv_in = nn.Conv3D(in_channels, block_out_channels[-1], kernel_size=(1, 1, 1))
        self.block_in = MochiMidBlock3D(
            in_channels=block_out_channels[-1],
            num_layers=layers_per_block[-1],
            add_attention=False,
        )

        up_blocks = []
        for i in range(len(block_out_channels) - 1):
            up_block = MochiUpBlock3D(
                in_channels=block_out_channels[-i - 1],
                out_channels=block_out_channels[-i - 2],
                num_layers=layers_per_block[-i - 2],
                temporal_expansion=temporal_expansions[-i - 1],
                spatial_expansion=spatial_expansions[-i - 1],
            )
            up_blocks.append(up_block)
        self.up_blocks = nn.LayerList(up_blocks)

        self.block_out = MochiMidBlock3D(
            in_channels=block_out_channels[0],
            num_layers=layers_per_block[0],
            add_attention=False,
        )
        self.proj_out = nn.Linear(block_out_channels[0], out_channels)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:

        hidden_states = self.conv_in(hidden_states)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            hidden_states = paddle.distributed.fleet.utils.recompute(
                create_custom_forward(self.block_in), hidden_states
            )

            for i, up_block in enumerate(self.up_blocks):
                hidden_states = paddle.distributed.fleet.utils.recompute(
                    create_custom_forward(up_block), hidden_states
                )
        else:
            hidden_states = self.block_in(hidden_states)

            for i, up_block in enumerate(self.up_blocks):
                hidden_states = up_block(hidden_states)

        hidden_states = self.block_out(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = hidden_states.transpose([0, 2, 3, 4, 1])
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.transpose([0, 4, 1, 2, 3])

        return hidden_states


class AutoencoderKLMochi(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["MochiResnetBlock3D"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 15,
        out_channels: int = 3,
        encoder_block_out_channels: Tuple[int] = (64, 128, 256, 384),
        decoder_block_out_channels: Tuple[int] = (128, 256, 512, 768),
        latent_channels: int = 12,
        layers_per_block: Tuple[int, ...] = (3, 3, 4, 6, 3),
        act_fn: str = "silu",
        temporal_expansions: Tuple[int, ...] = (1, 2, 3),
        spatial_expansions: Tuple[int, ...] = (2, 2, 2),
        add_attention_block: Tuple[bool, ...] = (False, True, True, True, True),
        latents_mean: Tuple[float, ...] = (
            -0.06730895953510081,
            -0.038011381506090416,
            -0.07477820912866141,
            -0.05565264470995561,
            0.012767231469026969,
            -0.04703542746246419,
            0.043896967884726704,
            -0.09346305707025976,
            -0.09918314763016893,
            -0.008729793427399178,
            -0.011931556316503654,
            -0.0321993391887285,
        ),
        latents_std: Tuple[float, ...] = (
            0.9263795028493863,
            0.9248894543193766,
            0.9393059390890617,
            0.959253732819592,
            0.8244560132752793,
            0.917259975397747,
            0.9294154431013696,
            1.3720942357788521,
            0.881393668867029,
            0.9168315692124348,
            0.9185249279345552,
            0.9274757570805041,
        ),
        scaling_factor: float = 1.0,
    ):
        super().__init__()

        self.encoder = MochiEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=layers_per_block,
            temporal_expansions=temporal_expansions,
            spatial_expansions=spatial_expansions,
            add_attention_block=add_attention_block,
            act_fn=act_fn,
        )
        self.decoder = MochiDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=layers_per_block,
            temporal_expansions=temporal_expansions,
            spatial_expansions=spatial_expansions,
            act_fn=act_fn,
        )

        self.spatial_compression_ratio = functools.reduce(lambda x, y: x * y, spatial_expansions, 1)
        self.temporal_compression_ratio = functools.reduce(lambda x, y: x * y, temporal_expansions, 1)

        self.use_slicing = False
        self.use_tiling = False
        self.use_framewise_encoding = False
        self.use_framewise_decoding = False
        self.drop_last_temporal_frames = True

        self.num_sample_frames_batch_size = 12
        self.num_latent_frames_batch_size = 2

        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
    ) -> None:
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width

    def disable_tiling(self) -> None:
        self.use_tiling = False

    def enable_slicing(self) -> None:
        self.use_slicing = True

    def disable_slicing(self) -> None:
        self.use_slicing = False

    def _enable_framewise_encoding(self):
        self.use_framewise_encoding = True
        for name, module in self.named_sublayers():
            if isinstance(module, CogVideoXCausalConv3d):
                module.pad_mode = "constant"

    def _enable_framewise_decoding(self):
        self.use_framewise_decoding = True
        for name, module in self.named_sublayers():
            if isinstance(module, CogVideoXCausalConv3d):
                module.pad_mode = "constant"

    def _encode(self, x: paddle.Tensor) -> paddle.Tensor:
        batch_size, num_channels, num_frames, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        if self.use_framewise_encoding:
            raise NotImplementedError(
                "Frame-wise encoding does not work with the Mochi VAE Encoder due to the presence of attention layers. "
                "As intermediate frames are not independent from each other, they cannot be encoded frame-wise."
            )
        else:
            enc = self.encoder(x)

        return enc

    @apply_forward_hook
    def encode(
        self, x: paddle.Tensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in paddle.split(x, num_or_sections=1, axis=0)]
            h = paddle.concat(encoded_slices, axis=0)
        else:
            h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: paddle.Tensor, return_dict: bool = True) -> Union[DecoderOutput, paddle.Tensor]:
        batch_size, num_channels, num_frames, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        if self.use_framewise_decoding:
            dec = []

            for i in range(0, num_frames, self.num_latent_frames_batch_size):
                z_intermediate = z[:, :, i : i + self.num_latent_frames_batch_size]
                z_intermediate = self.decoder(z_intermediate)
                dec.append(z_intermediate)

            dec = paddle.concat(dec, axis=2)
        else:
            dec = self.decoder(z)

        if self.drop_last_temporal_frames and dec.shape[2] >= self.temporal_compression_ratio:
            dec = dec[:, :, self.temporal_compression_ratio - 1 :]

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z: paddle.Tensor, return_dict: bool = True) -> Union[DecoderOutput, paddle.Tensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in paddle.split(z, num_or_sections=1, axis=0)]
            decoded = paddle.concat(decoded_slices, axis=0)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(self, a: paddle.Tensor, b: paddle.Tensor, blend_extent: int) -> paddle.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: paddle.Tensor, b: paddle.Tensor, blend_extent: int) -> paddle.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def tiled_encode(self, x: paddle.Tensor) -> paddle.Tensor:
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                if self.use_framewise_encoding:
                    raise NotImplementedError(
                        "Frame-wise encoding does not work with the Mochi VAE Encoder due to the presence of attention layers. "
                        "As intermediate frames are not independent from each other, they cannot be encoded frame-wise."
                    )
                else:
                    time = self.encoder(
                        x[:, :, :, i : i + self.tile_sample_min_height, j : j + self.tile_sample_min_width]
                    )

                row.append(time)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
            result_rows.append(paddle.concat(result_row, axis=4))

        enc = paddle.concat(result_rows, axis=3)[:, :, :, :latent_height, :latent_width]
        return enc

    def tiled_decode(self, z: paddle.Tensor, return_dict: bool = True) -> Union[DecoderOutput, paddle.Tensor]:
        batch_size, num_channels, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                if self.use_framewise_decoding:
                    time = []

                    for k in range(0, num_frames, self.num_latent_frames_batch_size):
                        tile = z[
                            :,
                            :,
                            k : k + self.num_latent_frames_batch_size,
                            i : i + tile_latent_min_height,
                            j : j + tile_latent_min_width,
                        ]
                        tile = self.decoder(tile)
                        time.append(tile)

                    time = paddle.concat(time, axis=2)
                else:
                    time = self.decoder(z[:, :, :, i : i + tile_latent_min_height, j : j + tile_latent_min_width])

                if self.drop_last_temporal_frames and time.shape[2] >= self.temporal_compression_ratio:
                    time = time[:, :, self.temporal_compression_ratio - 1 :]

                row.append(time)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(paddle.concat(result_row, axis=4))

        dec = paddle.concat(result_rows, axis=3)[:, :, :, :sample_height, :sample_width]

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: paddle.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[paddle.Generator] = None,
    ) -> Union[paddle.Tensor, paddle.Tensor]:
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z)
        if not return_dict:
            return (dec,)
        return dec
