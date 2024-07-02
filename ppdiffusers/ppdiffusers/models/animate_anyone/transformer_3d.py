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

from dataclasses import dataclass
from typing import Optional

import paddle
from einops import rearrange, repeat

from ppdiffusers.configuration_utils import ConfigMixin, register_to_config
from ppdiffusers.models import ModelMixin
from ppdiffusers.utils import BaseOutput

from .attention import TemporalBasicTransformerBlock


@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: paddle.Tensor


class Transformer3DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = paddle.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, epsilon=1e-06, weight_attr=True, bias_attr=True
        )
        if use_linear_projection:
            self.proj_in = paddle.nn.Linear(in_features=in_channels, out_features=inner_dim)
        else:
            self.proj_in = paddle.nn.Conv2D(
                in_channels=in_channels, out_channels=inner_dim, kernel_size=1, stride=1, padding=0
            )
        self.transformer_blocks = paddle.nn.LayerList(
            sublayers=[
                TemporalBasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                )
                for d in range(num_layers)
            ]
        )
        if use_linear_projection:
            self.proj_out = paddle.nn.Linear(in_features=in_channels, out_features=inner_dim)
        else:
            self.proj_out = paddle.nn.Conv2D(
                in_channels=inner_dim, out_channels=in_channels, kernel_size=1, stride=1, padding=0
            )

        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        return_dict: bool = True,
    ):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        if encoder_hidden_states.shape[0] != hidden_states.shape[0]:
            encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b f) n c", f=video_length)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.transpose(perm=[0, 2, 3, 1]).reshape((batch, height * weight, inner_dim))
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.transpose(perm=[0, 2, 3, 1]).reshape((batch, height * weight, inner_dim))
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for i, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length,
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape((batch, height, weight, inner_dim)).transpose(perm=[0, 3, 1, 2])
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape((batch, height, weight, inner_dim)).transpose(perm=[0, 3, 1, 2])

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)
