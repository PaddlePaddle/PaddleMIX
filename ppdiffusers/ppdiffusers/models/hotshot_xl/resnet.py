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
from einops import rearrange

import ppdiffusers
from ppdiffusers.models import resnet


class Upsample3D(resnet.Upsample2D):
    def forward(self, hidden_states, output_size=None, scale: float = 1.0):
        f = tuple(hidden_states.shape)[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        hidden_states = super(Upsample3D, self).forward(hidden_states, output_size, scale)
        return rearrange(hidden_states, "(b f) c h w -> b c f h w", f=f)


class Downsample3D(ppdiffusers.models.resnet.Downsample2D):
    def forward(self, hidden_states, scale: float = 1.0):
        f = tuple(hidden_states.shape)[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        hidden_states = super(Downsample3D, self).forward(hidden_states, scale)
        return rearrange(hidden_states, "(b f) c h w -> b c f h w", f=f)


class Conv3d(ppdiffusers.models.resnet.LoRACompatibleConv):
    def forward(self, hidden_states, scale: float = 1.0):
        f = tuple(hidden_states.shape)[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        hidden_states = super().forward(hidden_states, scale)
        return rearrange(hidden_states, "(b f) c h w -> b c f h w", f=f)


class ResnetBlock3D(paddle.nn.Layer):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-06,
        non_linearity="silu",
        time_embedding_norm="default",
        output_scale_factor=1.0,
        use_in_shortcut=None,
        conv_shortcut_bias: bool = True
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor
        if groups_out is None:
            groups_out = groups
        self.norm1 = paddle.nn.GroupNorm(
            num_groups=groups, num_channels=in_channels, epsilon=eps, weight_attr=True, bias_attr=True
        )
        self.conv1 = Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
            self.time_emb_proj = paddle.nn.Linear(in_features=temb_channels, out_features=time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None
        self.norm2 = paddle.nn.GroupNorm(
            num_groups=groups_out, num_channels=out_channels, epsilon=eps, weight_attr=True, bias_attr=True
        )
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.conv2 = Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        assert non_linearity == "silu"
        self.nonlinearity = paddle.nn.Silu()
        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut
        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = Conv3d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias_attr=conv_shortcut_bias
            )

    def forward(self, input_tensor, temb):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)
        if temb is not None:
            temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None, None]
        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)
        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = paddle.chunk(x=temb, chunks=2, axis=1)
            hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor
