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
from dataclasses import dataclass
from typing import Optional

import paddle
from einops import rearrange, repeat

import ppdiffusers


class PositionalEncoding(paddle.nn.Layer):
    """
    Implements positional encoding as described in "Attention Is All You Need".
    Adds sinusoidal based positional encodings to the input tensor.
    """

    _SCALE_FACTOR = 10000.0

    def __init__(self, dim: int, dropout: float = 0.0, max_length: int = 24):
        super(PositionalEncoding, self).__init__()
        self.dropout = paddle.nn.Dropout(p=dropout)
        positional_encoding = paddle.zeros(shape=[1, max_length, dim])
        position = paddle.arange(end=max_length).unsqueeze(axis=1)
        div_term = paddle.exp(x=paddle.arange(start=0, end=dim, step=2) * (-math.log(self._SCALE_FACTOR) / dim))
        positional_encoding[0, :, 0::2] = paddle.sin(x=position.astype(div_term.dtype) * div_term)
        positional_encoding[0, :, 1::2] = paddle.cos(x=position.astype(div_term.dtype) * div_term)
        self.register_buffer(name="positional_encoding", tensor=positional_encoding)

    def forward(self, hidden_states: paddle.Tensor, length: int) -> paddle.Tensor:
        hidden_states = hidden_states + self.positional_encoding[:, :length]
        return self.dropout(hidden_states)


class TemporalAttention(ppdiffusers.models.attention.Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_encoder = PositionalEncoding(kwargs["query_dim"], dropout=0)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, number_of_frames=8):
        sequence_length = tuple(hidden_states.shape)[1]
        hidden_states = rearrange(hidden_states, "(b f) s c -> (b s) f c", f=number_of_frames)
        hidden_states = self.pos_encoder(hidden_states, length=number_of_frames)
        if encoder_hidden_states:
            encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b s) n c", s=sequence_length)
        hidden_states = super().forward(hidden_states, encoder_hidden_states, attention_mask=attention_mask)
        return rearrange(hidden_states, "(b s) f c -> (b f) s c", s=sequence_length)


@dataclass
class TransformerTemporalOutput(ppdiffusers.utils.BaseOutput):
    sample: paddle.float32


class TransformerTemporal(paddle.nn.Layer):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        in_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        upcast_attention: bool = False,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        self.norm = paddle.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, epsilon=1e-06, weight_attr=True, bias_attr=True
        )
        self.proj_in = paddle.nn.Linear(in_features=in_channels, out_features=inner_dim)
        self.transformer_blocks = paddle.nn.LayerList(
            sublayers=[
                TransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.proj_out = paddle.nn.Linear(in_features=inner_dim, out_features=in_channels)

    def forward(self, hidden_states, encoder_hidden_states=None):
        _, num_channels, f, height, width = tuple(hidden_states.shape)
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        skip = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = rearrange(hidden_states, "bf c h w -> bf (h w) c")
        hidden_states = self.proj_in(hidden_states)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states, number_of_frames=f)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = rearrange(hidden_states, "bf (h w) c -> bf c h w", h=height, w=width).contiguous()
        output = hidden_states + skip
        output = rearrange(output, "(b f) c h w -> b c f h w", f=f)
        return output


class TransformerBlock(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        dropout=0.0,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        depth=2,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.is_cross = cross_attention_dim is not None
        attention_blocks = []
        norms = []
        for _ in range(depth):
            attention_blocks.append(
                TemporalAttention(
                    query_dim=dim,
                    cross_attention_dim=cross_attention_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                )
            )
            norms.append(paddle.nn.LayerNorm(normalized_shape=dim))
        self.attention_blocks = paddle.nn.LayerList(sublayers=attention_blocks)
        self.norms = paddle.nn.LayerList(sublayers=norms)
        self.ff = ppdiffusers.models.attention.FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = paddle.nn.LayerNorm(normalized_shape=dim)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, number_of_frames=None):
        if not self.is_cross:
            encoder_hidden_states = None
        for block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = (
                block(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    number_of_frames=number_of_frames,
                )
                + hidden_states
            )
        norm_hidden_states = self.ff_norm(hidden_states)
        hidden_states = self.ff(norm_hidden_states) + hidden_states
        output = hidden_states
        return output
