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
from typing import Any, Dict, Optional

import paddle
from einops import rearrange, repeat

import ppdiffusers


@dataclass
class Transformer3DModelOutput(ppdiffusers.utils.BaseOutput):
    """
    The output of [`Transformer3DModel`].

    Args:
        sample (`paddle.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
            The hidden states output conditioned on the `encoder_hidden_states` input.
    """

    sample: paddle.float32


class Transformer3DModel(ppdiffusers.models.transformer_2d.Transformer2DModel):
    def __init__(self, *args, **kwargs):
        super(Transformer3DModel, self).__init__(*args, **kwargs)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(self.proj_out.weight.data)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(self.proj_out.bias.data)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        timestep: Optional[int] = None,
        class_labels: Optional[int] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        encoder_attention_mask: Optional[paddle.Tensor] = None,
        enable_temporal_layers: bool = True,
        positional_embedding: Optional[paddle.Tensor] = None,
        return_dict: bool = True,
    ):
        is_video = len(tuple(hidden_states.shape)) == 5
        if is_video:
            f = tuple(hidden_states.shape)[2]
            hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
            encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b f) n c", f=f)
        hidden_states = super(Transformer3DModel, self).forward(
            hidden_states,
            encoder_hidden_states,
            timestep,
            class_labels,
            cross_attention_kwargs,
            attention_mask,
            encoder_attention_mask,
            return_dict=False,
        )[0]
        if is_video:
            hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=f)
        if not return_dict:
            return (hidden_states,)
        return Transformer3DModelOutput(sample=hidden_states)
