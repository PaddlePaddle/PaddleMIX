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

from typing import Optional, Tuple

import paddle
from taylorseer_utils import derivative_approximation, taylor_cache_init, taylor_formula


def taylorseer_flux_single_block_forward(
    self,
    hidden_states: paddle.Tensor,
    encoder_hidden_states: paddle.Tensor,
    temb: paddle.Tensor,
    attention_mask: Optional[paddle.Tensor] = None,
    image_rotary_emb: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
    joint_attention_kwargs=None,
):
    text_seq_length = tuple(encoder_hidden_states.shape)[1]
    hidden_states = paddle.concat(x=[hidden_states, encoder_hidden_states], axis=1)

    residual = hidden_states

    joint_attention_kwargs = joint_attention_kwargs or {}
    cache_dic = joint_attention_kwargs["cache_dic"]
    current = joint_attention_kwargs["current"]

    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

    if current["type"] == "full":

        current["module"] = "total"
        taylor_cache_init(cache_dic=cache_dic, current=current)

        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        attn_output = paddle.concat(x=[attn_output, context_attn_output], axis=1)

        hidden_states = paddle.concat(x=[attn_output, mlp_hidden_states], axis=2)

        hidden_states = self.proj_out(hidden_states)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=hidden_states)

    elif current["type"] == "Taylor":

        current["module"] = "total"
        hidden_states = taylor_formula(cache_dic=cache_dic, current=current)

    hidden_states = gate.unsqueeze(axis=1) * hidden_states
    hidden_states = residual + hidden_states

    hidden_states, encoder_hidden_states = (
        hidden_states[:, :-text_seq_length, :],
        hidden_states[:, -text_seq_length:, :],
    )

    return hidden_states, encoder_hidden_states
