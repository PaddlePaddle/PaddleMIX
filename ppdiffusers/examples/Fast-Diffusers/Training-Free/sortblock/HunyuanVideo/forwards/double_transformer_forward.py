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


def taylorseer_flux_double_block_forward(
    self,
    hidden_states: paddle.Tensor,
    encoder_hidden_states: paddle.Tensor,
    temb: paddle.Tensor,
    attention_mask: Optional[paddle.Tensor] = None,
    freqs_cis: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
    joint_attention_kwargs=None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:

    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        encoder_hidden_states, emb=temb
    )

    joint_attention_kwargs = joint_attention_kwargs or {}

    cache_dic = joint_attention_kwargs["cache_dic"]
    current = joint_attention_kwargs["current"]

    if current["type"] == "full":

        current["module"] = "attn"
        taylor_cache_init(cache_dic=cache_dic, current=current)
        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        # if len(attention_outputs) == 2:
        #     attn_output, context_attn_output = attention_outputs
        # elif len(attention_outputs) == 3:
        #     attn_output, context_attn_output, ip_attn_output = attention_outputs
        #     raise NotImplementedError("Not implemented for TaylorSeer yet.")

        # Process attention outputs for the `hidden_states`.
        current["module"] = "img_attn"
        taylor_cache_init(cache_dic=cache_dic, current=current)

        derivative_approximation(cache_dic=cache_dic, current=current, feature=attn_output)
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(axis=1)

        current["module"] = "img_mlp"
        taylor_cache_init(cache_dic=cache_dic, current=current)
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, (None)]) + shift_mlp[:, (None)]

        ff_output = self.ff(norm_hidden_states)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=ff_output)

        hidden_states = hidden_states + gate_mlp.unsqueeze(axis=1) * ff_output

        # if len(attention_outputs) == 3:
        #     hidden_states = hidden_states + ip_attn_output

        # Process attention outputs for the `encoder_hidden_states`.
        current["module"] = "txt_attn"
        taylor_cache_init(cache_dic=cache_dic, current=current)

        derivative_approximation(cache_dic=cache_dic, current=current, feature=context_attn_output)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(axis=1)

        current["module"] = "txt_mlp"
        taylor_cache_init(cache_dic=cache_dic, current=current)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, (None)]) + c_shift_mlp[:, (None)]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        derivative_approximation(cache_dic=cache_dic, current=current, feature=context_ff_output)

        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(axis=1) * context_ff_output

        if encoder_hidden_states.dtype == paddle.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    elif current["type"] == "Taylor":

        current["module"] = "attn"
        # Attention.
        # symbolic placeholder

        # Process attention outputs for the `hidden_states`.
        current["module"] = "img_attn"

        attn_output = taylor_formula(cache_dic=cache_dic, current=current)
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        current["module"] = "img_mlp"

        ff_output = taylor_formula(cache_dic=cache_dic, current=current)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        current["module"] = "txt_attn"

        context_attn_output = taylor_formula(cache_dic=cache_dic, current=current)

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        current["module"] = "txt_mlp"

        context_ff_output = taylor_formula(cache_dic=cache_dic, current=current)

        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        if encoder_hidden_states.dtype == paddle.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return hidden_states, encoder_hidden_states
