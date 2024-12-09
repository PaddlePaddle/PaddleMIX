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
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet.meta_parallel import ColumnParallelLinear as CPLinear
from paddle.distributed.fleet.meta_parallel import RowParallelLinear as RPLinear
from paddle.nn import LayerList as LayerList


class SimplifiedSD3(nn.Layer):
    def __init__(self, num_layers: int, dim: int, num_attention_heads: int, attention_head_dim: int, mp_degree: int):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.head_dim = 64

        self.mp_degree = mp_degree

        self.silu = nn.Silu()
        self.linear1 = LayerList([nn.Linear(self.dim, 6 * self.dim) for i in range(num_layers)])
        self.linear_context = LayerList(
            [nn.Linear(self.dim, (6 if i < num_layers - 1 else 2) * self.dim) for i in range(num_layers)]
        )
        self.norm_last_context = nn.LayerNorm(self.dim, epsilon=1e-6, weight_attr=False, bias_attr=True)

        if mp_degree > 1:
            self.qkv_mp = LayerList(
                [CPLinear(self.dim, 3 * self.dim, gather_output=False, has_bias=True) for i in range(num_layers)]
            )
            self.eqkv_mp = LayerList(
                [CPLinear(self.dim, 3 * self.dim, gather_output=False, has_bias=True) for i in range(num_layers)]
            )
            self.to_out_linear_mp = LayerList(
                [RPLinear(self.dim, self.dim, input_is_parallel=True, has_bias=True) for i in range(num_layers)]
            )
            # When using Model Parallel, for the symmetry of GEMM, we change num_layers-1 here to num_layers, which has no effect on the results.
            self.to_add_out_linear_mp = LayerList(
                [RPLinear(self.dim, self.dim, input_is_parallel=True, has_bias=True) for i in range(num_layers)]
            )

            self.ffn1_mp = LayerList(
                [CPLinear(self.dim, 4 * self.dim, gather_output=False, has_bias=True) for i in range(num_layers)]
            )
            self.ffn2_mp = LayerList(
                [RPLinear(self.dim * 4, self.dim, input_is_parallel=True, has_bias=True) for i in range(num_layers)]
            )
            self.ffn1_context_mp = LayerList(
                [CPLinear(self.dim, 4 * self.dim, gather_output=False, has_bias=True) for i in range(num_layers - 1)]
            )
            self.ffn2_context_mp = LayerList(
                [
                    RPLinear(self.dim * 4, self.dim, input_is_parallel=True, has_bias=True)
                    for i in range(num_layers - 1)
                ]
            )
        else:
            self.qkv = LayerList([nn.Linear(self.dim, self.dim * 3) for i in range(num_layers)])
            self.eqkv = LayerList([nn.Linear(self.dim, self.dim * 3) for i in range(num_layers)])
            self.to_out_linear = LayerList([nn.Linear(self.dim, self.dim) for i in range(num_layers)])
            # When using Model Parallel, for the symmetry of GEMM, we change num_layers-1 here to num_layers, which has no effect on the results.
            self.to_add_out_linear = LayerList([nn.Linear(self.dim, self.dim) for i in range(num_layers)])

            self.ffn1 = LayerList([nn.Linear(self.dim, self.dim * 4) for i in range(num_layers)])
            self.ffn2 = LayerList([nn.Linear(self.dim * 4, self.dim) for i in range(num_layers)])
            self.ffn1_context = LayerList([nn.Linear(self.dim, self.dim * 4) for i in range(num_layers - 1)])
            self.ffn2_context = LayerList([nn.Linear(self.dim * 4, self.dim) for i in range(num_layers - 1)])

    def forward(self, hidden_states, encoder_hidden_states, temb):
        print("--------------------this is simplified_sd3------------------------")
        temb_silu = self.silu(temb)

        last_ffn_output = None
        last_hidden_states = None
        last_gate_mlp = None

        last_context_ffn_output = None
        last_context_hidden_states = None
        last_context_gate_mlp = None

        seq1 = hidden_states.shape[1]
        seq2 = encoder_hidden_states.shape[1]

        for i in range(self.num_layers):
            context_pre_only = i == self.num_layers - 1

            emb = self.linear1[i](temb_silu)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, axis=1)

            import paddlemix

            if last_ffn_output is None:
                norm_hidden_states = paddlemix.triton_ops.adaptive_layer_norm(
                    hidden_states, scale_msa, shift_msa, epsilon=1e-06
                )
            else:
                hidden_states, norm_hidden_states = paddlemix.triton_ops.fused_adaLN_scale_residual(
                    last_hidden_states, last_ffn_output, last_gate_mlp, scale_msa, shift_msa, epsilon=1e-06
                )

            emb = self.linear_context[i](temb_silu)
            if not context_pre_only:
                shift_msa, scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = emb.chunk(6, axis=1)
                if last_context_ffn_output is None:
                    norm_encoder_hidden_states = paddlemix.triton_ops.adaptive_layer_norm(
                        encoder_hidden_states, scale_msa, shift_msa, epsilon=1e-06
                    )
                else:
                    (
                        encoder_hidden_states,
                        norm_encoder_hidden_states,
                    ) = paddlemix.triton_ops.fused_adaLN_scale_residual(
                        last_context_hidden_states,
                        last_context_ffn_output,
                        last_context_gate_mlp,
                        scale_msa,
                        shift_msa,
                        epsilon=1e-06,
                    )
            else:
                # the last layer.
                scale, shift = paddle.chunk(emb, 2, axis=1)
                (encoder_hidden_states, norm_encoder_hidden_states,) = paddlemix.triton_ops.fused_adaLN_scale_residual(
                    last_context_hidden_states,
                    last_context_ffn_output,
                    last_context_gate_mlp,
                    scale,
                    shift,
                    epsilon=1e-06,
                )

            if self.mp_degree > 1:
                qkv = self.qkv_mp[i](norm_hidden_states)
                eqkv = self.eqkv_mp[i](norm_encoder_hidden_states)

            else:
                qkv = self.qkv[i](norm_hidden_states)
                eqkv = self.eqkv[i](norm_encoder_hidden_states)

            q, k, v = paddlemix.triton_ops.split_concat(qkv, eqkv)

            bs = hidden_states.shape[0]
            head_nums = q.shape[2] // self.head_dim
            q = q.reshape([bs, -1, head_nums, self.head_dim])
            k = k.reshape([bs, -1, head_nums, self.head_dim])
            v = v.reshape([bs, -1, head_nums, self.head_dim])

            norm_hidden_states1 = F.scaled_dot_product_attention_(q, k, v, dropout_p=0.0, is_causal=False)
            norm_hidden_states1 = norm_hidden_states1.reshape([bs, -1, head_nums * self.head_dim])
            attn_output, context_attn_output = paddle.split(norm_hidden_states1, num_or_sections=[seq1, seq2], axis=1)

            # attn_output, context_attn_output = paddlemix.triton_ops.triton_split(
            #     norm_hidden_states1, num_or_sections=[1024, 154], axis=1
            # )

            if self.mp_degree > 1:
                attn_output = self.to_out_linear_mp[i](attn_output)
                context_attn_output = self.to_add_out_linear_mp[i](context_attn_output)
            else:
                attn_output = self.to_out_linear[i](attn_output)
                context_attn_output = self.to_add_out_linear[i](context_attn_output)

            hidden_states, norm_hidden_states = paddlemix.triton_ops.fused_adaLN_scale_residual(
                hidden_states, attn_output, gate_msa, scale_mlp, shift_mlp, epsilon=1e-06
            )

            # ffn1
            if self.mp_degree > 1:
                ffn_output = self.ffn1_mp[i](norm_hidden_states)
                ffn_output = F.gelu(ffn_output, approximate=True)
                ffn_output = self.ffn2_mp[i](ffn_output)
            else:
                ffn_output = self.ffn1[i](norm_hidden_states)
                ffn_output = F.gelu(ffn_output, approximate=True)
                ffn_output = self.ffn2[i](ffn_output)

            if context_pre_only:
                ffn_output = gate_mlp.unsqueeze(1) * ffn_output
                hidden_states = hidden_states + ffn_output
            else:
                last_ffn_output = ffn_output
                last_hidden_states = hidden_states
                last_gate_mlp = gate_mlp

            # ffn2
            if not context_pre_only:
                (encoder_hidden_states, norm_encoder_hidden_states,) = paddlemix.triton_ops.fused_adaLN_scale_residual(
                    encoder_hidden_states, context_attn_output, c_gate_msa, c_scale_mlp, c_shift_mlp, epsilon=1e-06
                )

                if self.mp_degree > 1:
                    context_ffn_output = self.ffn1_context_mp[i](norm_encoder_hidden_states)
                    context_ffn_output = F.gelu(context_ffn_output, approximate=True)
                    context_ffn_output = self.ffn2_context_mp[i](context_ffn_output)
                else:
                    context_ffn_output = self.ffn1_context[i](norm_encoder_hidden_states)
                    context_ffn_output = F.gelu(context_ffn_output, approximate=True)
                    context_ffn_output = self.ffn2_context[i](context_ffn_output)

                last_context_ffn_output = context_ffn_output
                last_context_hidden_states = encoder_hidden_states
                last_context_gate_mlp = c_gate_mlp

        return hidden_states
