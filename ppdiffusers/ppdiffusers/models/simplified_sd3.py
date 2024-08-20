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

# import math
# import os

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.incubate.nn.functional import fused_linear, fused_linear_activation

optimize = True


class SimplifiedSD3(nn.Layer):
    def __init__(
        self, num_layers: int, dim: int, num_attention_heads: int, attention_head_dim: int, context_pre_only=False
    ):
        super().__init__()

        self.context_pre_only = context_pre_only
        self.num_layers = num_layers
        self.dim = dim
        self.bias = True
        self.elementwise_affine = True

        # layer List

        # silu + matmul + add
        # self.silu1 = nn.LayerList([nn.Silu() for i in range(num_layers)])
        self.silu = nn.Silu()
        self.linear1 = nn.LayerList([nn.Linear(1536, 6 * 1536) for i in range(num_layers)])  # 1536,9216
        # self.linear1 = nn.Linear(1536, 6 * 1536 * 24)

        norm_elementwise_affine_kwargs = dict(weight_attr=False, bias_attr=False)
        self.norm1 = nn.LayerList(
            [nn.LayerNorm(1536, epsilon=1e-6, **norm_elementwise_affine_kwargs) for i in range(num_layers)]
        )

        # not last layer
        # self.silu2_context01 = nn.LayerList([nn.Silu() for i in range(num_layers - 1)])
        self.linear_context01 = nn.LayerList([nn.Linear(1536, 6 * 1536) for i in range(num_layers - 1)])  # 1536,9216
        self.norm1_context01 = nn.LayerList(
            [nn.LayerNorm(1536, epsilon=1e-6, **norm_elementwise_affine_kwargs) for i in range(num_layers - 1)]
        )  # another

        # last layer
        # self.silu2_context0 = nn.Silu()
        self.linear_context0 = nn.Linear(1536, 1536 * 2, bias_attr=self.bias)
        self.norm1_context0 = nn.LayerNorm(1536, epsilon=1e-06, weight_attr=False, bias_attr=self.bias)

        # attention
        # self.q = nn.LayerList([nn.Linear(1536, 1536) for i in range(num_layers)])
        # self.k = nn.LayerList([nn.Linear(1536, 1536) for i in range(num_layers)])
        # self.v = nn.LayerList([nn.Linear(1536, 1536) for i in range(num_layers)])
        self.qkv = nn.LayerList([nn.Linear(1536, 1536 * 3) for i in range(num_layers)])

        # self.eq = nn.LayerList([nn.Linear(1536, 1536) for i in range(num_layers)])
        # self.ek = nn.LayerList([nn.Linear(1536, 1536) for i in range(num_layers)])
        # self.ev = nn.LayerList([nn.Linear(1536, 1536) for i in range(num_layers)])
        self.eqkv = nn.LayerList([nn.Linear(1536, 1536 * 3) for i in range(num_layers)])
        self.to_out_linear = nn.LayerList([nn.Linear(1536, 1536) for i in range(num_layers)])
        # self.to_out =  nn.LayerList([nn.Dropout(0.0) for i in range(num_layers)])

        # not last layer
        self.to_add_out_linear = nn.LayerList([nn.Linear(1536, 1536) for i in range(num_layers - 1)])

        self.ffn_norm = nn.LayerList(
            [nn.LayerNorm(1536, weight_attr=False, bias_attr=False, epsilon=1e-6) for i in range(num_layers)]
        )
        self.ffn1 = nn.LayerList([nn.Linear(1536, 1536 * 4) for i in range(num_layers)])
        self.ffn2 = nn.LayerList([nn.Linear(1536 * 4, 1536) for i in range(num_layers)])

        # not last layer
        self.ffn_context_norm = nn.LayerList(
            [nn.LayerNorm(1536, epsilon=1e-6, weight_attr=False, bias_attr=False) for i in range(num_layers - 1)]
        )
        self.ffn_context1 = nn.LayerList([nn.Linear(1536, 1536 * 4) for i in range(num_layers - 1)])
        self.ffn_context2 = nn.LayerList([nn.Linear(1536 * 4, 1536) for i in range(num_layers - 1)])

    def forward(self, hidden_states, encoder_hidden_states, temb):

        temb_silu = self.silu(temb)
        # emb1 = self.linear1(temb_silu)
        for i in range(self.num_layers):
            # emb=emb1[:,i*6*1536:(i+1)*1536*6]
            context_pre_only = i == self.num_layers - 1

            # emb  = self.linear1[i](self.silu1(temb))
            emb = self.linear1[i](temb_silu)

            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, axis=1)
            if optimize:
                import paddlemix

                norm_hidden_states = paddlemix.triton_ops.adaptive_layer_norm(
                    hidden_states, scale_msa, shift_msa, epsilon=1e-06
                )
            else:
                norm_hidden_states = self.norm1[i](hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]

            if not context_pre_only:
                # emb = self.linear_context01[i](self.silu2_context01[i](temb))
                emb = self.linear_context01[i](temb_silu)
                shift_msa, scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = emb.chunk(6, axis=1)

                if optimize:
                    import paddlemix

                    norm_encoder_hidden_states = paddlemix.triton_ops.adaptive_layer_norm(
                        encoder_hidden_states, scale_msa, shift_msa, epsilon=1e-06
                    )
                else:
                    norm_encoder_hidden_states = (
                        self.norm1_context01[i](encoder_hidden_states) * (1 + scale_msa[:, None]) + shift_msa[:, None]
                    )

            else:  # last layer
                emb = self.linear_context0(temb_silu.cast(encoder_hidden_states.dtype))
                scale, shift = paddle.chunk(emb, 2, axis=1)

                if optimize:
                    import paddlemix

                    norm_encoder_hidden_states = paddlemix.triton_ops.adaptive_layer_norm(
                        encoder_hidden_states, scale, shift, bias=self.norm1_context0.bias
                    )
                else:
                    norm_encoder_hidden_states = (
                        self.norm1_context0(encoder_hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
                    )

            # -------------------------^ attention ^-----------------------
            # residual = norm_hidden_states
            # q = self.q[i](norm_hidden_states)
            # k = self.k[i](norm_hidden_states)
            # v = self.v[i](norm_hidden_states)
            qkv = self.qkv[i](norm_hidden_states)
            # q,k,v = paddle.split(qkv,axis=2, num_or_sections=3)

            # eq = self.eq[i](norm_encoder_hidden_states)
            # ek = self.ek[i](norm_encoder_hidden_states)
            # ev = self.ev[i](norm_encoder_hidden_states)
            eqkv = self.eqkv[i](norm_encoder_hidden_states)
            # eq,ek,ev = paddle.split(eqkv,axis=2, num_or_sections=3)

            # q = paddle.concat([q, eq], axis=1).reshape([2, -1, 24, 64])
            # k = paddle.concat([k, ek], axis=1).reshape([2, -1, 24, 64])
            # v = paddle.concat([v, ev], axis=1).reshape([2, -1, 24, 64])

            import paddlemix

            q, k, v = paddlemix.triton_ops.my_splcat(qkv, eqkv)
            q = q.reshape([2, -1, 24, 64])
            k = k.reshape([2, -1, 24, 64])
            v = v.reshape([2, -1, 24, 64])

            # qkv = paddle.concat([q, eq, k, ek, v, ev], axis=1).reshape([2, -1, 24, 64])
            # q,k,v = paddle.split(qkv,axis=1, num_or_sections=3)

            norm_hidden_states1 = F.scaled_dot_product_attention_(q, k, v, dropout_p=0.0, is_causal=False)
            norm_hidden_states1 = norm_hidden_states1.reshape([2, -1, 1536])
            norm_hidden_states1 = norm_hidden_states1.astype(q.dtype)

            # attn_output, context_attn_output = (
            #     norm_hidden_states1[:, : residual.shape[1]],
            #     norm_hidden_states1[:, residual.shape[1] :],
            # )
            attn_output, context_attn_output = paddle.split(norm_hidden_states1, num_or_sections=[1024, 154], axis=1)

            attn_output = paddle.nn.functional.linear(
                attn_output, self.to_out_linear[i].weight, self.to_out_linear[i].bias
            )

            if not context_pre_only:
                context_attn_output = self.to_add_out_linear[i](context_attn_output)

            # -------------------------^ attention ^-----------------------
            # ===============FF_First

            if optimize:
                import paddlemix

                hidden_states, norm_hidden_states = paddlemix.triton_ops.fused_adaLN_scale_residual(
                    hidden_states, attn_output, gate_msa, scale_mlp, shift_mlp, epsilon=1e-06
                )
            else:
                attn_output = gate_msa.unsqueeze(1) * attn_output
                hidden_states = hidden_states + attn_output
                norm_hidden_states = self.ffn_norm[i](hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ffn1[i](norm_hidden_states)
            ff_output = F.gelu(ff_output, approximate=True)
            ff_output = self.ffn2[i](ff_output)

            ff_output = gate_mlp.unsqueeze(1) * ff_output
            hidden_states = hidden_states + ff_output

            # ===========FF_Second
            if not context_pre_only:
                if optimize:
                    import paddlemix

                    (
                        encoder_hidden_states,
                        norm_encoder_hidden_states,
                    ) = paddlemix.triton_ops.fused_adaLN_scale_residual(
                        encoder_hidden_states, context_attn_output, c_gate_msa, c_scale_mlp, c_shift_mlp, epsilon=1e-06
                    )
                else:
                    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
                    encoder_hidden_states = encoder_hidden_states + context_attn_output
                    norm_encoder_hidden_states = self.ffn_context_norm[i](encoder_hidden_states)
                    norm_encoder_hidden_states = (
                        norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
                    )

                context_ff_output = self.ffn_context1[i](norm_encoder_hidden_states)
                context_ff_output = F.gelu(context_ff_output, approximate=True)
                context_ff_output = self.ffn_context2[i](context_ff_output)

                encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            else:
                encoder_hidden_states = None
        return encoder_hidden_states, hidden_states
