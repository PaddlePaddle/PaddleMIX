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

from typing import Optional

import paddle
from sam2.modeling.sam2_utils import get_activation_fn, get_clones
from sam2.modeling.sam.transformer import RoPEAttention


class MemoryAttentionLayer(paddle.nn.Layer):
    def __init__(
        self,
        activation: str,
        cross_attention: paddle.nn.Layer,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        self_attention: paddle.nn.Layer,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention
        self.linear1 = paddle.nn.Linear(in_features=d_model, out_features=dim_feedforward)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.linear2 = paddle.nn.Linear(in_features=dim_feedforward, out_features=d_model)
        self.norm1 = paddle.nn.LayerNorm(normalized_shape=d_model)
        self.norm2 = paddle.nn.LayerNorm(normalized_shape=d_model)
        self.norm3 = paddle.nn.LayerNorm(normalized_shape=d_model)
        self.dropout1 = paddle.nn.Dropout(p=dropout)
        self.dropout2 = paddle.nn.Dropout(p=dropout)
        self.dropout3 = paddle.nn.Dropout(p=dropout)
        self.activation_str = activation
        self.activation = get_activation_fn(activation)
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[paddle.Tensor] = None,
        query_pos: Optional[paddle.Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> paddle.Tensor:
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(paddle.nn.Layer):
    def __init__(
        self, d_model: int, pos_enc_at_input: bool, layer: paddle.nn.Layer, num_layers: int, batch_first: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = paddle.nn.LayerNorm(normalized_shape=d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        curr: paddle.Tensor,
        memory: paddle.Tensor,
        curr_pos: Optional[paddle.Tensor] = None,
        memory_pos: Optional[paddle.Tensor] = None,
        num_obj_ptr_tokens: int = 0,
    ):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = curr[0], curr_pos[0]
        assert tuple(curr.shape)[1] == tuple(memory.shape)[1], "Batch size must be the same for curr and memory"
        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos
        if self.batch_first:
            x = output
            perm_45 = list(range(x.ndim))
            perm_45[0] = 1
            perm_45[1] = 0
            output = x.transpose(perm=perm_45)
            x = curr_pos
            perm_46 = list(range(x.ndim))
            perm_46[0] = 1
            perm_46[1] = 0
            curr_pos = x.transpose(perm=perm_46)
            x = memory
            perm_47 = list(range(x.ndim))
            perm_47[0] = 1
            perm_47[1] = 0
            memory = x.transpose(perm=perm_47)
            x = memory_pos
            perm_48 = list(range(x.ndim))
            perm_48[0] = 1
            perm_48[1] = 0
            memory_pos = x.transpose(perm=perm_48)
        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}
            output = layer(tgt=output, memory=memory, pos=memory_pos, query_pos=curr_pos, **kwds)
        normed_output = self.norm(output)
        if self.batch_first:
            x = normed_output
            perm_49 = list(range(x.ndim))
            perm_49[0] = 1
            perm_49[1] = 0
            normed_output = x.transpose(perm=perm_49)
            x = curr_pos
            perm_50 = list(range(x.ndim))
            perm_50[0] = 1
            perm_50[1] = 0
            curr_pos = x.transpose(perm=perm_50)
        return normed_output
