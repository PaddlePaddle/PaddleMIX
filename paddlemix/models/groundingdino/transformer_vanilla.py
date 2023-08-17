# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import Tensor, nn

from .layers import MultiHeadAttention
from .utils import _get_activation_fn, _get_clones


class TextTransformer(nn.Layer):
    def __init__(self, num_layers, d_model=256, nheads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward
        self.norm = None

        single_encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.layers = _get_clones(single_encoder_layer, num_layers)

    def forward(self, memory_text: paddle.Tensor, text_attention_mask: paddle.Tensor):
        """

        Args:
            text_attention_mask: bs, num_token
            memory_text: bs, num_token, d_model

        Raises:
            RuntimeError: _description_

        Returns:
            output: bs, num_token, d_model
        """

        output = memory_text

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=text_attention_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Layer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # repeat attn mask
        if src_mask.dim() == 3 and src_mask.shape[0] == src.shape[1]:
            # bs, num_q, num_k
            src_mask = src_mask.tile([self.nhead, 1, 1])

        q = k = self.with_pos_embed(src, pos)

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
