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

from __future__ import annotations

import math
import sys

import paddle
from paddlemix.models.diffsinger.utils import paddle_aux
from paddle.nn import GELU, LayerNorm
from paddle.nn import MultiHeadAttention as MultiheadAttention
from paddle.nn import ReLU
from paddle.nn import Silu as SiLU

import paddlemix.models.diffsinger.utils as utils


class NormalInitEmbedding(paddle.nn.Embedding):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: (int | None) = None, *args, **kwargs):
        super().__init__(num_embeddings, embedding_dim, *args, padding_idx=padding_idx, **kwargs)
        init_Normal = paddle.nn.initializer.Normal(mean=0, std=self._embedding_dim**-0.5)
        init_Normal(self.weight)
        if padding_idx is not None:
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(self.weight[padding_idx])


class XavierUniformInitLinear(paddle.nn.Linear):
    def __init__(self, in_features: int, out_features: int, *args, bias: bool = True, **kwargs):
        super().__init__(in_features, out_features, *args, bias_attr=bias, **kwargs)
        init_XavierUniform = paddle.nn.initializer.XavierUniform()
        init_XavierUniform(self.weight)
        if bias:
            init_Constant = paddle.nn.initializer.Constant(value=0.0)
            init_Constant(self.bias)


class SinusoidalPositionalEmbedding(paddle.nn.Layer):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(init_size, embedding_dim, padding_idx)
        self.register_buffer(name="_float_tensor", tensor=paddle.empty(shape=[1], dtype="float32"))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = paddle.exp(x=paddle.arange(dtype="float32", end=half_dim) * -emb)
        emb = paddle.arange(dtype="float32", end=num_embeddings).unsqueeze(axis=1) * emb.unsqueeze(axis=0)
        emb = paddle.concat(x=[paddle.sin(x=emb), paddle.cos(x=emb)], axis=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            emb = paddle.concat(x=[emb, paddle.zeros(shape=[num_embeddings, 1])], axis=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, x, incremental_state=None, timestep=None, positions=None):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = tuple(x.shape)[:2]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.shape[0]:
            self.weights = SinusoidalPositionalEmbedding.get_embedding(max_pos, self.embedding_dim, self.padding_idx)
        self.weights = self.weights.to(self._float_tensor)
        if incremental_state is not None:
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(shape=[bsz, 1, -1])
        positions = utils.make_positions(x, self.padding_idx) if positions is None else positions
        return self.weights.index_select(axis=0, index=positions.view(-1)).view(bsz, seq_len, -1).detach()

    @staticmethod
    def max_positions():
        """Maximum number of supported positions."""
        return int(100000.0)


class TransformerFFNLayer(paddle.nn.Layer):
    def __init__(self, hidden_size, filter_size, kernel_size=1, dropout=0.0, act="gelu"):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.act = act
        self.ffn_1 = paddle.nn.Conv1D(
            in_channels=hidden_size, out_channels=filter_size, kernel_size=kernel_size, padding=kernel_size // 2
        )
        if self.act == "relu":
            self.act_fn = paddle.nn.ReLU()
        elif self.act == "gelu":
            self.act_fn = paddle.nn.GELU()
        elif self.act == "swish":
            self.act_fn = paddle.nn.Silu()
        self.ffn_2 = XavierUniformInitLinear(filter_size, hidden_size)

    def forward(self, x):
        x = self.ffn_1(x.transpose(perm=[1, 2, 0])).transpose(perm=[2, 0, 1])
        x = x * self.kernel_size**-0.5
        x = self.act_fn(x)
        x = paddle.nn.functional.dropout(x=x, p=self.dropout, training=self.training)
        x = self.ffn_2(x)
        return x


class EncSALayer(paddle.nn.Layer):
    def __init__(self, c, num_heads, dropout, attention_dropout=0.1, relu_dropout=0.1, kernel_size=9, act="gelu"):
        super().__init__()
        self.dropout = dropout
        self.layer_norm1 = paddle.nn.LayerNorm(normalized_shape=c)
        self.self_attn = MultiheadAttention(
            c,
            num_heads,
            dropout=attention_dropout,
            bias_attr=False,
        )
        self.layer_norm2 = paddle.nn.LayerNorm(normalized_shape=c)
        self.ffn = TransformerFFNLayer(c, 4 * c, kernel_size=kernel_size, dropout=relu_dropout, act=act)

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get("layer_norm_training", None)
        if layer_norm_training is not None:
            self.layer_norm1.training = layer_norm_training
            self.layer_norm2.training = layer_norm_training
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=paddle.any(encoder_padding_mask, -1),  # key_padding_mask=encoder_padding_mask
        )
        x = paddle.nn.functional.dropout(x=x, p=self.dropout, training=self.training)
        x = residual + x
        x = (
            x
            * (1 - encoder_padding_mask.astype(dtype="float32")).transpose(
                perm=paddle_aux.transpose_aux_func((1 - encoder_padding_mask.astype(dtype="float32")).ndim, 0, 1)
            )[..., None]
        )
        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = paddle.nn.functional.dropout(x=x, p=self.dropout, training=self.training)
        x = residual + x
        x = (
            x
            * (1 - encoder_padding_mask.astype(dtype="float32")).transpose(
                perm=paddle_aux.transpose_aux_func((1 - encoder_padding_mask.astype(dtype="float32")).ndim, 0, 1)
            )[..., None]
        )
        return x


class SinusoidalPosEmb(paddle.nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.place
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = paddle.exp(x=paddle.arange(end=half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = paddle.concat(x=(emb.sin(), emb.cos()), axis=-1)
        return emb
