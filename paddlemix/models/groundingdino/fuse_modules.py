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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.framework import in_dynamic_mode
from paddle.nn.initializer import Constant
from paddlenlp.utils.initializer import constant_, xavier_uniform_

from .layers import DropPath
from .utils import masked_fill


class FeatureResizer(nn.Layer):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = paddle.abs(X).sum(axis=dim, keepdim=True) + eps
    X = paddle.divide(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X"""
    norm = paddle.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = paddle.divide(X, norm)
    return X


def func_attention(query, context, smooth=1, raw_feature_norm="softmax", eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.shape[:2]
    batch_size, sourceL = context.shape[:2]

    # Get attention
    # --> (batch, d, queryL)
    queryT = query.transpose([0, 2, 1])

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = paddle.bmm(context, queryT)
    if raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.reshape([batch_size * sourceL, queryL])
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.reshape(batch_size, sourceL, queryL)
    elif raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    else:
        raise ValueError("unknown first norm type:", raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = attn.transpose([0, 2, 1])
    # --> (batch*queryL, sourceL)
    attn = attn.reshape([batch_size * queryL, sourceL])
    attn = nn.Softmax()(attn * smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.reshape([batch_size, queryL, sourceL])
    # --> (batch, sourceL, queryL)
    attnT = attn.transpose([0, 2, 1])

    # --> (batch, d, sourceL)
    contextT = context.transpose([0, 2, 1])
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = paddle.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = weightedContext.transpose([0, 2, 1])

    return weightedContext, attnT


class BiMultiHeadAttention(nn.Layer):
    def __init__(self, v_dim, l_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(BiMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.l_proj = nn.Linear(self.l_dim, self.embed_dim)
        self.values_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.values_l_proj = nn.Linear(self.l_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_l_proj = nn.Linear(self.embed_dim, self.l_dim)

        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor, seq_len, bsz):
        return tensor.reshape([bsz, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])

    def _reset_parameters(self):
        xavier_uniform_(self.v_proj.weight)
        constant_(self.v_proj.bias)
        xavier_uniform_(self.l_proj.weight)
        constant_(self.l_proj.bias)
        xavier_uniform_(self.values_v_proj.weight)
        constant_(self.values_v_proj.bias)
        xavier_uniform_(self.values_l_proj.weight)
        constant_(self.values_l_proj.bias)
        xavier_uniform_(self.out_v_proj.weight)
        constant_(self.out_v_proj.bias)
        xavier_uniform_(self.out_l_proj.weight)
        constant_(self.out_l_proj.bias)

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        """_summary_

        Args:
            v (_type_): bs, n_img, dim
            l (_type_): bs, n_text, dim
            attention_mask_v (_type_, optional): _description_. bs, n_img
            attention_mask_l (_type_, optional): _description_. bs, n_text

        Returns:
            _type_: _description_
        """

        bsz, tgt_len, _ = v.shape

        query_states = self.v_proj(v) * self.scale
        key_states = self._shape(self.l_proj(l), -1, bsz)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).reshape(proj_shape)
        key_states = key_states.reshape(proj_shape)
        value_v_states = value_v_states.reshape(proj_shape)
        value_l_states = value_l_states.reshape(proj_shape)

        src_len = key_states.shape[1]
        attn_weights = paddle.bmm(query_states, key_states.transpose([0, 2, 1]))  # bs*nhead, nimg, ntxt

        if in_dynamic_mode() and attn_weights.shape != [bsz * self.num_heads, tgt_len, src_len]:
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.shape}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()

        if self.clamp_min_for_underflow:
            attn_weights = paddle.clip(
                attn_weights, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = paddle.clip(
                attn_weights, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        attn_weights_T = attn_weights.transpose([0, 2, 1])
        attn_weights_l = attn_weights_T - paddle.max(attn_weights_T, axis=-1, keepdim=True)
        if self.clamp_min_for_underflow:
            attn_weights_l = paddle.clip(
                attn_weights_l, min=-50000
            )  # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights_l = paddle.clip(
                attn_weights_l, max=50000
            )  # Do not increase 50000, data type half has quite limited range

        # mask vison for language
        if attention_mask_v is not None:

            attention_mask_v = (
                attention_mask_v[:, None, None, :].cast(paddle.float32).tile([1, self.num_heads, 1, 1]).flatten(0, 1)
            )
            attn_weights_l = masked_fill(attn_weights_l, attention_mask_v == 1.0, float("-inf"))

        attn_weights_l = F.softmax(attn_weights_l, axis=-1)

        # mask language for vision
        if attention_mask_l is not None:
            attention_mask_l = (
                attention_mask_l[:, None, None, :].cast(paddle.float32).tile([1, self.num_heads, 1, 1]).flatten(0, 1)
            )
            attn_weights = masked_fill(attn_weights, attention_mask_l == 1.0, float("-inf"))

        attn_weights_v = F.softmax(attn_weights, axis=-1)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training)
        attn_probs_l = F.dropout(attn_weights_l, p=self.dropout, training=self.training)

        attn_output_v = paddle.bmm(attn_probs_v, value_l_states)
        attn_output_l = paddle.bmm(attn_probs_l, value_v_states)

        if in_dynamic_mode() and attn_output_v.shape != [bsz * self.num_heads, tgt_len, self.head_dim]:
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.shape}"
            )

        if in_dynamic_mode() and attn_output_l.shape != [bsz * self.num_heads, src_len, self.head_dim]:
            raise ValueError(
                f"`attn_output_l` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_l.shape}"
            )

        attn_output_v = attn_output_v.reshape([bsz, self.num_heads, tgt_len, self.head_dim])
        attn_output_v = attn_output_v.transpose([0, 2, 1, 3])
        attn_output_v = attn_output_v.reshape([bsz, tgt_len, self.embed_dim])

        attn_output_l = attn_output_l.reshape([bsz, self.num_heads, src_len, self.head_dim])
        attn_output_l = attn_output_l.transpose([0, 2, 1, 3])
        attn_output_l = attn_output_l.reshape([bsz, src_len, self.embed_dim])

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_l = self.out_l_proj(attn_output_l)

        return attn_output_v, attn_output_l


# Bi-Direction MHA (text->image, image->text)
class BiAttentionBlock(nn.Layer):
    def __init__(
        self,
        v_dim,
        l_dim,
        embed_dim,
        num_heads,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
        cfg=None,
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(BiAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(
            v_dim=v_dim,
            l_dim=l_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # add layer scale for training stability
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = self.create_parameter(
            shape=[v_dim],
            attr=paddle.ParamAttr(initializer=Constant(init_values)),
        )
        self.gamma_l = self.create_parameter(
            shape=[l_dim],
            attr=paddle.ParamAttr(initializer=Constant(init_values)),
        )

    def forward(self, v, l, attention_mask_v=None, attention_mask_l=None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l)
        # v, l = v + delta_v, l + delta_l
        v = v + self.drop_path(self.gamma_v * delta_v)
        l = l + self.drop_path(self.gamma_l * delta_l)
        return v, l
