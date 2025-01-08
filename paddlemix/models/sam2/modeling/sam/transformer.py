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
import warnings
from functools import partial
from typing import Tuple, Type

import paddle

from paddlemix.models.sam2.modeling.position_encoding import (
    apply_rotary_enc,
    compute_axial_cis,
)
from paddlemix.models.sam2.modeling.sam2_utils import MLP

warnings.simplefilter(action="ignore", category=FutureWarning)


class TwoWayTransformer(paddle.nn.Layer):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[paddle.nn.Layer] = paddle.nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = paddle.nn.LayerList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=i == 0,
                )
            )
        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = paddle.nn.LayerNorm(normalized_shape=embedding_dim)

    def forward(
        self, image_embedding: paddle.Tensor, image_pe: paddle.Tensor, point_embedding: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        bs, c, h, w = tuple(image_embedding.shape)
        image_embedding = image_embedding.flatten(start_axis=2).transpose(perm=[0, 2, 1])
        image_pe = image_pe.flatten(start_axis=2).transpose(perm=[0, 2, 1])
        queries = point_embedding
        keys = image_embedding
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe)
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        return queries, keys


class TwoWayAttentionBlock(paddle.nn.Layer):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[paddle.nn.Layer] = paddle.nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = paddle.nn.LayerNorm(normalized_shape=embedding_dim)
        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = paddle.nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation)
        self.norm3 = paddle.nn.LayerNorm(normalized_shape=embedding_dim)
        self.norm4 = paddle.nn.LayerNorm(normalized_shape=embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: paddle.Tensor, keys: paddle.Tensor, query_pe: paddle.Tensor, key_pe: paddle.Tensor
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        return queries, keys


class Attention(paddle.nn.Layer):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self, embedding_dim: int, num_heads: int, downsample_rate: int = 1, dropout: float = 0.0, kv_in_dim: int = None
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        self.q_proj = paddle.nn.Linear(in_features=embedding_dim, out_features=self.internal_dim)
        self.k_proj = paddle.nn.Linear(in_features=self.kv_in_dim, out_features=self.internal_dim)
        self.v_proj = paddle.nn.Linear(in_features=self.kv_in_dim, out_features=self.internal_dim)
        self.out_proj = paddle.nn.Linear(in_features=self.internal_dim, out_features=embedding_dim)
        self.dropout_p = dropout

    def _separate_heads(self, x: paddle.Tensor, num_heads: int) -> paddle.Tensor:
        b, n, c = tuple(x.shape)
        x = x.reshape([b, n, num_heads, c // num_heads])

        return x.transpose([0, 2, 1, 3])

    def _recombine_heads(self, x: paddle.Tensor) -> paddle.Tensor:
        b, n_heads, n_tokens, c_per_head = tuple(x.shape)

        x = x.transpose(perm=[0, 2, 1, 3])

        return x.reshape([b, n_tokens, n_heads * c_per_head])

    def forward(self, q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor) -> paddle.Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        dropout_p = self.dropout_p if self.training else 0.0

        out = paddle.nn.functional.scaled_dot_product_attention_(
            q.transpose([0, 2, 1, 3]), k.transpose([0, 2, 1, 3]), v.transpose([0, 2, 1, 3]), dropout_p=dropout_p
        )
        out = out.transpose([0, 2, 1, 3])

        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class RoPEAttention(Attention):
    """Attention with rotary position encoding."""

    def __init__(self, *args, rope_theta=10000.0, rope_k_repeat=False, feat_sizes=(32, 32), **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_cis = partial(compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta)
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis
        self.rope_k_repeat = rope_k_repeat

    def forward(
        self, q: paddle.Tensor, k: paddle.Tensor, v: paddle.Tensor, num_k_exclude_rope: int = 0
    ) -> paddle.Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)
        w = h = math.sqrt(tuple(q.shape)[-2])

        if tuple(self.freqs_cis.shape)[0] != tuple(q.shape)[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h)
        if tuple(q.shape)[-2] != tuple(k.shape)[-2]:
            assert self.rope_k_repeat
        num_k_rope = k.shape[-2] - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q, k[:, :, :num_k_rope], freqs_cis=self.freqs_cis, repeat_freqs_k=self.rope_k_repeat
        )
        dropout_p = self.dropout_p if self.training else 0.0

        out = paddle.nn.functional.scaled_dot_product_attention_(
            q.transpose([0, 2, 1, 3]), k.transpose([0, 2, 1, 3]), v.transpose([0, 2, 1, 3]), dropout_p=dropout_p
        )
        out = out.transpose([0, 2, 1, 3])
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out
