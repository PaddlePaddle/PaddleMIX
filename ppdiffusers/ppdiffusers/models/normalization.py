# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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

from typing import Dict, Optional, Tuple

import paddle
import paddle.nn as nn

from .activations import get_activation
from .embeddings import CombinedTimestepLabelEmbeddings, CombinedTimestepSizeEmbeddings


class AdaLayerNorm(nn.Layer):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.Silu()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        norm_elementwise_affine_kwargs = dict(weight_attr=False, bias_attr=False)
        self.norm = nn.LayerNorm(embedding_dim, **norm_elementwise_affine_kwargs)

    def forward(self, x: paddle.Tensor, timestep: paddle.Tensor) -> paddle.Tensor:
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = paddle.chunk(emb, 2)
        x = self.norm(x) * (1 + scale) + shift
        return x


class AdaLayerNormZero(nn.Layer):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()

        self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)

        self.silu = nn.Silu()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)
        norm_elementwise_affine_kwargs = dict(weight_attr=False, bias_attr=False)
        self.norm = nn.LayerNorm(embedding_dim, epsilon=1e-6, **norm_elementwise_affine_kwargs)

    def forward(
        self,
        x: paddle.Tensor,
        timestep: paddle.Tensor,
        class_labels: paddle.Tensor,
        hidden_dtype: Optional[paddle.dtype] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        emb = self.linear(self.silu(self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, axis=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormSingle(nn.Layer):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False):
        super().__init__()

        self.emb = CombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions
        )

        self.silu = nn.Silu()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)

    def forward(
        self,
        timestep: paddle.Tensor,
        added_cond_kwargs: Optional[Dict[str, paddle.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype: Optional[paddle.dtype] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        # No modulation happening here.
        embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class AdaGroupNorm(nn.Layer):
    r"""
    GroupNorm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
        num_groups (`int`): The number of groups to separate the channels into.
        act_fn (`str`, *optional*, defaults to `None`): The activation function to use.
        eps (`float`, *optional*, defaults to `1e-5`): The epsilon value to use for numerical stability.
    """

    def __init__(
        self, embedding_dim: int, out_dim: int, num_groups: int, act_fn: Optional[str] = None, eps: float = 1e-5
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

        if act_fn is None:
            self.act = None
        else:
            self.act = get_activation(act_fn)

        self.linear = nn.Linear(embedding_dim, out_dim * 2)

        norm_elementwise_affine_kwargs = dict(weight_attr=False, bias_attr=False)
        self.group_norm = nn.GroupNorm(num_groups, out_dim, epsilon=eps, **norm_elementwise_affine_kwargs)
        self.group_norm.weight = None
        self.group_norm.bias = None

    def forward(self, x: paddle.Tensor, emb: paddle.Tensor) -> paddle.Tensor:
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, axis=1)

        x = self.group_norm(x)
        x = x * (1 + scale) + shift
        return x
