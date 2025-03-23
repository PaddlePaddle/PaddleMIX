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

import os
from typing import Dict, Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .activations import get_activation
from .embeddings import CombinedTimestepLabelEmbeddings, CombinedTimestepSizeEmbeddings


class AdaLayerNorm(nn.Layer):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(
        self,
        embedding_dim: int, 
        num_embeddings: Optional[int] = None,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        
        self.chunk_dim = chunk_dim
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.Silu()
        self.linear = nn.Linear(embedding_dim, output_dim)
        if norm_elementwise_affine:
            norm_elementwise_affine_kwargs = dict(weight_attr=None, bias_attr=None)
        else:
            norm_elementwise_affine_kwargs = dict(weight_attr=False, bias_attr=False)
        self.norm = nn.LayerNorm(output_dim // 2, epsilon=norm_eps, **norm_elementwise_affine_kwargs)

    def forward(self, x: paddle.Tensor, timestep: Optional[paddle.Tensor] = None, temb: Optional[paddle.Tensor] = None) -> paddle.Tensor:
        if self.emb is not None:
            temb = self.emb(timestep)
        temb = self.linear(self.silu(temb))
        if self.chunk_dim == 1:
            # This is a bit weird why we have the order of "shift, scale" here and "scale, shift" in the
            # other if-branch. This branch is specific to CogVideoX for now.
            shift, scale = paddle.chunk(temb, 2, axis=1)
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        else:
            scale, shift = paddle.chunk(temb, 2)
        x = self.norm(x) * (1 + scale) + shift
        return x


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.astype('float32'),
            normalized_shape=self._normalized_shape,
            weight=self.weight.astype('float32') if self.weight is not None else None,
            bias=self.bias.astype('float32') if self.bias is not None else None,
            epsilon=self._epsilon,
        ).astype(origin_dtype)


class SD35AdaLayerNormZeroX(nn.Layer):
    r"""
    Norm layer adaptive layer norm zero (AdaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, norm_type: str = "layer_norm", bias: bool = True) -> None:
        super().__init__()

        self.silu = nn.Silu()
        self.linear = nn.Linear(embedding_dim, 9 * embedding_dim, bias_attr=bias)
        if norm_type == "layer_norm":
            norm_elementwise_affine_kwargs = dict(weight_attr=False, bias_attr=False)
            self.norm = nn.LayerNorm(embedding_dim, epsilon=1e-6, **norm_elementwise_affine_kwargs)
        else:
            raise ValueError(f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'.")
    
    def forward(
        self,
        hidden_states: paddle.Tensor,
        emb: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, ...]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = emb.chunk(
            9, axis=1
        )
        norm_hidden_states = self.norm(hidden_states)
        hidden_states = norm_hidden_states * (1 + scale_msa[:, None]) + shift_msa[:, None]
        norm_hidden_states2 = norm_hidden_states * (1 + scale_msa2[:, None]) + shift_msa2[:, None]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2    


class AdaLayerNormZero(nn.Layer):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None):
        super().__init__()

        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.Silu()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim)
        norm_elementwise_affine_kwargs = dict(weight_attr=False, bias_attr=False)
        self.norm = nn.LayerNorm(embedding_dim, epsilon=1e-6, **norm_elementwise_affine_kwargs)

    def forward(
        self,
        x: paddle.Tensor,
        timestep: Optional[paddle.Tensor] = None,
        class_labels: Optional[paddle.Tensor] = None,
        hidden_dtype: Optional[paddle.dtype] = None,
        emb: Optional[paddle.Tensor] = None,
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        # emb = self.linear(self.silu(self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)))
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, axis=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(paddle.nn.Layer):
    """
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True):
        super().__init__()

        self.silu = paddle.nn.Silu()
        self.linear = paddle.nn.Linear(in_features=embedding_dim, out_features=3 * embedding_dim, bias_attr=bias)
        if norm_type == "layer_norm":
            self.norm = paddle.nn.LayerNorm(
                normalized_shape=embedding_dim,
                weight_attr=False,
                bias_attr=False,
                epsilon=1e-06,
            )
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def forward(
        self, x: paddle.Tensor, emb: Optional[paddle.Tensor] = None
    ) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(chunks=3, axis=1)
        x = self.norm(x) * (1 + scale_msa[:, (None)]) + shift_msa[:, (None)]
        return x, gate_msa


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


class AdaLayerNormContinuous(nn.Layer):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = nn.Silu()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias_attr=bias)
        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, eps, weight_attr=elementwise_affine, bias_attr=bias if elementwise_affine else False)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: paddle.Tensor, conditioning_embedding: paddle.Tensor) -> paddle.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear(self.silu(conditioning_embedding).cast(x.dtype))
        scale, shift = paddle.chunk(emb, 2, axis=1)
        if os.getenv("INFERENCE_OPTIMIZE_TRITON"):
            # NOTE:(changwenbin,zhoukangkang)
            # This is a fused faster op using Triton, only used in inference, not used in training.
            import paddlemix

            x = paddlemix.triton_ops.adaptive_layer_norm(x, scale, shift, self.norm.weight, self.norm.bias)
        else:
            x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class RMSNorm(nn.Layer):
    def __init__(self, dim, epsilon: float, elementwise_affine: bool = True):
        super().__init__()
        self.epsilon = epsilon
        self.dim = dim
        if elementwise_affine:
            self.weight = paddle.create_parameter(
                shape=[dim],
                dtype=paddle.get_default_dtype(),
                default_initializer=nn.initializer.Constant(1.0),
            )
        else:
            self.weight = None

    def forward(self, hidden_states, begin_norm_axis=None):
        return paddle.incubate.nn.functional.fused_rms_norm(
            x=hidden_states,
            norm_weight=self.weight,
            norm_bias=None,
            epsilon=self.epsilon,
            begin_norm_axis=len(hidden_states.shape)-1 if begin_norm_axis is None else begin_norm_axis,
        )[0]


class LpNorm(nn.Layer):
    def __init__(self, p: int = 2, axis: int = -1, epsilon: float = 1e-12):
        super().__init__()

        self.p = p
        self.axis = axis
        self.epsilon = epsilon

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return F.normalize(hidden_states, p=self.p, axis=self.dim, epsilon=self.eps)


class CogVideoXLayerNormZero(paddle.nn.Layer):

    def __init__(self, conditioning_dim: int, embedding_dim: int,
        elementwise_affine: bool=True, eps: float=1e-05, bias: bool=True
        ) ->None:
        super().__init__()
        self.silu = paddle.nn.Silu()
        self.linear = paddle.nn.Linear(in_features=conditioning_dim,
            out_features=6 * embedding_dim, bias_attr=bias)
        self.norm = paddle.nn.LayerNorm(normalized_shape=embedding_dim,
            epsilon=eps, weight_attr=elementwise_affine, bias_attr=
            elementwise_affine)

    def forward(self, hidden_states: paddle.Tensor, encoder_hidden_states:
        paddle.Tensor, temb: paddle.Tensor) ->Tuple[paddle.Tensor, paddle.
        Tensor]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self
            .silu(temb)).chunk(chunks=6, axis=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :
            ] + shift[:, None, :]
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 +
            enc_scale)[:, None, :] + enc_shift[:, None, :]
        return hidden_states, encoder_hidden_states, gate[:, None, :
            ], enc_gate[:, None, :]
