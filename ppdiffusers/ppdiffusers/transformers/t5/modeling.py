# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" Paddle T5 model."""


import copy
import math
from functools import partial
from typing import List, Optional, Tuple, Union

import paddle
from paddle import nn
from paddle.amp.auto_cast import amp_state
from paddle.distributed import fleet
from paddle.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from paddlenlp.transformers.activations import ACT2FN
from paddlenlp.transformers.conversion_utils import (
    StateDictNameMapping,
    init_name_mappings,
    split_or_merge_func,
)
from paddlenlp.transformers.model_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from paddlenlp.transformers.model_utils import register_base_model

from ...utils import logging
from ..model_utils import ALL_LAYERNORM_LAYERS, PretrainedModel
from .configuration import T5Config

logger = logging.get_logger(__name__)


####################################################
# This dict contains ids and associated url
# for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]


def parallel_matmul(lm_output, logit_weights, parallel_output=True, transpose_y=True):
    hcg = fleet.get_hybrid_communicate_group()
    model_parallel_group = hcg.get_model_parallel_group()
    world_size = hcg.get_model_parallel_world_size()

    if world_size > 1:
        # _c_identity is backwards is reduce
        input_parallel = paddle.distributed.collective._c_identity(lm_output, group=model_parallel_group)

        logits = paddle.matmul(input_parallel, logit_weights, transpose_y=transpose_y)

        if parallel_output:
            return logits

        # _c_concat has not grad backwards
        return paddle.distributed.collective._c_concat(logits, group=model_parallel_group)
    else:
        logits = paddle.matmul(lm_output, logit_weights, transpose_y=transpose_y)
        return logits


class T5LayerNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(paddle.ones((hidden_size,)))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus variance is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.cast(dtype=paddle.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if amp_state() or self.weight.dtype in [paddle.float16, paddle.bfloat16]:
            hidden_states = hidden_states.cast(self.weight.dtype)

        return self.weight * hidden_states


ALL_LAYERNORM_LAYERS.append(T5LayerNorm)


class T5DenseActDense(nn.Layer):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.tensor_parallel_degree > 1:
            self.wi = fleet.meta_parallel.ColumnParallelLinear(
                config.d_model, config.d_ff, has_bias=False, gather_output=False
            )
            self.wo = fleet.meta_parallel.RowParallelLinear(
                config.d_ff, config.d_model, input_is_parallel=True, has_bias=False
            )
        else:
            self.wi = nn.Linear(config.d_model, config.d_ff, bias_attr=False)
            self.wo = nn.Linear(config.d_ff, config.d_model, bias_attr=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, paddle.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != paddle.int8
        ):
            hidden_states = hidden_states.cast(dtype=self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Layer):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.tensor_parallel_degree > 1:
            self.wi_0 = fleet.meta_parallel.ColumnParallelLinear(
                config.d_model, config.d_ff, has_bias=False, gather_output=False
            )
            self.wi_1 = fleet.meta_parallel.ColumnParallelLinear(
                config.d_model, config.d_ff, has_bias=False, gather_output=False
            )
            self.wo = fleet.meta_parallel.RowParallelLinear(
                config.d_ff, config.d_model, input_is_parallel=True, has_bias=False
            )
        else:
            self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias_attr=False)
            self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias_attr=False)
            self.wo = nn.Linear(config.d_ff, config.d_model, bias_attr=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)

        # To make 8bit quantization work for google/flan-t5-xxl, self.wo is kept in float32.
        # See https://github.com/huggingface/transformers/issues/20287
        # we also make sure the weights are not in `int8` in case users will force `_keep_in_fp32_modules` to be `None``
        # NOTE: (yujun06) NEW ADDED, make sure self.wo.weight.dtype is float32
        if isinstance(self.wo.weight, paddle.Tensor) and self.wo.weight.dtype != paddle.float32:
            self.wo.to(dtype=paddle.float32)
        if (
            isinstance(self.wo.weight, paddle.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != paddle.int8
        ):
            hidden_states = hidden_states.cast(dtype=self.wo.weight.dtype)

        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Layer):
    def __init__(self, config: T5Config):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(config)
        else:
            self.DenseReluDense = T5DenseActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Layer):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        if config.tensor_parallel_degree > 1:
            assert self.n_heads % config.tensor_parallel_degree == 0
            self.q = fleet.meta_parallel.ColumnParallelLinear(
                self.d_model, self.inner_dim, has_bias=False, gather_output=False
            )
            self.k = fleet.meta_parallel.ColumnParallelLinear(
                self.d_model, self.inner_dim, has_bias=False, gather_output=False
            )
            self.v = fleet.meta_parallel.ColumnParallelLinear(
                self.d_model, self.inner_dim, has_bias=False, gather_output=False
            )
            self.o = fleet.meta_parallel.RowParallelLinear(
                self.inner_dim, self.d_model, has_bias=False, input_is_parallel=True
            )
            self.n_heads = self.n_heads // config.tensor_parallel_degree
            self.inner_dim = self.inner_dim // config.tensor_parallel_degree
        else:
            # Mesh TensorFlow initialization to avoid scaling before softmax
            self.q = nn.Linear(self.d_model, self.inner_dim, bias_attr=False)
            self.k = nn.Linear(self.d_model, self.inner_dim, bias_attr=False)
            self.v = nn.Linear(self.d_model, self.inner_dim, bias_attr=False)
            self.o = nn.Linear(self.inner_dim, self.d_model, bias_attr=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.gradient_checkpointing = False

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).cast(dtype=paddle.int64) * num_buckets
            relative_position = paddle.abs(relative_position)
        else:
            relative_position = -paddle.minimum(relative_position, paddle.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            paddle.log(relative_position.cast("float32") / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).cast(dtype=paddle.int64)
        relative_position_if_large = paddle.minimum(
            relative_position_if_large, paddle.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += paddle.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = paddle.arange(query_length, dtype=paddle.int64)[:, None]
        memory_position = paddle.arange(key_length, dtype=paddle.int64)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.transpose([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.reshape([batch_size, -1, self.n_heads, self.key_value_proj_dim]).transpose([0, 2, 1, 3])

        def unshape(states):
            """reshape"""
            return states.transpose([0, 2, 1, 3]).reshape([batch_size, -1, self.inner_dim])

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = paddle.concat([past_key_value, hidden_states], axis=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        scores = paddle.matmul(
            query_states, key_states, transpose_y=True
        )  # equivalent of paddle.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = paddle.zeros((1, self.n_heads, real_seq_length, key_length), dtype=scores.dtype)
                if self.gradient_checkpointing and self.training:
                    position_bias.stop_gradient = False
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -paddle.shape(hidden_states)[1] :, :]

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        position_bias_masked = position_bias

        scores += position_bias_masked
        attn_weights = nn.functional.softmax(scores.cast(dtype=paddle.float32), axis=-1).cast(
            dtype=scores.dtype
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        attn_output = unshape(paddle.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5LayerSelfAttention(nn.Layer):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Layer):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.LayerList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if amp_state() or hidden_states.dtype == paddle.float16:
            clamp_value = paddle.where(
                paddle.isinf(hidden_states).any(),
                paddle.finfo(hidden_states.dtype).max - 1000,
                paddle.finfo(hidden_states.dtype).max,
            )
            hidden_states = paddle.clip(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = paddle.shape(present_key_value_state[0])[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if amp_state() or hidden_states.dtype == paddle.float16:
                clamp_value = paddle.where(
                    paddle.isinf(hidden_states).any(),
                    paddle.finfo(hidden_states.dtype).max - 1000,
                    paddle.finfo(hidden_states.dtype).max,
                )
                hidden_states = paddle.clip(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if amp_state() or hidden_states.dtype == paddle.float16:
            clamp_value = paddle.where(
                paddle.isinf(hidden_states).any(),
                paddle.finfo(hidden_states.dtype).max - 1000,
                paddle.finfo(hidden_states.dtype).max,
            )
            hidden_states = paddle.clip(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5ClassificationHead(nn.Layer):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: T5Config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = paddle.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class T5PretrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = T5Config
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["T5Block"]
    _keep_in_fp32_modules = ["wo"]

    _deprecated_dict = {
        "key": "t5.",
        "name_mapping": {
            # common
            "t5.": "transformer.",
        },
    }

    @property
    def dummy_inputs(self):
        DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
        DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]]
        input_ids = paddle.to_tensor(DUMMY_INPUTS)
        input_mask = paddle.to_tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    @classmethod
    def _get_tensor_parallel_mappings(cls, config, is_split=True):
        # paddle tensor parallel

        fn = split_or_merge_func(
            is_split=is_split,
            tensor_parallel_degree=config.tensor_parallel_degree,
            tensor_parallel_rank=config.tensor_parallel_rank,
            num_attention_heads=config.num_heads,
        )

        def get_tensor_parallel_split_mappings(num_layers):
            final_actions = {}
            base_actions = {
                # Column Linear
                "encoder.block.0.layer.0.SelfAttention.q.weight": partial(fn, is_column=True),
                "encoder.block.0.layer.0.SelfAttention.k.weight": partial(fn, is_column=True),
                "encoder.block.0.layer.0.SelfAttention.v.weight": partial(fn, is_column=True),
                "decoder.block.0.layer.0.SelfAttention.q.weight": partial(fn, is_column=True),
                "decoder.block.0.layer.0.SelfAttention.k.weight": partial(fn, is_column=True),
                "decoder.block.0.layer.0.SelfAttention.v.weight": partial(fn, is_column=True),
                # Row Linear
                "encoder.block.0.layer.0.SelfAttention.o.weight": partial(fn, is_column=False),
                "decoder.block.0.layer.0.SelfAttention.o.weight": partial(fn, is_column=False),
                "encoder.block.0.layer.1.DenseReluDense.wo.weight": partial(fn, is_column=False),
                "decoder.block.0.layer.2.DenseReluDense.wo.weight": partial(fn, is_column=False),
                "shared.weight": partial(fn, is_column=False),
            }
            # mv T5LayerCrossAttention here
            base_actions["decoder.block.0.layer.1.EncDecAttention.q.weight"] = partial(fn, is_column=True)
            base_actions["decoder.block.0.layer.1.EncDecAttention.k.weight"] = partial(fn, is_column=True)
            base_actions["decoder.block.0.layer.1.EncDecAttention.v.weight"] = partial(fn, is_column=True)
            base_actions["decoder.block.0.layer.1.EncDecAttention.o.weight"] = partial(fn, is_column=False)

            if config.feed_forward_proj == "relu":
                base_actions["encoder.block.0.layer.1.DenseReluDense.wi.weight"] = partial(fn, is_column=True)
                base_actions["decoder.block.0.layer.2.DenseReluDense.wi.weight"] = partial(fn, is_column=True)
            elif config.feed_forward_proj == "gated-gelu":
                for i in range(2):
                    base_actions[f"encoder.block.0.layer.1.DenseReluDense.wi_{i}.weight"] = partial(fn, is_column=True)
                    base_actions[f"decoder.block.0.layer.2.DenseReluDense.wi_{i}.weight"] = partial(fn, is_column=True)

            final_actions["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = partial(
                fn, is_column=True
            )
            final_actions["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = partial(
                fn, is_column=True
            )

            for key, action in base_actions.items():
                if "block.0." in key:
                    for i in range(num_layers):
                        final_actions[key.replace("block.0.", f"block.{i}.")] = action
                final_actions[key] = action

            return final_actions

        mappings = get_tensor_parallel_split_mappings(config.num_hidden_layers)

        return mappings

    @classmethod
    def _get_name_mappings(cls, config: T5Config) -> List[StateDictNameMapping]:
        # from pytorch to paddle
        architectures = config.architectures + [cls.__name__]
        mappings: List[StateDictNameMapping] = []
        model_mappings = [
            "shared.weight",
            "encoder.embed_tokens.weight",
            "encoder.final_layer_norm.weight",
            "decoder.embed_tokens.weight",
            "decoder.final_layer_norm.weight",
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
        ]
        for layer_index in range(config.num_hidden_layers):
            for att_head in ["q", "k", "v", "o"]:
                model_mappings.extend(
                    [
                        [
                            f"encoder.block.{layer_index}.layer.0.SelfAttention.{att_head}.weight",
                            None,
                            "transpose",
                        ],
                        [
                            f"decoder.block.{layer_index}.layer.0.SelfAttention.{att_head}.weight",
                            None,
                            "transpose",
                        ],
                        [
                            f"decoder.block.{layer_index}.layer.1.EncDecAttention.{att_head}.weight",
                            None,
                            "transpose",
                        ],
                    ]
                )

            layer_mappings = [
                [
                    f"encoder.block.{layer_index}.layer.1.DenseReluDense.wo.weight",
                    None,
                    "transpose",
                ],
                [
                    f"decoder.block.{layer_index}.layer.2.DenseReluDense.wo.weight",
                    None,
                    "transpose",
                ],
                f"encoder.block.{layer_index}.layer.0.layer_norm.weight",
                f"encoder.block.{layer_index}.layer.1.layer_norm.weight",
                f"decoder.block.{layer_index}.layer.0.layer_norm.weight",
                f"decoder.block.{layer_index}.layer.1.layer_norm.weight",
                f"decoder.block.{layer_index}.layer.2.layer_norm.weight",
            ]

            if config.feed_forward_proj == "relu":
                layer_mappings.extend(
                    [
                        [
                            f"encoder.block.{layer_index}.layer.1.DenseReluDense.wi.weight",
                            None,
                            "transpose",
                        ],
                        [
                            f"decoder.block.{layer_index}.layer.2.DenseReluDense.wi.weight",
                            None,
                            "transpose",
                        ],
                    ]
                )
            elif "gated" in config.feed_forward_proj:
                for i in range(2):
                    layer_mappings.extend(
                        [
                            [
                                f"encoder.block.{layer_index}.layer.1.DenseReluDense.wi_{i}.weight",
                                None,
                                "transpose",
                            ],
                            [
                                f"decoder.block.{layer_index}.layer.2.DenseReluDense.wi_{i}.weight",
                                None,
                                "transpose",
                            ],
                        ]
                    )

            model_mappings.extend(layer_mappings)

        init_name_mappings(model_mappings)

        torch_prefix = ""
        paddle_prefix = ""
        if "T5ForSequenceClassification" in config.architectures:
            torch_prefix = "transformer."
        if "T5ForSequenceClassification" in [cls.__name__]:
            paddle_prefix = "transformer."

        # add prefix
        for mapping in model_mappings:
            mapping[0] = torch_prefix + mapping[0]
            mapping[1] = paddle_prefix + mapping[1]

        if "T5ForConditionalGeneration" in architectures:
            model_mappings.append(["lm_head.weight", "lm_head.weight", "transpose"])

        if "T5EncoderModel" in architectures:
            new_model_mappings = []
            for mapping in model_mappings:
                # remove dedocker model
                if "decoder." in mapping[0]:
                    continue
                new_model_mappings.append(mapping)
            model_mappings = new_model_mappings

        mappings = [StateDictNameMapping(*mapping) for mapping in model_mappings]
        return mappings

    @paddle.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.fill_(factor * 1.0)
        elif isinstance(
            module,
            (T5Model, T5ForConditionalGeneration, T5EncoderModel, T5ForQuestionAnswering),
        ):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            nn.init.normal_(module.shared.weight, mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                nn.init.normal_(module.lm_head.weight, mean=0.0, std=factor * 1.0)
            if hasattr(module, "qa_outputs"):
                nn.init.normal_(module.qa_outputs.weight, mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
                module.qa_outputs.bias.zero_()
        elif isinstance(module, T5ClassificationHead):
            nn.init.normal_(module.dense.weight, mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.dense, "bias") and module.dense.bias is not None:
                module.dense.bias.zero_()
            nn.init.normal_(module.out_proj.weight, mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.out_proj, "bias") and module.out_proj.bias is not None:
                module.out_proj.bias.zero_()
        elif isinstance(module, T5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            nn.init.normal_(module.wi.weight, mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.zero_()
            nn.init.normal_(module.wo.weight, mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.zero_()
        elif isinstance(module, T5DenseGatedActDense):
            nn.init.normal_(module.wi_0.weight, mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.zero_()
            nn.init.normal_(module.wi_1.weight, mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.zero_()
            nn.init.normal_(module.wo.weight, mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.zero_()
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads

            nn.init.normal_(module.q.weight, mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            nn.init.normal_(module.k.weight, mean=0.0, std=factor * (d_model**-0.5))
            nn.init.normal_(module.v.weight, mean=0.0, std=factor * (d_model**-0.5))
            nn.init.normal_(module.o.weight, mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))

            if module.has_relative_attention_bias:
                nn.init.normal_(module.relative_attention_bias.weight, mean=0.0, std=factor * ((d_model) ** -0.5))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
                "See T5 docs for more information."
            )

        # shift inputs to the right
        shifted_input_ids = paddle.zeros_like(input_ids)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids = paddle.masked_fill(shifted_input_ids, shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids


class T5Stack(T5PretrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.LayerList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = paddle.shape(input_ids)
            input_ids = input_ids.reshape([-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = paddle.shape(inputs_embeds)[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = (
            paddle.shape(past_key_values[0][0])[2] + seq_length if past_key_values is not None else seq_length
        )

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if attention_mask is None:
            attention_mask = paddle.ones([batch_size, mask_seq_length])

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = paddle.shape(encoder_hidden_states)
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = paddle.ones(encoder_hidden_shape, dtype=paddle.int64)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and not hidden_states.stop_gradient:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if not use_cache:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


@register_base_model
class T5Model(T5PretrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)

        if config.tensor_parallel_degree > 1:
            self.shared = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.d_model,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        decoder_input_ids: Optional[paddle.Tensor] = None,
        decoder_attention_mask: Optional[paddle.Tensor] = None,
        encoder_output: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        decoder_inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[paddle.Tensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from ppdiffusers.transformers import AutoTokenizer, T5Model

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5Model.from_pretrained("t5-small")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pd"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pd").input_ids  # Batch size 1

        >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
        >>> # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
        >>> decoder_input_ids = model._shift_right(decoder_input_ids)

        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_output is None:
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_output, BaseModelOutput):
            encoder_output = BaseModelOutput(
                last_hidden_state=encoder_output[0],
                hidden_states=encoder_output[1] if len(encoder_output) > 1 else None,
                attentions=encoder_output[2] if len(encoder_output) > 2 else None,
            )

        hidden_states = encoder_output[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_output

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_output.last_hidden_state,
            encoder_hidden_states=encoder_output.hidden_states,
            encoder_attentions=encoder_output.attentions,
        )


class T5ForConditionalGeneration(T5PretrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        if config.tensor_parallel_degree > 1:
            self.shared = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.d_model,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        if config.tie_word_embeddings:
            if config.tensor_parallel_degree > 1:
                self.lm_head_forward_func = lambda x: parallel_matmul(
                    x, self.shared.weight, config.tensor_parallel_output, transpose_y=True
                )
            else:
                self.lm_head_forward_func = lambda x: paddle.matmul(x, self.shared.weight, transpose_y=True)
        else:
            if config.tensor_parallel_degree > 1:
                self.lm_head = nn.Linear(
                    config.d_model, config.vocab_size // config.tensor_parallel_degree, bias_attr=False
                )
                self.lm_head.weight.is_distributed = True
                self.lm_head.weight.split_axis = 1
                self.lm_head_forward_func = lambda x: parallel_matmul(
                    x, self.lm_head.weight, config.tensor_parallel_output, transpose_y=False
                )
            else:
                self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias_attr=False)
                self.lm_head_forward_func = lambda x: self.lm_head(x)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        if self.config.tie_word_embeddings:
            logger.warning(
                "`set_output_embeddings` method is called when `config.tie_word_embeddings=True`. This is not expected. We will do nothing!"
            )
        else:
            self.lm_head = new_embeddings

    def get_output_embeddings(self):
        if self.config.tie_word_embeddings:
            return None
        else:
            return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        decoder_input_ids: Optional[paddle.Tensor] = None,
        decoder_attention_mask: Optional[paddle.Tensor] = None,
        encoder_output: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        decoder_inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[paddle.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from ppdiffusers.transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pd").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pd").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pd"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_output is None:
            # Convert encoder inputs in embeddings if needed
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_output, BaseModelOutput):
            encoder_output = BaseModelOutput(
                last_hidden_state=encoder_output[0],
                hidden_states=encoder_output[1] if len(encoder_output) > 1 else None,
                attentions=encoder_output[2] if len(encoder_output) > 2 else None,
            )

        hidden_states = encoder_output[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head_forward_func(sequence_output)

        loss = None
        if labels is not None:
            if self.config.tensor_parallel_degree > 1 and self.config.tensor_parallel_output:
                parallel_loss_func = fleet.meta_parallel.ParallelCrossEntropy()
                ignore_index = -100
                filtered_logits = lm_logits[labels != ignore_index]
                filtered_labels = labels[labels != ignore_index]
                loss = parallel_loss_func(filtered_logits, filtered_labels).mean()
            else:
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                # move labels to correct to enable PP
                loss = loss_fct(
                    lm_logits.reshape([-1, lm_logits.shape[-1]]),
                    labels.reshape(
                        [
                            -1,
                        ]
                    ),
                )
                # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_output
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_output.last_hidden_state,
            encoder_hidden_states=encoder_output.hidden_states,
            encoder_attentions=encoder_output.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        use_cache=None,
        encoder_output=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_output": encoder_output,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: paddle.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(axis=0, index=beam_idx),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


class T5EncoderModel(T5PretrainedModel):
    _tied_weights_keys = ["encoder.embed_tokens.weight"]
    _keys_to_ignore_on_load_unexpected = [r"decoder"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        if config.tensor_parallel_degree > 1:
            self.shared = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.d_model,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    # def _tie_weights(self):
    #     if self.config.tie_word_embeddings:
    #         self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)

    def get_encoder(self):
        return self.encoder

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[paddle.Tensor], BaseModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from ppdiffusers.transformers import AutoTokenizer, T5EncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-small")
        >>> model = T5EncoderModel.from_pretrained("t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_output


class T5ForSequenceClassification(T5PretrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.transformer = T5Model(config)
        self.classification_head = T5ClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        decoder_input_ids: Optional[paddle.Tensor] = None,
        decoder_attention_mask: Optional[paddle.Tensor] = None,
        encoder_output: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        decoder_inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        r"""
        labels (`paddle.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is not None:
            raise NotImplementedError(
                f"Passing input embeddings is currently not supported for {self.__class__.__name__}"
            )

        # Copied from models.bart.modeling_bart.BartModel.forward different to other models, T5 automatically creates
        # decoder_input_ids from input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )
            decoder_input_ids = self._shift_right(input_ids)

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_output=encoder_output,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        eos_mask = input_ids == self.config.eos_token_id

        if len(paddle.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        batch_size, _, hidden_size = sequence_output.shape
        sentence_representation = sequence_output[eos_mask, :].reshape([batch_size, -1, hidden_size])[:, -1, :]
        logits = self.classification_head(sentence_representation)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == paddle.int64 or labels.dtype == paddle.int32):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.reshape([-1, self.config.num_labels]),
                    labels.reshape(
                        [
                            -1,
                        ]
                    ),
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


class T5ForQuestionAnswering(T5PretrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        if config.tensor_parallel_degree > 1:
            self.shared = fleet.meta_parallel.VocabParallelEmbedding(
                config.vocab_size,
                config.d_model,
                weight_attr=paddle.ParamAttr(initializer=nn.initializer.XavierNormal()),
            )
        else:
            self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.num_labels = config.num_labels
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        decoder_input_ids: Optional[paddle.Tensor] = None,
        decoder_attention_mask: Optional[paddle.Tensor] = None,
        encoder_output: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        start_positions: Optional[paddle.Tensor] = None,
        end_positions: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        decoder_inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[paddle.Tensor], Seq2SeqQuestionAnsweringModelOutput]:
        r"""
        start_positions (`paddle.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`paddle.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (*sequence_length*). Position outside of the sequence
            are not taken into account for computing the loss.
        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if start_positions is not None and end_positions is not None:
            use_cache = False

        # Copied from models.bart.modeling_bart.BartModel.forward
        #   different to other models, T5 automatically creates decoder_input_ids from
        #   input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )
            decoder_input_ids = self._shift_right(input_ids)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_output is None:
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_output, BaseModelOutput):
            encoder_output = BaseModelOutput(
                last_hidden_state=encoder_output[0],
                hidden_states=encoder_output[1] if len(encoder_output) > 1 else None,
                attentions=encoder_output[2] if len(encoder_output) > 2 else None,
            )

        hidden_states = encoder_output[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=None,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.chunk(2, axis=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clip(0, ignored_index)
            end_positions = end_positions.clip(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + decoder_outputs[1:] + encoder_output
            return ((total_loss,) + output) if total_loss is not None else output

        return Seq2SeqQuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_output.last_hidden_state,
            encoder_hidden_states=encoder_output.hidden_states,
            encoder_attentions=encoder_output.attentions,
        )
