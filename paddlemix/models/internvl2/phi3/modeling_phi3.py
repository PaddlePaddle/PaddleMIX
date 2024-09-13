# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

""" Paddle Phi-3 model."""
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from paddlenlp.transformers.activations import ACT2FN
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from paddlenlp.transformers.model_utils import PretrainedModel

from .configuration_phi3 import Phi3Config
from ppdiffusers.utils import logging
logger = logging.get_logger(__name__)

from ..bert_padding import pad_input, unpad_input, index_first_axis
try:
    from paddle.nn.functional.flash_attention import flash_attention as flash_attn_func
    from paddle.nn.functional.flash_attention import flash_attn_unpadded as flash_attn_varlen_func
    print("modeling_phi3 has_flash_attn is True.")
    has_flash_attn = True
except:
    flash_attn_func, flash_attn_varlen_func = None, None
    print("modeling_phi3 has_flash_attn is False.")
    has_flash_attn = False

# Transformers scans dependencies in the modeling file, causing issues on conditional loading. The regex only ignores try/catch blocks, but not if statements
# if is_flash_attn_2_available():
_flash_supports_window_size = False
try:
    _flash_supports_window_size = 'window_size' in list(inspect.signature(flash_attn_func).parameters)
except ImportError as error:
    logger.warning(
        f'`flash-attention` package not found, consider installing for better performance: {error}.'
    )
    if not _flash_supports_window_size:
        logger.warning(
            "Current `flash-attenton` does not support `window_size`. Either upgrade or use `attn_implementation='eager'`."
        )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Phi3
class Phi3RMSNorm(nn.Layer):
    def __init__(self, hidden_size, epsilon=1e-6):
        """
        Phi3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        out_0 = paddle.create_parameter(
            shape=paddle.ones(shape=hidden_size).shape,
            dtype=paddle.ones(shape=hidden_size).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=hidden_size)),
        )
        out_0.stop_gradient = not True
        self.weight = out_0
        self.variance_epsilon = epsilon

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to("float32")
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype="int32")
    indices = paddle.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(paddle.cumsum(seqlens_in_batch, axis=0), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.gemma.modeling_gemma.GemmaRotaryEmbedding with gemma->phi3, Gemma->Phi3
class Phi3RotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer('inv_freq', None, persistable=False) #persistent=False)

    @paddle.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq = 1.0 / (
                self.base ** (paddle.arange(0, self.dim, 2, dtype=paddle.int64).astype('float32') / self.dim)
            )
        inv_freq_expanded = self.inv_freq[None, :, None].astype('float32').expand([position_ids.shape[0], -1, 1])
        position_ids_expanded = position_ids[:, None, :].astype('float32')
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        with paddle.amp.auto_cast(enable=False):
            freqs = (inv_freq_expanded.astype('float32') @ position_ids_expanded.astype('float32')).transpose([0, 2, 1]) #.transpose(1, 2)
            emb = paddle.concat((freqs, freqs), axis=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Phi3SuScaledRotaryEmbedding(Phi3RotaryEmbedding):
    def __init__(self, dim, config):
        super().__init__(dim, config.max_position_embeddings, config.rope_theta)

        self.short_factor = config.rope_scaling['short_factor']
        self.long_factor = config.rope_scaling['long_factor']
        self.original_max_position_embeddings = config.original_max_position_embeddings

    @paddle.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        seq_len = paddle.max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = paddle.to_tensor(self.long_factor, dtype=paddle.float32)
        else:
            ext_factors = paddle.to_tensor(self.short_factor, dtype=paddle.float32)

        inv_freq_shape = paddle.arange(0, self.dim, 2, dtype=paddle.int64).astype('float32') / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)

        inv_freq_expanded = self.inv_freq[None, :, None].astype('float32').expand([position_ids.shape[0], -1, 1])
        position_ids_expanded = position_ids[:, None, :].astype('float32')

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        with paddle.amp.auto_cast(enable=False):
            freqs = (inv_freq_expanded.astype('float32') @ position_ids_expanded.astype('float32')).transpose([0, 2, 1]) #.transpose(1, 2)
            emb = paddle.concat((freqs, freqs), axis=-1)

            scale = self.max_position_embeddings / self.original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

            cos = emb.cos() * scaling_factor
            sin = emb.sin() * scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Phi3YarnScaledRotaryEmbedding(Phi3RotaryEmbedding):
    def __init__(self, dim, config):
        super().__init__(dim, config.max_position_embeddings, config.rope_theta)

        self.short_factor = config.rope_scaling['short_factor']
        self.long_factor = config.rope_scaling['long_factor']
        self.original_max_position_embeddings = config.original_max_position_embeddings

    @paddle.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        seq_len = paddle.max(position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = paddle.to_tensor(self.long_factor, dtype=paddle.float32)
        else:
            ext_factors = paddle.to_tensor(self.short_factor, dtype=paddle.float32)

        inv_freq_shape = paddle.arange(0, self.dim, 2, dtype=paddle.int64).astype('float32') / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)

        inv_freq_expanded = self.inv_freq[None, :, None].astype('float32').expand([position_ids.shape[0], -1, 1])
        position_ids_expanded = position_ids[:, None, :].astype('float32')

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        with paddle.amp.auto_cast(enable=False):
            freqs = (inv_freq_expanded.astype('float32') @ position_ids_expanded.astype('float32')).transpose([0, 2, 1]) # transpose(1, 2)
            emb = paddle.concat((freqs, freqs), axis=-1)

            scale = self.max_position_embeddings / self.original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = 0.1 * math.log(scale) + 1.0

            cos = emb.cos() * scaling_factor
            sin = emb.sin() * scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat((-x2, x1), axis=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`paddle.Tensor`): The query tensor.
        k (`paddle.Tensor`): The key tensor.
        cos (`paddle.Tensor`): The cosine part of the rotary embedding.
        sin (`paddle.Tensor`): The sine part of the rotary embedding.
        position_ids (`paddle.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(paddle.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Phi3MLP(nn.Layer):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias_attr=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias_attr=False)

        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        up_states = self.gate_up_proj(hidden_states)

        gate, up_states = up_states.chunk(2, axis=-1)
        up_states = up_states * self.activation_fn(gate)

        return self.down_proj(up_states)


# Copied from transformers.models.llama.modeling_llama.repeat_kv with llama->phi
def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of paddle.repeat_interleave(x, axis=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand([batch, num_key_value_heads, n_rep, slen, head_dim])
    return hidden_states.reshape([batch, num_key_value_heads * n_rep, slen, head_dim])


class Phi3Attention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Phi3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f'Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will '
                'lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` '
                'when creating this class.'
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.rope_theta = config.rope_theta
        self.rope_scaling = config.rope_scaling
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}'
                f' and `num_heads`: {self.num_heads}).'
            )

        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias_attr=False)
        self.qkv_proj = nn.Linear(self.hidden_size, op_size, bias_attr=False)
        self._init_rope()

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rotary_emb = Phi3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling['type']
            if scaling_type == 'su':
                self.rotary_emb = Phi3SuScaledRotaryEmbedding(self.head_dim, self.config)
            elif scaling_type == 'yarn':
                self.rotary_emb = Phi3YarnScaledRotaryEmbedding(self.head_dim, self.config)
            else:
                raise ValueError(f'Unknown RoPE scaling type {scaling_type}')

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        logger.warning_once('You are not running the flash-attention implementation, expect numerical differences.')

        bsz, q_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3]) # transpose(1, 2)
        key_states = key_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                    'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                    'with a layer index.'
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) / math.sqrt(self.head_dim) # transpose(2, 3)

        if attn_weights.shape != [bsz, self.num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f'Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is'
                f' {attn_weights.shape}'
            )

        if attention_mask is not None:
            if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}'
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, axis=-1, dtype=paddle.float32).to(value_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = paddle.matmul(attn_weights, value_states)

        if attn_output.shape != [bsz, self.num_heads, q_len, self.head_dim]:
            raise ValueError(
                f'`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is'
                f' {attn_output.shape}'
            )

        attn_output = attn_output.transpose([0, 2, 1, 3]) # transpose(1, 2)
        attn_output = attn_output.reshape([bsz, q_len, self.hidden_size])

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Phi3FlashAttention2(Phi3Attention):
    """
    Phi-3 flash attention module. This module inherits from `Phi3Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        # self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self._flash_attn_uses_top_left_mask = True # TODO

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        # Phi3FlashAttention2 attention does not support output_attentions

        # if not _flash_supports_window_size:
        #     logger.warning_once(
        #         "The current flash attention version does not support sliding window attention. Please use `attn_implementation='eager'` or upgrade flash-attn library."
        #     )
        #     raise ValueError('The current flash attention version does not support sliding window attention.')

        output_attentions = False

        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`'
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop('padding_mask')

        bsz, q_len, _ = hidden_states.shape

        try:
            qkv = self.qkv_proj(hidden_states)
        except:
            qkv = self.qkv_proj(hidden_states.astype('bfloat16'))

        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3]) # transpose(1, 2)
        key_states = key_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                    'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                    'with a layer index.'
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, 'sliding_window', None) is not None
            and kv_seq_len > self.config.sliding_window
        )

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, 'sliding_window', None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :]
                past_value = past_value[:, :, slicing_tokens:, :]

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f'past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got'
                        f' {past_key.shape}'
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = paddle.concat([attention_mask, paddle.ones_like(attention_mask[:, -1:])], axis=-1)

            cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_dropout = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32.

        if query_states.dtype == paddle.float32:
            # if paddle.is_autocast_enabled():
            #     target_dtype = paddle.get_autocast_gpu_dtype()
            # # Handle the case where the model is quantized
            if hasattr(self.config, '_pre_quantization_dtype'):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.qkv_proj.weight.dtype

            logger.warning_once(
                f'The input hidden states seems to be silently casted in float32, this might be related to'
                f' the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in'
                f' {target_dtype}.'
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose([0, 2, 1, 3]) # transpose(1, 2)
        key_states = key_states.transpose([0, 2, 1, 3])
        value_states = value_states.transpose([0, 2, 1, 3])

        original_dtype = query_states.dtype
        query_states = query_states.astype('bfloat16')
        key_states = key_states.astype('bfloat16')
        value_states = value_states.astype('bfloat16')
        # return tuple 0 is output, 1 is softmax return
        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=attn_dropout,
            use_sliding_windows=use_sliding_windows,
        )
        attn_output = attn_output.astype(original_dtype)
        attn_output = attn_output.reshape([bsz, q_len, self.hidden_size])
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    # Copied from transformers.models.mistral.modeling_mistral.MistralFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`paddle.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`paddle.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`paddle.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`paddle.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        head_dim = query_states.shape[-1]
        softmax_scale = head_dim**-0.5 # TODO: 需要手动加上

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]

            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout=dropout,
                    scale=softmax_scale, # not softmax_scale=
                    causal=causal,
                )[0]
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout=dropout,
                    scale=softmax_scale, # not softmax_scale=
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )[0]

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states, # [1, 1879, 32, 96]
                    key_states, # [1, 1879, 32, 96]
                    value_states, # [1, 1879, 32, 96]
                    dropout,
                    #softmax_scale,
                    causal=causal,
                )[0] # return tuple, [0] is attn_output, [1] is None
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    #softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )[0]

        return attn_output

    # Copied from transformers.models.mistral.modeling_mistral.MistralFlashAttention2._upad_input
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape([batch_size * kv_seq_len, num_heads, head_dim]), indices_k)
        value_layer = index_first_axis(value_layer.reshape([batch_size * kv_seq_len, num_heads, head_dim]), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape([batch_size * kv_seq_len, num_heads, head_dim]), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = paddle.arange(
                batch_size + 1, dtype=paddle.int32,
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


# copied from transformers.models.llama.modeling_llama.LlamaSdpaAttention with Llama->Phi3
# TODO @Arthur no longer copied from LLama after static cache
class Phi3SdpaAttention(Phi3Attention):
    """
    Phi3 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Phi3Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Phi3Attention.forward
    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[paddle.Tensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                'Phi3Model is using Phi3SdpaAttention, but `paddle.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, '
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.shape

        try:
            qkv = self.qkv_proj(hidden_states)
        except:
            qkv = self.qkv_proj(hidden_states.astype('bfloat16'))
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3]) # transpose(1, 2)
        key_states = key_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {'sin': sin, 'cos': cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
                raise ValueError(
                    f'Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}'
                )

        # # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # # Reference: https://github.com/pytorch/pytorch/issues/112577.
        # if query_states.device.type == 'cuda' and attention_mask is not None:
        #     query_states = query_states.contiguous()
        #     key_states = key_states.contiguous()
        #     value_states = value_states.contiguous()

        attn_output = paddle.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        ) # use flash_attention in scaled_dot_product_attention

        attn_output = attn_output.transpose([0, 2, 1, 3]) # transpose(1, 2)
        attn_output = attn_output.reshape([bsz, q_len, self.hidden_size])

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


PHI3_ATTENTION_CLASSES = {
    'eager': Phi3Attention,
    'flash_attention_2': Phi3FlashAttention2,
    'sdpa': Phi3SdpaAttention,
}


class Phi3DecoderLayer(nn.Layer):
    def __init__(self, config: Phi3Config, layer_idx: int):
        super().__init__()

        self.config = config
        self.self_attn = PHI3_ATTENTION_CLASSES[config.attn_implementation](config, layer_idx=layer_idx)

        self.mlp = Phi3MLP(config)
        self.input_layernorm = Phi3RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = Phi3RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        if 'padding_mask' in kwargs:
            warnings.warn(
                'Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`'
            )
        """
        Args:
            hidden_states (`paddle.Tensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`paddle.Tensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """

        original_dtype = hidden_states.dtype
        residual = hidden_states.astype(original_dtype) # todo

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask, # None
            position_ids=position_ids,
            past_key_value=past_key_value, # None
            output_attentions=output_attentions, # False
            use_cache=use_cache, # True
        )
        # [1, 1879, 3072], None, None

        hidden_states = residual + self.resid_attn_dropout(attn_outputs)

        original_dtype = hidden_states.dtype
        residual = hidden_states.astype(original_dtype) # todo
        hidden_states = self.post_attention_layernorm(hidden_states)
        try:
            hidden_states = self.mlp(hidden_states)
        except:
            hidden_states = self.mlp(hidden_states.astype('bfloat16'))
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Phi3PreTrainedModel(PretrainedModel):
    config_class = Phi3Config
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['Phi3DecoderLayer']
    _skip_keys_device_placement = 'past_key_values'
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, layer):
        std = self.config.initializer_range
        if isinstance(layer, nn.Linear):
            paddle.nn.initializer.Normal(mean=0.0, std=std)(layer.weight)
            if layer.bias is not None:
                paddle.nn.initializer.Constant(0.0)(layer.bias)
        elif isinstance(layer, nn.Embedding):
            paddle.nn.initializer.Normal(mean=0.0, std=std)(layer.weight)
            if layer._padding_idx is not None:
                with paddle.no_grad():
                    layer.weight[layer._padding_idx] = 0.0


class Phi3Model(Phi3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Phi3DecoderLayer`]

    Args:
        config: Phi3Config
    """

    def __init__(self, config: Phi3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_dropout = nn.Dropout(config.embd_pdrop)
        self.layers = nn.LayerList(
            [Phi3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config.attn_implementation
        self.norm = Phi3RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        #self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')

        past_key_values_length = 0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
                )
                use_cache = False

        if use_cache: # TODO: check past_key_values is None
            # use_legacy_cache = not isinstance(past_key_values, Cache)
            # if use_legacy_cache:
            #     past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            # past_key_values_length = past_key_values.get_usable_length(seq_length)
            pass
            # TODO: fix this

        if position_ids is None:
            position_ids = paddle.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype='int64',
            )
            position_ids = position_ids.unsqueeze(0).reshape([-1, seq_length])
        else:
            # run this, must fix use_cache, TODO: fix this
            position_ids = position_ids[:, -1:]
            # seq_length = 1
            position_ids = position_ids.reshape([-1, 1]).cast('int64')

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == 'flash_attention_2' and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    ' this may lead to unexpected behaviour for Flash Attention version of Phi3. Make sure to '
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == 'flash_attention_2':
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask.numpy()) else None
        else:
            raise NotImplementedError
            # # 4d mask is passed through the layers # TODO
            # attention_mask = _prepare_4d_causal_attention_mask(
            #     attention_mask,
            #     (batch_size, seq_length),
            #     inputs_embeds,
            #     past_key_values_length,
            #     sliding_window=self.config.sliding_window,
            # )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # layer_outputs = self._gradient_checkpointing_func(
                #     decoder_layer.__call__,
                #     hidden_states,
                #     attention_mask,
                #     position_ids,
                #     past_key_values,
                #     output_attentions,
                #     use_cache,
                # )
                layer_outputs = paddle.distributed.fleet.utils.recompute(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            use_legacy_cache = False # TODO, next_decoder_cache is None
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Phi3ForCausalLM(Phi3PreTrainedModel):
    _tied_weights_keys = ['lm_head.weight']

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.__init__ with Llama->Phi3
    def __init__(self, config):
        super().__init__(config)
        self.model = Phi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)

        # Initialize weights and apply final processing
        #self.post_init()

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_input_embeddings
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_input_embeddings
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_output_embeddings
    def get_output_embeddings(self):
        return self.lm_head

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_output_embeddings
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.set_decoder
    def set_decoder(self, decoder):
        self.model = decoder

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.get_decoder
    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Phi3ForCausalLM

        >>> model = Phi3ForCausalLM.from_pretrained("microsoft/phi-3-mini-4k-instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")

        >>> prompt = "This is an example script ."
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'This is an example script .\n Certainly! Below is a sample script that demonstrates a simple task, such as calculating the sum'
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        try:
            logits = self.lm_head(hidden_states)
        except:
            logits = self.lm_head(hidden_states.cast('bfloat16'))
        logits = logits.cast('float32')

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.reshape([-1, self.config.vocab_size])
            shift_labels = shift_labels.reshape([-1])
            # Enable model parallelism
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Copied from transformers.models.persimmon.modeling_persimmon.PersimmonForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            if 0: #isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get('position_ids', None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.cast('int64').cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids}

        model_inputs.update(
            {
                'position_ids': position_ids,
                'past_key_values': past_key_values,
                'use_cache': kwargs.get('use_cache'),
                'attention_mask': attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM._reorder_cache
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past
