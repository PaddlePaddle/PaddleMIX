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

import math
import warnings
from typing import List, Optional, Tuple, Union

import bert_padding
import paddle
import paddle.autograd
import paddle.nn.functional as F

# import paddle_aux
import paddlenlp
from configuration import Phi3Config
from paddlenlp.transformers.activations import ACT2FN
from paddlenlp.transformers.model_outputs import (
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from paddlenlp.transformers.model_utils import PretrainedModel

from ppdiffusers.utils import logging

logger = logging.get_logger(__name__)

# Transformers scans dependencies in the modeling file, causing issues on conditional loading. The regex only ignores try/catch blocks, but not if statements
# if is_flash_attn_2_available():
_flash_supports_window_size = True  # TODO  需要检查该实现

flash_attn_func, flash_attn_varlen_func = None, None
pad_input, index_first_axis, unpad_input = None, None, None

try:
    from paddle.nn.functional.flash_attention import flash_attention as _flash_attn_func
    from paddle.nn.functional.flash_attention import (
        flash_attn_unpadded as _flash_attn_varlen_func,
    )
    from utils import index_first_axis as _index_first_axis
    from utils import pad_input as _pad_input
    from utils import unpad_input as _unpad_input

    flash_attn_func, flash_attn_varlen_func = _flash_attn_func, _flash_attn_varlen_func
    pad_input, index_first_axis, unpad_input = _pad_input, _index_first_axis, _unpad_input
    has_flash_attn = True
except:
    has_flash_attn = False

PHI3_PRETRAINED_MODEL_ARCHIVE_LIST = ["microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3-mini-128k-instruct"]


class Phi3RMSNorm(paddle.nn.Layer):
    def __init__(self, hidden_size, eps=1e-06):
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
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to("float32")
        variance = hidden_states.pow(y=2).mean(axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(x=variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype="int32")
    indices = paddle.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(paddle.cumsum(seqlens_in_batch, axis=0, dtype=paddle.torch.int32), (1, 0))
    # cu_seqlens = paddle_aux._FUNCTIONAL_PAD(pad=(1, 0), x=paddle.cumsum(x=
    #     seqlens_in_batch, axis=0, dtype='int32'))
    return indices, cu_seqlens, max_seqlen_in_batch


class Phi3RotaryEmbedding(paddle.nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer(name="inv_freq", tensor=None, persistable=False)

    @paddle.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        if self.inv_freq is None:
            self.inv_freq = 1.0 / self.base ** (
                paddle.arange(start=0, end=self.dim, step=2, dtype="int64").astype(dtype="float32") / self.dim
            )
        inv_freq_expanded = (
            self.inv_freq[None, :, None].astype(dtype="float32").expand(shape=[tuple(position_ids.shape)[0], -1, 1])
        )
        position_ids_expanded = position_ids[:, None, :].astype(dtype="float32")
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with paddle.amp.auto_cast(enable=False):
            x = inv_freq_expanded.astype(dtype="float32") @ position_ids_expanded.astype(dtype="float32")
            perm_0 = list(range(x.ndim))
            perm_0[1] = 2
            perm_0[2] = 1
            freqs = x.transpose(perm=perm_0)
            emb = paddle.concat(x=(freqs, freqs), axis=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Phi3SuScaledRotaryEmbedding(Phi3RotaryEmbedding):
    def __init__(self, dim, config, device=None):
        super().__init__(dim, config.max_position_embeddings, config.rope_theta, device)
        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings

    @paddle.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        seq_len = paddle.max(x=position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = paddle.to_tensor(data=self.long_factor, dtype="float32", place=x.place)
        else:
            ext_factors = paddle.to_tensor(data=self.short_factor, dtype="float32", place=x.place)
        inv_freq_shape = paddle.arange(start=0, end=self.dim, step=2, dtype="int64").astype(dtype="float32") / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].astype(dtype="float32").expand(shape=[tuple(position_ids.shape)[0], -1, 1])
        )
        position_ids_expanded = position_ids[:, None, :].astype(dtype="float32")
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with paddle.amp.auto_cast(enable=False):
            x = inv_freq_expanded.astype(dtype="float32") @ position_ids_expanded.astype(dtype="float32")
            perm_1 = list(range(x.ndim))
            perm_1[1] = 2
            perm_1[2] = 1
            freqs = x.transpose(perm=perm_1)
            emb = paddle.concat(x=(freqs, freqs), axis=-1)
            scale = self.max_position_embeddings / self.original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))
            cos = emb.cos() * scaling_factor
            sin = emb.sin() * scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Phi3YarnScaledRotaryEmbedding(Phi3RotaryEmbedding):
    def __init__(self, dim, config, device=None):
        super().__init__(dim, config.max_position_embeddings, config.rope_theta, device)
        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.original_max_position_embeddings = config.original_max_position_embeddings

    @paddle.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        seq_len = paddle.max(x=position_ids) + 1
        if seq_len > self.original_max_position_embeddings:
            ext_factors = paddle.to_tensor(data=self.long_factor, dtype="float32", place=x.place)
        else:
            ext_factors = paddle.to_tensor(data=self.short_factor, dtype="float32", place=x.place)
        inv_freq_shape = paddle.arange(start=0, end=self.dim, step=2, dtype="int64").astype(dtype="float32") / self.dim
        self.inv_freq = 1.0 / (ext_factors * self.base**inv_freq_shape)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].astype(dtype="float32").expand(shape=[tuple(position_ids.shape)[0], -1, 1])
        )
        position_ids_expanded = position_ids[:, None, :].astype(dtype="float32")
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with paddle.amp.auto_cast(enable=False):
            x = inv_freq_expanded.astype(dtype="float32") @ position_ids_expanded.astype(dtype="float32")
            perm_2 = list(range(x.ndim))
            perm_2[1] = 2
            perm_2[2] = 1
            freqs = x.transpose(perm=perm_2)
            emb = paddle.concat(x=(freqs, freqs), axis=-1)
            scale = self.max_position_embeddings / self.original_max_position_embeddings
            if scale <= 1.0:
                scaling_factor = 1.0
            else:
                scaling_factor = 0.1 * math.log(scale) + 1.0
            cos = emb.cos() * scaling_factor
            sin = emb.sin() * scaling_factor
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : tuple(x.shape)[-1] // 2]
    x2 = x[..., tuple(x.shape)[-1] // 2 :]
    return paddle.concat(x=(-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(axis=unsqueeze_dim)
    sin = sin.unsqueeze(axis=unsqueeze_dim)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class Phi3MLP(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate_up_proj = paddle.nn.Linear(
            in_features=config.hidden_size, out_features=2 * config.intermediate_size, bias_attr=False
        )
        self.down_proj = paddle.nn.Linear(
            in_features=config.intermediate_size, out_features=config.hidden_size, bias_attr=False
        )
        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: paddle.Tensor) -> paddle.float32:
        up_states = self.gate_up_proj(hidden_states)
        gate, up_states = up_states.chunk(chunks=2, axis=-1)
        up_states = up_states * self.activation_fn(gate)
        return self.down_proj(up_states)


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = tuple(hidden_states.shape)
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(shape=[batch, num_key_value_heads, n_rep, slen, head_dim])
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Phi3Attention(paddle.nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Phi3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` when creating this class."
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
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads})."
            )
        op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        self.o_proj = paddle.nn.Linear(
            in_features=self.num_heads * self.head_dim, out_features=self.hidden_size, bias_attr=False
        )
        self.qkv_proj = paddle.nn.Linear(in_features=self.hidden_size, out_features=op_size, bias_attr=False)
        self._init_rope()

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rotary_emb = Phi3RotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            if scaling_type == "su":
                self.rotary_emb = Phi3SuScaledRotaryEmbedding(self.head_dim, self.config)
            elif scaling_type == "yarn":
                self.rotary_emb = Phi3YarnScaledRotaryEmbedding(self.head_dim, self.config)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        logger.warning_once("You are not running the flash-attention implementation, expect numerical differences.")
        bsz, q_len, _ = tuple(hidden_states.shape)
        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]
        x = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        perm_3 = list(range(x.ndim))
        perm_3[1] = 2
        perm_3[2] = 1
        query_states = x.transpose(perm=perm_3)
        x = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        perm_4 = list(range(x.ndim))
        perm_4[1] = 2
        perm_4[2] = 1
        key_states = x.transpose(perm=perm_4)
        x = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        perm_5 = list(range(x.ndim))
        perm_5[1] = 2
        perm_5[2] = 1
        value_states = x.transpose(perm=perm_5)
        kv_seq_len = tuple(key_states.shape)[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} for auto-regressive decoding with k/v caching, please make sure to initialize the attention class with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        x = key_states
        perm_6 = list(range(x.ndim))
        perm_6[2] = 3
        perm_6[3] = 2
        attn_weights = paddle.matmul(x=query_states, y=x.transpose(perm=perm_6)) / math.sqrt(self.head_dim)
        if tuple(attn_weights.shape) != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {bsz, self.num_heads, q_len, kv_seq_len}, but is {tuple(attn_weights.shape)}"
            )
        if attention_mask is not None:
            if tuple(attention_mask.shape) != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {bsz, 1, q_len, kv_seq_len}, but is {tuple(attention_mask.shape)}"
                )
            attn_weights = attn_weights + attention_mask
        attn_weights = paddle.nn.functional.softmax(x=attn_weights, axis=-1, dtype="float32").to(value_states.dtype)
        attn_weights = paddle.nn.functional.dropout(x=attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = paddle.matmul(x=attn_weights, y=value_states)
        if tuple(attn_output.shape) != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {bsz, self.num_heads, q_len, self.head_dim}, but is {tuple(attn_output.shape)}"
            )
        x = attn_output
        perm_7 = list(range(x.ndim))
        perm_7[1] = 2
        perm_7[2] = 1
        attn_output = x.transpose(perm=perm_7)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flash_attn_uses_top_left_mask = True
        # self._flash_attn_uses_top_left_mask = (not transformers.utils.
        #     is_flash_attn_greater_or_equal_2_10())  # TODO 需要检查

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention. Please use `attn_implementation='eager'` or upgrade flash-attn library."
            )
            raise ValueError("The current flash attention version does not support sliding window attention.")
        output_attentions = False
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
            attention_mask = kwargs.pop("padding_mask")
        bsz, q_len, _ = tuple(hidden_states.shape)
        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]
        x = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        perm_8 = list(range(x.ndim))
        perm_8[1] = 2
        perm_8[2] = 1
        query_states = x.transpose(perm=perm_8)
        x = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        perm_9 = list(range(x.ndim))
        perm_9[1] = 2
        perm_9[2] = 1
        key_states = x.transpose(perm=perm_9)
        x = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        perm_10 = list(range(x.ndim))
        perm_10[1] = 2
        perm_10[2] = 1
        value_states = x.transpose(perm=perm_10)
        kv_seq_len = tuple(key_states.shape)[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} for auto-regressive decoding with k/v caching, please make sure to initialize the attention class with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=rotary_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
        )
        if past_key_value is not None:
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window
                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]
                past_key = past_key[:, :, slicing_tokens:, :]
                past_value = past_value[:, :, slicing_tokens:, :]
                if tuple(past_key.shape)[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got {tuple(past_key.shape)}"
                    )
                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = paddle.concat(
                        x=[attention_mask, paddle.ones_like(x=attention_mask[:, -1:])], axis=-1
                    )
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_dropout = self.attention_dropout if self.training else 0.0
        if query_states.dtype == "float32":
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.qkv_proj.weight.dtype
            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        x = query_states
        perm_11 = list(range(x.ndim))
        perm_11[1] = 2
        perm_11[2] = 1
        query_states = x.transpose(perm=perm_11)
        x = key_states
        perm_12 = list(range(x.ndim))
        perm_12[1] = 2
        perm_12[2] = 1
        key_states = x.transpose(perm=perm_12)
        x = value_states
        perm_13 = list(range(x.ndim))
        perm_13[1] = 2
        perm_13[2] = 1
        value_states = x.transpose(perm=perm_13)
        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=attn_dropout,
            use_sliding_windows=use_sliding_windows,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

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
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
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
            causal = self.is_causal and query_length != 1
        if attention_mask is not None:
            batch_size = tuple(query_states.shape)[0]
            (query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens) = self._upad_input(
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
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )
            attn_output = bert_padding.pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        elif not use_sliding_windows:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=(self.config.sliding_window, self.config.sliding_window),
            )
        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = tuple(key_layer.shape)
        if kv_seq_len != tuple(attention_mask.shape)[-1]:
            attention_mask_num_tokens = tuple(attention_mask.shape)[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        key_layer = bert_padding.index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
        )
        value_layer = bert_padding.index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = bert_padding.index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = paddle.arange(dtype="int32", end=batch_size + 1)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(axis=1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = bert_padding.unpad_input(
                query_layer, attention_mask
            )
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class Phi3SdpaAttention(Phi3Attention):
    """
    Phi3 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Phi3Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if output_attentions:
            logger.warning_once(
                'Phi3Model is using Phi3SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        bsz, q_len, _ = tuple(hidden_states.shape)
        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]
        x = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        perm_14 = list(range(x.ndim))
        perm_14[1] = 2
        perm_14[2] = 1
        query_states = x.transpose(perm=perm_14)
        x = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        perm_15 = list(range(x.ndim))
        perm_15[1] = 2
        perm_15[2] = 1
        key_states = x.transpose(perm=perm_15)
        x = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        perm_16 = list(range(x.ndim))
        perm_16[1] = 2
        perm_16[2] = 1
        value_states = x.transpose(perm=perm_16)
        kv_seq_len = tuple(key_states.shape)[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        if attention_mask is not None:
            if tuple(attention_mask.shape) != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {bsz, 1, q_len, kv_seq_len}, but is {tuple(attention_mask.shape)}"
                )
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states
            key_states = key_states
            value_states = value_states
        attn_output = paddle.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )
        x = attn_output
        perm_17 = list(range(x.ndim))
        perm_17[1] = 2
        perm_17[2] = 1
        attn_output = x.transpose(perm=perm_17)
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_value


PHI3_ATTENTION_CLASSES = {"eager": Phi3Attention, "flash_attention_2": Phi3FlashAttention2, "sdpa": Phi3SdpaAttention}


class Phi3DecoderLayer(paddle.nn.Layer):
    def __init__(self, config: Phi3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.self_attn = PHI3_ATTENTION_CLASSES[config.attn_implementation](config, layer_idx=layer_idx)
        self.mlp = Phi3MLP(config)
        self.input_layernorm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.resid_attn_dropout = paddle.nn.Dropout(p=config.resid_pdrop)
        self.resid_mlp_dropout = paddle.nn.Dropout(p=config.resid_pdrop)
        self.post_attention_layernorm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + self.resid_attn_dropout(attn_outputs)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class Phi3PreTrainedModel(PretrainedModel):
    config_class = Phi3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Phi3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True
    _version = "0.0.5"

    def __init__(self, config: Phi3Config):
        if not has_flash_attn:
            config._attn_implementation = "eager"
            print("Warning: Flash attention is not available, using eager attention instead.")
        super().__init__(config)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, paddle.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, paddle.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


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
        self.embed_tokens = paddle.nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, padding_idx=self.padding_idx
        )
        self.embed_dropout = paddle.nn.Dropout(p=config.embd_pdrop)
        self.layers = paddle.nn.LayerList(
            sublayers=[Phi3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

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
    ) -> Union[Tuple, paddlenlp.transformers.model_outputs.BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = tuple(input_ids.shape)[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = tuple(inputs_embeds.shape)[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        past_key_values_length = 0
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        if position_ids is None:
            _ = input_ids.place if input_ids is not None else inputs_embeds.place
            position_ids = paddle.arange(
                start=past_key_values_length, end=seq_length + past_key_values_length, dtype="int64"
            )
            position_ids = position_ids.unsqueeze(axis=0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).astype(dtype="int64")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right' this may lead to unexpected behaviour for Flash Attention version of Phi3. Make sure to  call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )
        attention_mask = attention_mask if attention_mask is not None and 0 in attention_mask else None

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
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
                _ = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return paddlenlp.transformers.model_outputs.BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Phi3ForCausalLM(Phi3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Phi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = paddle.nn.Linear(
            in_features=config.hidden_size, out_features=config.vocab_size, bias_attr=False
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

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
    ) -> Union[Tuple, paddlenlp.transformers.model_outputs.CausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
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
        'This is an example script .\\n Certainly! Below is a sample script that demonstrates a simple task, such as calculating the sum'
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
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
        logits = self.lm_head(hidden_states)
        logits = logits.astype(dtype="float32")
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss_fct = paddle.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.place)
            loss = loss_fct(shift_logits, shift_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return paddlenlp.transformers.model_outputs.CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            cache_length = past_length = tuple(past_key_values[0][0].shape)[2]
            max_cache_length = None
            if attention_mask is not None and tuple(attention_mask.shape)[1] > tuple(input_ids.shape)[1]:
                input_ids = input_ids[:, -(tuple(attention_mask.shape)[1] - past_length) :]
            elif past_length < tuple(input_ids.shape)[1]:
                input_ids = input_ids[:, past_length:]
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + tuple(input_ids.shape)[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.astype(dtype="int64").cumsum(axis=-1) - 1
            position_ids.masked_fill_(mask=attention_mask == 0, value=1)
            if past_key_values:
                position_ids = position_ids[:, -tuple(input_ids.shape)[1] :]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(axis=0, index=beam_idx.to(past_state.place)) for past_state in layer_past
                ),
            )
        return reordered_past


class Phi3ForSequenceClassification(Phi3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Phi3Model(config)
        self.score = paddle.nn.Linear(in_features=config.hidden_size, out_features=self.num_labels, bias_attr=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

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
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        model_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = model_outputs[0]
        logits = self.score(hidden_states)
        if input_ids is not None:
            batch_size = tuple(input_ids.shape)[0]
        else:
            batch_size = tuple(inputs_embeds.shape)[0]
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        elif input_ids is not None:
            sequence_lengths = (
                paddle.equal(x=input_ids, y=self.config.pad_token_id).astype(dtype="int32").argmax(axis=-1) - 1
            )
            sequence_lengths = sequence_lengths % tuple(input_ids.shape)[-1]
            sequence_lengths = sequence_lengths.to(logits.place)
        else:
            sequence_lengths = -1
        pooled_logits = logits[paddle.arange(end=batch_size), sequence_lengths]
        loss = None
        if labels is not None:
            labels = labels.to(logits.place)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == "int64" or labels.dtype == "int32"):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = paddle.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = paddle.nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = paddle.nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + model_outputs[1:]
            return (loss,) + output if loss is not None else output
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )


class Phi3ForTokenClassification(Phi3PreTrainedModel):
    def __init__(self, config: Phi3Config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Phi3Model(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = paddle.nn.Dropout(p=classifier_dropout)
        self.classifier = paddle.nn.Linear(in_features=config.hidden_size, out_features=config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor, paddle.Tensor], ...]] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments
    ) -> Union[Tuple[paddle.Tensor], TokenClassifierOutput]:
        """
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        model_outputs = self.model(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = model_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        loss = None
        if labels is not None:
            labels = labels.to(logits.place)
            batch_size, seq_length = tuple(labels.shape)
            loss_fct = paddle.nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(batch_size * seq_length, self.num_labels), labels.view(batch_size * seq_length)
            )
        if not return_dict:
            output = (logits,) + model_outputs[2:]
            return (loss,) + output if loss is not None else output
        return paddle.transformers.model_outputs.TokenClassifierOutput(
            loss=loss, logits=logits, hidden_states=model_outputs.hidden_states, attentions=model_outputs.attentions
        )


if __name__ == "__main__":
    model = Phi3DecoderLayer(Phi3Config(), 32)
    model = model.astype("float16")
    dummy_tensor = paddle.randn([1, 1024, 3072]).cuda()
    dummy_tensor = dummy_tensor.astype("float16")
    out = model(dummy_tensor)
