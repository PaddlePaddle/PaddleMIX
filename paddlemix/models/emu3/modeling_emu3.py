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
""" Paddle Emu3 model."""

import warnings
from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddlenlp.transformers.activations import ACT2FN
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from paddlenlp.transformers.model_utils import PretrainedModel

from ...utils.log import logger
from .configuration_emu3 import Emu3Config


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


def is_casual_mask(attention_mask):
    """
    Upper triangular of attention_mask equals to attention_mask is casual
    """
    return (paddle.triu(attention_mask) == attention_mask).all().item()


def _make_causal_mask(input_ids_shape, past_key_values_length):
    """
    Make causal mask used for self-attention
    """
    batch_size, target_length = input_ids_shape  # target_length: seq_len

    mask = paddle.tril(paddle.ones((target_length, target_length), dtype="bool"))

    if past_key_values_length > 0:
        # [tgt_len, tgt_len + past_len]
        mask = paddle.concat([paddle.ones([target_length, past_key_values_length], dtype="bool"), mask], axis=-1)

    # [bs, 1, tgt_len, tgt_len + past_len]
    return mask[None, None, :, :].expand([batch_size, 1, target_length, target_length + past_key_values_length])


def _expand_2d_mask(mask, dtype, tgt_length):
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape[0], mask.shape[-1]
    tgt_length = tgt_length if tgt_length is not None else src_length

    mask = mask[:, None, None, :].astype("bool")
    mask.stop_gradient = True
    expanded_mask = mask.expand([batch_size, 1, tgt_length, src_length])

    return expanded_mask


class Emu3RMSNorm(nn.Layer):
    def __init__(self, hidden_size, epsilon=1e-6):
        """
        Emu3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.ones(shape=hidden_size))
        self.variance_epsilon = epsilon

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(paddle.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Emu3RotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / self.base ** (
            paddle.arange(start=0, end=self.dim, step=2).astype(dtype="float32").to(device) / self.dim
        )
        self.register_buffer(name="inv_freq", tensor=inv_freq, persistable=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.place, dtype=paddle.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(dtype=self.inv_freq.dtype, end=self.max_seq_len_cached)
        freqs = paddle.outer(x=t, y=self.inv_freq)
        emb = paddle.concat(x=(freqs, freqs), axis=-1)
        self.register_buffer(name="cos_cached", tensor=emb.cos().to(dtype), persistable=False)
        self.register_buffer(name="sin_cached", tensor=emb.sin().to(dtype), persistable=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.place, dtype=x.dtype)
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)


class Emu3LinearScalingRotaryEmbedding(Emu3RotaryEmbedding):
    """Emu3RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(dtype=self.inv_freq.dtype, end=self.max_seq_len_cached)
        t = t / self.scaling_factor
        freqs = paddle.outer(x=t, y=self.inv_freq)
        emb = paddle.concat(x=(freqs, freqs), axis=-1)
        self.register_buffer(name="cos_cached", tensor=emb.cos().to(dtype), persistable=False)
        self.register_buffer(name="sin_cached", tensor=emb.sin().to(dtype), persistable=False)


class Emu3DynamicNTKScalingRotaryEmbedding(Emu3RotaryEmbedding):
    """Emu3RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                self.scaling_factor * seq_len / self.max_position_embeddings - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / base ** (
                paddle.arange(start=0, end=self.dim, step=2).astype(dtype="float32").to(device) / self.dim
            )
            self.register_buffer(name="inv_freq", tensor=inv_freq, persistable=False)
        t = paddle.arange(dtype=self.inv_freq.dtype, end=self.max_seq_len_cached)
        freqs = paddle.outer(x=t, y=self.inv_freq)
        emb = paddle.concat(x=(freqs, freqs), axis=-1)
        self.register_buffer(name="cos_cached", tensor=emb.cos().to(dtype), persistable=False)
        self.register_buffer(name="sin_cached", tensor=emb.sin().to(dtype), persistable=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat(x=(-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`paddle.Tensor`): The query tensor.
        k (`paddle.Tensor`): The key tensor.
        cos (`paddle.Tensor`): The cosine part of the rotary embedding.
        sin (`paddle.Tensor`): The sine part of the rotary embedding.
        position_ids (`paddle.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
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
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Emu3MLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias_attr=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias_attr=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        try:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        except:
            x = x.astype(self.up_proj.weight.dtype)
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand([batch, num_key_value_heads, n_rep, slen, head_dim])
    return hidden_states.reshape([batch, num_key_value_heads * n_rep, slen, head_dim])


class Emu3Attention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Emu3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger(
                log_level=30,
                msg=f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class.",
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias_attr=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias_attr=False)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = Emu3RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = Emu3LinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = Emu3DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: paddle.Tensor, seq_len: int, bsz: int):
        return tensor.reshape([bsz, seq_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[paddle.Tensor] = None,
        **kwargs,
    ):

        bsz, q_len, _ = hidden_states.shape  # [1, 4188, 4096]

        try:
            query_states = self.q_proj(hidden_states)  # 4096->1024
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        except:
            hidden_states = hidden_states.astype(self.q_proj.weight.dtype)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        key_states = key_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )

        kv_seq_len = key_states.shape[-2]  # [bs, num_head, seq_len, head_dim]
        if past_key_value is not None:
            kv_seq_len += int(cache_position[0]) + 1

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = paddle.concat([past_key_value[0], key_states], axis=2)
            value_states = paddle.concat([past_key_value[1], value_states], axis=2)
        past_key_value = (key_states, value_states) if use_cache else None

        # [1, 8, 4188, 128]
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            query_states.transpose([0, 2, 1, 3]),
            key_states.transpose([0, 2, 1, 3]),
            value_states.transpose([0, 2, 1, 3]),
            attn_mask=attention_mask,
            training=self.training,
        )
        # naive attention
        # attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) / math.sqrt(self.head_dim)

        # if tuple(attn_weights.shape) != (bsz, self.num_heads, q_len, kv_seq_len):
        #     raise ValueError(
        #         f'Attention weights should be of size {bsz, self.num_heads, q_len, kv_seq_len}, but is {tuple(attn_weights.shape)}'
        #     )
        # if attention_mask is not None:
        # attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        # attn_weights = F.softmax(attn_weights, axis=-1,dtype='float32').to(query_states.dtype)
        # attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # attn_output = paddle.matmul(attn_weights, value_states)

        # if tuple(attn_output.shape) != (bsz, self.num_heads, q_len, self.head_dim):
        #     raise ValueError(
        #         f'`attn_output` should be of size {bsz, self.num_heads, q_len, self.head_dim}, but is {tuple(attn_output.shape)}'
        #     )

        # attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([bsz, q_len, self.hidden_size])
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class Emu3FlashAttention2(Emu3Attention):
    """
    Emu3 flash attention module. This module inherits from `Emu3Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        output_attentions = False

        bsz, q_len, _ = tuple(hidden_states.shape)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        key_states = key_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose([0, 2, 1, 3])
        key_states = key_states.transpose([0, 2, 1, 3])  # transpose(1, 2)
        value_states = value_states.transpose([0, 2, 1, 3])  # transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (Emu3RMSNorm handles it correctly)

        # input_dtype = query_states.dtype
        # if input_dtype == paddle.float32:
        #     # Handle the case where the model is quantized
        #     if hasattr(self.config, "_pre_quantization_dtype"):
        #         target_dtype = self.config._pre_quantization_dtype
        #     else:
        #         target_dtype = self.q_proj.weight.dtype

        #     logger.warning_once(
        #         f"The input hidden states seems to be silently casted in float32, this might be related to"
        #         f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
        #         f" {target_dtype}."
        #     )

        #     query_states = query_states.to(target_dtype)
        #     key_states = key_states.to(target_dtype)
        #     value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape([bsz, q_len, self.hidden_size])
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
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
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in Emu3FlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            # TODO: Currently not support Flash Attention
            attn_output_unpad = F.flash_attn_unpadded(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout=dropout,
                scale=softmax_scale,
                causal=causal,
            )
            return attn_output_unpad
            # attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        # else:
        # attn_output = flash_attn_func(
        #     query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
        # )

        # return attn_output

    # def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
    #     indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    #     batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    #     key_layer = index_first_axis(
    #         key_layer.reshape([batch_size * kv_seq_len, num_key_value_heads, head_dim]), indices_k
    #     )
    #     value_layer = index_first_axis(
    #         value_layer.reshape([batch_size * kv_seq_len, num_key_value_heads, head_dim]), indices_k
    #     )
    #     if query_length == kv_seq_len:
    #         query_layer = index_first_axis(
    #             query_layer.reshape([batch_size * kv_seq_len, self.num_heads, head_dim]), indices_k
    #         )
    #         cu_seqlens_q = cu_seqlens_k
    #         max_seqlen_in_batch_q = max_seqlen_in_batch_k
    #         indices_q = indices_k
    #     elif query_length == 1:
    #         max_seqlen_in_batch_q = 1
    #         cu_seqlens_q = paddle.arange(
    #             batch_size + 1,
    #             dtype=paddle.int32,
    #         )  # There is a memcpy here, that is very bad.
    #         indices_q = cu_seqlens_q[:-1]
    #         query_layer = query_layer.squeeze(1)
    #     else:
    #         # The -q_len: slice assumes left padding.
    #         attention_mask = attention_mask[:, -query_length:]
    #         query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

    #     return (
    #         query_layer,
    #         key_layer,
    #         value_layer,
    #         indices_q,
    #         (cu_seqlens_q, cu_seqlens_k),
    #         (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    #     )


EMU3_ATTENTION_CLASSES = {
    "eager": Emu3Attention,
    # "flash_attention_2": Emu3FlashAttention2,
}


class Emu3DecoderLayer(nn.Layer):
    def __init__(self, config: Emu3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.attention_dropout)
        self.self_attn = Emu3Attention(
            config=config, layer_idx=layer_idx
        )  # EMU3_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = Emu3MLP(config)
        self.input_layernorm = Emu3RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)
        self.post_attention_layernorm = Emu3RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[paddle.Tensor] = None,
        **kwargs,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + self.dropout(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Emu3PreTrainedModel(PretrainedModel):
    config_class = Emu3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Emu3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    # _supports_flash_attn_2 = True
    # _supports_sdpa = True
    # _supports_cache_class = True

    # def _init_weights(self, module):
    #     std = self.config.initializer_range
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=std)
    #         if module.padding_idx is not None:
    #             module.weight.data[module.padding_idx].zero_()


class Emu3Model(Emu3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Emu3DecoderLayer`]

    Args:
        config: Emu3Config
    """

    def __init__(self, config: Emu3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.dropout = nn.Dropout(config.attention_dropout)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.LayerList(
            [Emu3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = False  # config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = False  # config._attn_implementation == "flash_attention_2"
        self.norm = Emu3RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @staticmethod
    def _prepare_decoder_attention_mask(attention_mask, input_shape, past_key_values_length, dtype):
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            if len(attention_mask.shape) == 2:
                expanded_attn_mask = _expand_2d_mask(attention_mask, dtype, tgt_length=input_shape[-1])
                # For decoding phase in generation, seq_length = 1, we don't need to add causal mask
                if input_shape[-1] > 1:
                    combined_attention_mask = _make_causal_mask(
                        input_shape,
                        past_key_values_length=past_key_values_length,
                    )
                    expanded_attn_mask = expanded_attn_mask & combined_attention_mask
            # [bsz, seq_len, seq_len] -> [bsz, 1, seq_len, seq_len]
            elif len(attention_mask.shape) == 3:
                expanded_attn_mask = attention_mask.unsqueeze(1).astype("bool")
            # if attention_mask is already 4-D, do nothing
            else:
                expanded_attn_mask = attention_mask
        else:
            expanded_attn_mask = _make_causal_mask(
                input_shape,
                past_key_values_length=past_key_values_length,
            )
        # Convert bool attention_mask to float attention mask, which will be added to attention_scores later
        expanded_attn_mask = paddle.where(expanded_attn_mask, 0.0, paddle.finfo(dtype).min).astype(dtype)
        return expanded_attn_mask

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
        cache_position: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        # NOTE: to make cache can be clear in-time
        past_key_values = list(past_key_values)

        seq_length_with_past = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = past_key_values[0][0].shape[2]
            seq_length_with_past += cache_length

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            # [bs, seq_len]
            attention_mask = paddle.ones((batch_size, seq_length_with_past), dtype=paddle.bool)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), cache_length, inputs_embeds.dtype
        )  # [bs, 1, seq_len, seq_len]

        if cache_position is None:
            past_seen_tokens = past_key_values[0][0].shape[2] if past_key_values[0] is not None else 0
            cache_position = paddle.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1])

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # embed positions
        hidden_states = self.dropout(inputs_embeds)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = ()

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            # NOTE: clear outdate cache after it has been used for memory saving
            past_key_value = past_key_values[idx] = None

            hidden_states = layer_outputs[0]

            # if use_cache:
            #     next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            next_decoder_cache = next_decoder_cache + (layer_outputs[-1],) if use_cache else None

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # next_cache = None
        # if use_cache:
        #     next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Emu3ForCausalLM(Emu3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Emu3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias_attr=False)

        # Initialize weights and apply final processing
        # self.post_init()

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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
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
            logits = self.lm_head(hidden_states.astype(self.lm_head.weight.dtype))
        logits = logits.astype(dtype="float32")

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            # loss_fct = CrossEntropyLoss()
            loss_fct = nn.CrossEntropyLoss(reduction="sum")
            shift_logits = shift_logits.reshape([-1, self.config.vocab_size])
            shift_labels = shift_labels.reshape([-1])
            # Enable model parallelism
            loss = loss_fct(shift_logits, shift_labels)
            label_sum = paddle.sum(shift_labels != -100).cast("float32")
            loss = loss / label_sum

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

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        batch_size, seq_length = input_ids.shape
        attention_mask = paddle.ones((batch_size, seq_length), dtype=paddle.bool)

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

        # {'use_cache': True}
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.astype(dtype="int64").cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
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

    def get_logits_processor(
        self,
        min_length=None,
        max_length=None,
        eos_token_id=None,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        num_beams=1,
        num_beam_groups=1,
        diversity_rate=0.0,
        repetition_penalty=None,
        no_repeat_ngram_size=None,
        logits_processors=None,
    ):
        return logits_processors
