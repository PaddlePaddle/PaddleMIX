# -*- coding: utf-8 -*-

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# @Time    : 2025/4/19 下午8:37
# @Author  : zhaop-l(zhaopuzxjc@126.com)

from functools import partial
from typing import Optional, Tuple, Union

import paddle
from packaging import version
from paddle import Tensor, nn
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from paddlenlp.transformers.model_utils import PretrainedModel

from paddlemix.models.flash_attn_utils import has_flash_attn_func
from ppdiffusers.utils import logging
from ppdiffusers.utils.paddle_utils import dim2perm

from ...activations import ACT2FN
from .configuration_llama import CustomLlamaConfig

try:
    from apex.megatron_layer_norm import MixedFusedLayerNorm as LayerNorm
except ImportError:
    from paddle.nn import LayerNorm

USE_FLASH_ATTN = False
try:
    import flash_attn

    if version.parse(flash_attn.__version__) >= version.parse("2.1.0"):
        USE_FLASH_ATTN = True
        flash_attn_func, flash_attn_varlen_func = has_flash_attn_func()
        from paddlemix.models.qwen2_vl.bert_padding import (
            index_first_axis,
            pad_input,
            unpad_input,
        )
except ImportError:
    pass
logger = logging.get_logger(__name__)


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype="int32")
    indices = paddle.nonzero(x=attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max_func().item()
    cu_seqlens = nn.functional.pad(
        x=paddle.cumsum(x=seqlens_in_batch, axis=0, dtype="int32"), pad=(1, 0), pad_from_left_axis=False
    )
    return indices, cu_seqlens, max_seqlen_in_batch


class RMSNorm(nn.Layer):
    def __init__(self, dim: int, eps: float = 1e-06):
        """
        Initialize the RMSNorm normalization layer.

        Args:
                dim (int): The dimension of the input tensor.
                eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
                eps (float): A small value added to the denominator for numerical stability.
                weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = paddle.create_parameter(
            shape=[dim],
            default_initializer=nn.initializer.Constant(1.0),
            dtype=paddle.get_default_dtype(),
        )

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
                x (torch.Tensor): The input tensor.

        Returns:
                torch.Tensor: The normalized tensor.

        """
        return x * paddle.rsqrt(x=x.pow(y=2).mean(axis=-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
                x (torch.Tensor): The input tensor.

        Returns:
                torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.astype(dtype="float32")).astype(dtype=x.dtype)
        return output * self.weight


def get_norm(config: CustomLlamaConfig):
    norm_type = config.norm_type
    if norm_type == "rms_norm":
        return partial(RMSNorm, eps=config.layernorm_epsilon)
    elif norm_type == "layer_norm":
        return partial(LayerNorm, eps=config.layernorm_epsilon)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")


def _make_causal_mask(input_ids_shape: list, dtype: paddle.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = paddle.full(shape=(tgt_len, tgt_len), fill_value=paddle.finfo(dtype=dtype).min)
    mask_cond = paddle.arange(end=mask.shape[-1])
    mask.masked_fill_(mask=mask_cond < (mask_cond + 1).reshape((mask.shape[-1], 1)), value=0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = paddle.concat(x=[paddle.zeros(shape=[tgt_len, past_key_values_length], dtype=dtype), mask], axis=-1)
    return mask[None, None, :, :].expand(shape=[bsz, 1, tgt_len, tgt_len + past_key_values_length])


def _expand_mask(mask: Tensor, dtype: paddle.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = tuple(mask.shape)
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(shape=[bsz, 1, tgt_len, src_len]).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(mask=inverted_mask.to("bool"), value=paddle.finfo(dtype=dtype).min)


class RotaryEmbedding(nn.Layer):
    def __init__(self, dim, base=10000, compress=1.0):
        super().__init__()
        self.inv_freq = 1.0 / base ** (paddle.arange(start=0, end=dim, step=2).astype(dtype="float32") / dim)
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None
        self.compress = compress

    def forward(self, x, seq_len):
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            self.inv_freq = self.inv_freq
            t = paddle.arange(dtype=self.inv_freq.dtype, end=seq_len) * self.compress
            freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
            emb = paddle.concat(x=(freqs, freqs), axis=-1)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


def rotate_half(x):
    x1, x2 = x[..., : tuple(x.shape)[-1] // 2], x[..., tuple(x.shape)[-1] // 2 :]
    return paddle.concat(x=(-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos, sin = (cos[..., offset : tuple(q.shape)[-2] + offset, :], sin[..., offset : tuple(q.shape)[-2] + offset, :])
    q_embed = q.astype(dtype="float32") * cos + rotate_half(q.astype(dtype="float32")) * sin
    k_embed = k.astype(dtype="float32") * cos + rotate_half(k.astype(dtype="float32")) * sin
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def apply_rotary_pos_emb_torch(q, k, cos, sin, offset: int = 0):
    cos, sin = (cos[..., offset : tuple(q.shape)[-2] + offset, :], sin[..., offset : tuple(q.shape)[-2] + offset, :])
    q_embed = q.astype(dtype="float32") * cos + rotate_half(q.astype(dtype="float32")) * sin
    k_embed = k.astype(dtype="float32") * cos + rotate_half(k.astype(dtype="float32")) * sin
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class CustomLlamaAttention(nn.Layer):
    def __init__(self, config: CustomLlamaConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        self.max_positions = config.max_position_embeddings
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims, base=config.rotary_emb_base, compress=config.rotary_compress
        )
        self.norm_factor = paddle.sqrt(x=paddle.to_tensor(data=self.head_size, dtype="float32")).to(
            paddle.get_default_dtype()
        )

        if self.use_gqa:
            self.query_dense = nn.Linear(
                in_features=config.hidden_size,
                out_features=config.hidden_size,
                bias_attr=getattr(config, "qkv_proj_bias", True),
            )
            self.key_value_dense = nn.Linear(
                in_features=config.hidden_size,
                out_features=self.head_size * 2 * config.num_kv_heads,
                bias_attr=getattr(config, "qkv_proj_bias", True),
            )
        else:
            self.query_key_value = nn.Linear(
                in_features=config.hidden_size,
                out_features=3 * config.hidden_size,
                bias_attr=getattr(config, "qkv_proj_bias", True),
            )
        self.dense = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.hidden_size,
            bias_attr=getattr(config, "out_proj_bias", True),
        )
        self.apply_rotary_fn = apply_rotary_pos_emb_torch if config.dtype == "bfloat16" else apply_rotary_pos_emb

    @property
    def use_gqa(self):
        return self.num_kv_heads < self.num_attention_heads

    def forward(
        self, hidden_states, attention_mask, head_mask=None, layer_past=None, use_cache=False, output_attentions=False
    ):
        has_layer_past = layer_past is not None

        if self.use_gqa:
            q = self.query_dense(hidden_states)
            new_q_shape = tuple(q.shape)[:-1] + (self.num_attention_heads, self.head_size)

            q = q.reshape(new_q_shape)
            kv = self.key_value_dense(hidden_states)

            new_kv_shape = tuple(kv.shape)[:-1] + (self.num_kv_heads, 2 * self.head_size)
            kv = kv.reshape(new_kv_shape)
            query = q.transpose(perm=[0, 2, 1, 3])
            key = kv[..., : self.head_size].transpose(perm=[0, 2, 1, 3])
            value = kv[..., self.head_size :].transpose(perm=[0, 2, 1, 3])
        else:
            qkv = self.query_key_value(hidden_states)
            new_qkv_shape = tuple(qkv.shape)[:-1] + (self.num_attention_heads, 3 * self.head_size)
            qkv = qkv.reshape(new_qkv_shape)
            query = qkv[..., : self.head_size].transpose(perm=[0, 2, 1, 3])
            key = qkv[..., self.head_size : 2 * self.head_size].transpose(perm=[0, 2, 1, 3])
            value = qkv[..., 2 * self.head_size :].transpose(perm=[0, 2, 1, 3])

        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        seq_len = tuple(key.shape)[-2]
        offset = 0
        if has_layer_past:
            offset = tuple(layer_past[0].shape)[-2]
            seq_len += offset
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = self.apply_rotary_fn(query_rot, key_rot, cos, sin, offset=offset)
        query = paddle.concat(x=(query, query_pass), axis=-1)
        key = paddle.concat(x=(key, key_pass), axis=-1)
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = paddle.concat(x=(past_key, key), axis=-2)
            value = paddle.concat(x=(past_value, value), axis=-2)
        present = (key, value) if use_cache else None
        if USE_FLASH_ATTN:
            attn_output, attn_weights = self._flash_attn(query, key, value, attention_mask, head_mask)
            attn_output = attn_output.reshape(
                (attn_output.shape[0], attn_output.shape[1], self.hidden_size)
            ).contiguous()
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
            attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_size)
        attn_output = self.dense(attn_output)
        outputs = attn_output, present
        if output_attentions:
            outputs += (attn_weights,)
        return outputs

    @classmethod
    def _split_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        new_shape = tuple(tensor.shape)[:-1] + (num_attention_heads, attn_head_size)
        tensor = tensor.reshape(new_shape)
        tensor = tensor.transpose(perm=[0, 2, 1, 3])
        return tensor

    @classmethod
    def _merge_heads(cls, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        tensor = tensor.transpose(perm=[0, 2, 1, 3]).contiguous()
        tensor = tensor.reshape((tensor.shape[0], tensor.shape[1], num_attention_heads * attn_head_size))
        return tensor

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        batch_size, num_attention_heads, query_length, attn_head_size = tuple(query.shape)
        _, num_attention_groups, key_length, _ = tuple(key.shape)

        group_size = num_attention_heads // num_attention_groups
        if not self.use_gqa:
            assert group_size == 1

        key = (
            key.reshape((batch_size, num_attention_groups, 1, key_length, attn_head_size))
            .tile(repeat_times=[1, 1, group_size, 1, 1])
            .reshape((batch_size, num_attention_heads, key_length, attn_head_size))
        )
        value = (
            value.reshape((batch_size, num_attention_groups, 1, key_length, attn_head_size))
            .tile(repeat_times=[1, 1, group_size, 1, 1])
            .reshape((batch_size, num_attention_heads, key_length, attn_head_size))
        )
        query = query.reshape((batch_size * num_attention_heads, query_length, attn_head_size))
        key = key.reshape((batch_size * num_attention_heads, key_length, attn_head_size))
        attn_scores = paddle.zeros(
            shape=[batch_size * num_attention_heads, query_length, key_length], dtype=query.dtype
        )

        # attn_scores = paddle.add(1.0 * attn_scores, paddle.to_tensor(data=1.0, dtype=self.norm_factor.dtype,place=self.norm_factor.place) / self.norm_factor * paddle.bmm(query,key.transpose(perm=dim2perm(key.ndim,1, 2))))
        attn_scores = paddle.assign(
            paddle.baddbmm(
                attn_scores, query, key.transpose(dim2perm(key.ndim, 1, 2)), beta=1.0, alpha=(1.0 / self.norm_factor)
            )
        )
        attn_scores = attn_scores.reshape((batch_size, num_attention_heads, query_length, key_length))
        mask_value = paddle.finfo(dtype=attn_scores.dtype).min
        mask_value = paddle.to_tensor(data=mask_value, dtype=attn_scores.dtype)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_weights = nn.functional.softmax(x=attn_scores, axis=-1)
        attn_weights = attn_weights.to(value.dtype)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = paddle.matmul(x=attn_weights, y=value)
        return attn_output, attn_weights

    def _flash_attn(self, query, key, value, attention_mask=None, head_mask=None):
        assert head_mask is None, "head_mask is not supported in _flash_attn"
        query = query.transpose(perm=dim2perm(query.ndim, 1, 2))
        key = key.transpose(perm=dim2perm(key.ndim, 1, 2))
        value = value.transpose(perm=dim2perm(value.ndim, 1, 2))
        query_length = query.shape[1]
        causal = query_length != 1
        if attention_mask is not None:
            batch_size = query.shape[0]
            query, key, value, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query, key, value, attention_mask, query_length
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
            # todo
            attn_output_unpad = flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=0,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(query, key, value, 0, causal=causal)
        return attn_output, None

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = tuple(key_layer.shape)
        num_attention_heads = tuple(query_layer.shape)[2]
        # TODO
        key_layer = index_first_axis(
            key_layer.reshape((batch_size * kv_seq_len, num_key_value_heads, head_dim)), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape((batch_size * kv_seq_len, num_key_value_heads, head_dim)), indices_k
        )

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape((batch_size * kv_seq_len, num_attention_heads, head_dim)), indices_k
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
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)
        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


def swiglu(x):
    x1, x2 = x.chunk(chunks=2, axis=x.ndim - 1)
    return x1 * nn.functional.silu(x=x2)


def get_activation(act_name: str):
    if act_name == "gelu":
        return ACT2FN["gelu_new"]
    elif act_name == "swiglu":
        return swiglu
    else:
        return ACT2FN[act_name]


class CustomLlamaMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        h_to_4h_out_channels = config.ffn_hidden_size * 2 if config.hidden_act == "swiglu" else config.ffn_hidden_size
        self.dense_h_to_4h = nn.Linear(
            in_features=config.hidden_size,
            out_features=h_to_4h_out_channels,
            bias_attr=getattr(config, "mlp_fc1_bias", True),
        )
        self.dense_4h_to_h = nn.Linear(
            in_features=config.ffn_hidden_size,
            out_features=config.hidden_size,
            bias_attr=getattr(config, "mlp_fc2_bias", True),
        )
        self.act = get_activation(config.hidden_act)

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class CustomLlamaLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        norm_func = get_norm(config)
        self.input_layernorm = norm_func(config.hidden_size)
        self.post_attention_layernorm = norm_func(config.hidden_size)
        self.attention = CustomLlamaAttention(config)
        self.mlp = CustomLlamaMLP(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        layer_past=None,
        output_attentions=False,
    ):
        attn_in = self.input_layernorm(hidden_states)
        attention_layer_outputs = self.attention(
            attn_in,
            attention_mask=attention_mask,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attention_layer_outputs[0]
        outputs = attention_layer_outputs[1:]

        attn_output = attn_output + hidden_states
        mlp_input = self.post_attention_layernorm(attn_output)
        mlp_output = self.mlp(mlp_input)
        hidden_states = mlp_output + attn_output
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        return outputs


class CustomLlamaPreTrainedModel(PretrainedModel):
    config_class = CustomLlamaConfig
    base_model_prefix = "lm"
    _no_split_modules = ["CustomLlamaLayer"]


class CustomLlamaModel(CustomLlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embed_in = nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self.layers = nn.LayerList(sublayers=[CustomLlamaLayer(config) for _ in range(config.num_layers)])

        norm_func = get_norm(config)
        self.final_layer_norm = norm_func(config.hidden_size)

    # Initialize weights and apply final processing
    # self.post_init()

    def get_input_embeddings(self):
        return self.embed_in

    def set_input_embeddings(self, value):
        self.embed_in = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    def get_head_mask(
        self,
        head_mask: Optional[paddle.Tensor],
        num_hidden_layers: int,
        is_attention_chunked: bool = False,
    ) -> paddle.Tensor:
        """
        Prepare the head mask if needed.
        Args:
                head_mask (`paddle.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                        The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
                num_hidden_layers (`int`):
                        The number of hidden layers in the model.
                is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                        Whether or not the attentions scores are computed by chunks or not.
        Returns:
                `paddle.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
                `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.ndim == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand([num_hidden_layers, -1, -1, -1, -1])
        elif head_mask.ndim == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.ndim == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.cast(dtype=self.config.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def forward(
        self,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        head_mask: Optional[paddle.Tensor] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[paddle.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = tuple(input_ids.shape)
        elif inputs_embeds is not None:
            input_shape = tuple(inputs_embeds.shape)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = tuple(past_key_values[0][0].shape)[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        else:
            past_key_values = tuple([None] * self.config.num_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        if attention_mask is None:
            attention_mask = paddle.ones(shape=(batch_size, seq_length_with_past), dtype="bool")
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)

        if USE_FLASH_ATTN:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds
        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)
        hidden_states = self.final_layer_norm(hidden_states)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class CustomLlamaForCausalLM(CustomLlamaPreTrainedModel):
    _tied_weights_keys = ["embed_out.weight"]
    _keys_to_ignore_on_load_unexpected = [r"lm.layers.\d+.attention.rotary_emb.inv_freq"]

    def __init__(self, config):
        super().__init__(config)
        self.lm = CustomLlamaModel(config)
        self.embed_out = nn.Linear(in_features=config.hidden_size, out_features=config.vocab_size, bias_attr=False)

    # Initialize weights and apply final processing
    # self.post_init()

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[Tensor]]] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
                `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
                `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are
                only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks that can be used (see
                `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
                don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
                `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
                `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
                ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
        use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
                `past_key_values`).

        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.lm(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if labels is not None:
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_logits = shift_logits.reshape([-1, shift_logits.shape[-1]])
            labels = labels[:, 1:].contiguous()
            labels = labels.reshape([-1])
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits, labels)

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return (lm_loss,) + output if lm_loss is not None else output
        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **model_kwargs
    ):

        input_shape = tuple(input_ids.shape)

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = paddle.ones(shape=input_shape, dtype=input_ids.dtype)

        # cut decoder_input_ids if past is used
        if past_key_values and past_key_values[0] is not None:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({"attention_mask": attention_mask, "past_key_values": past_key_values})
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(axis=0, index=beam_idx) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
