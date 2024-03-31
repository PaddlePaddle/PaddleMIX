# """Paddle InternLM2 model."""
# import sys
# import paddle
# import math
# import warnings
# from typing import List, Optional, Tuple, Union
# from einops import rearrange

# import paddlenlp.transformers as transformers
# from paddlenlp.transformers.activations import ACT2FN
# from paddlenlp.transformers.model_outputs import BaseModelOutputWithPast
# from paddlenlp.transformers.model_utils import PretrainedModel


# try:
#     from transformers.generation.streamers import BaseStreamer
# except:  # noqa # pylint: disable=bare-except
#     BaseStreamer = None

# from .build_mlp import PLoRA
# from .configuration_internlm_xcomposer2 import InternLMXcomposer2Config as InternLM2Config
# _CONFIG_FOR_DOC = 'InternLM2Config'

# def masked_fill(x, mask, value):
#     y = paddle.full(x.shape, value, x.dtype)
#     return paddle.where(mask, y, x)

# def _make_causal_mask(input_ids_shape: list, dtype: paddle.dtype, past_key_values_length: int = 0):
#     """
#     Make causal mask used for bi-directional self-attention.
#     """
#     bsz, tgt_len = input_ids_shape
#     mask = paddle.full(shape=(tgt_len, tgt_len), fill_value=paddle.finfo(dtype).min)
#     mask_cond = paddle.arange(end=mask.shape[-1])

#     mask = masked_fill(mask, mask_cond < (mask_cond + 1).reshape([mask.shape[-1], 1]), 0)
#     mask = mask.astype(dtype)
#     if past_key_values_length > 0:
#         mask = paddle.concat(x=[paddle.zeros(shape=[tgt_len, past_key_values_length], dtype=dtype), mask], axis=-1)
#     return mask[(None), (None), :, :].expand(shape=[bsz, 1, tgt_len, tgt_len + past_key_values_length])

# def masked_fill(x, mask, value):
#     y = paddle.full(x.shape, value, x.dtype)
#     return paddle.where(mask, y, x)

# def _expand_mask(mask: paddle.Tensor, dtype: paddle.dtype, tgt_len: Optional[int] = None):
#     """
#     Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
#     """
#     bsz, src_len = mask.shape
#     tgt_len = tgt_len if tgt_len is not None else src_len
#     expanded_mask = mask[:, (None), (None), :].expand(shape=[bsz, 1, tgt_len, src_len]).astype(dtype)
#     inverted_mask = 1.0 - expanded_mask

#     return masked_fill(inverted_mask, inverted_mask.astype("bool"), paddle.finfo(dtype).min)

# class InternLM2RMSNorm(paddle.nn.Layer):

#     def __init__(self, hidden_size, eps=1e-06):
#         """InternLM2RMSNorm is equivalent to T5LayerNorm."""
#         super().__init__()
#         out_2 = paddle.create_parameter(shape=paddle.ones(shape=hidden_size
#             ).shape, dtype=paddle.ones(shape=hidden_size).numpy().dtype,
#             default_initializer=paddle.nn.initializer.Assign(paddle.ones(
#             shape=hidden_size)))
#         out_2.stop_gradient = not True
#         self.weight = out_2
#         self.variance_epsilon = eps

#     def forward(self, hidden_states):
#         input_dtype = hidden_states.dtype
#         hidden_states = hidden_states.to('float32')
#         variance = hidden_states.pow(y=2).mean(axis=-1, keepdim=True)
#         hidden_states = hidden_states * paddle.rsqrt(x=variance + self.
#             variance_epsilon)
#         return self.weight * hidden_states.to(input_dtype)


# class InternLM2RotaryEmbedding(paddle.nn.Layer):

#     def __init__(self, dim, max_position_embeddings=2048, base=10000,
#         device=None):
#         super().__init__()
#         self.dim = dim
#         self.max_position_embeddings = max_position_embeddings
#         self.base = base
#         inv_freq = 1.0 / self.base ** (paddle.arange(start=0, end=self.dim,
#             step=2).astype(dtype='float32') / self.dim)
#         self.register_buffer(name='inv_freq', tensor=inv_freq, persistable=
#             False)
#         self._set_cos_sin_cache(seq_len=max_position_embeddings,device='cuda', dtype=paddle.get_default_dtype())

#     def _set_cos_sin_cache(self, seq_len, device, dtype):
#         self.max_seq_len_cached = seq_len
#         t = paddle.arange(dtype=self.inv_freq.dtype, end=self.
#             max_seq_len_cached)
#         freqs = paddle.einsum('i,j->ij', t, self.inv_freq)
#         emb = paddle.concat(x=(freqs, freqs), axis=-1)
#         self.register_buffer(name='cos_cached', tensor=emb.cos().to(dtype),
#             persistable=False)
#         self.register_buffer(name='sin_cached', tensor=emb.sin().to(dtype),
#             persistable=False)

#     def forward(self, x, seq_len=None):
#         if seq_len > self.max_seq_len_cached:
#             self._set_cos_sin_cache(seq_len=seq_len, device='cuda', dtype=
#                 x.dtype)
#         return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:
#             seq_len].to(dtype=x.dtype)


# class InternLM2LinearScalingRotaryEmbedding(InternLM2RotaryEmbedding):
#     """InternLM2RotaryEmbedding extended with linear scaling.

#     Credits to the Reddit user /u/kaiokendev
#     """

#     def __init__(self, dim, max_position_embeddings=2048, base=10000,
#         device=None, scaling_factor=1.0):
#         self.scaling_factor = scaling_factor
#         super().__init__(dim, max_position_embeddings, base, device)

#     def _set_cos_sin_cache(self, seq_len, device, dtype):
#         self.max_seq_len_cached = seq_len
#         t = paddle.arange(dtype=self.inv_freq.dtype, end=self.
#             max_seq_len_cached)
#         t = t / self.scaling_factor
#         freqs = paddle.einsum('i,j->ij', t, self.inv_freq)
#         emb = paddle.concat(x=(freqs, freqs), axis=-1)
#         self.register_buffer(name='cos_cached', tensor=emb.cos().to(dtype),
#             persistable=False)
#         self.register_buffer(name='sin_cached', tensor=emb.sin().to(dtype),
#             persistable=False)


# class InternLM2DynamicNTKScalingRotaryEmbedding(InternLM2RotaryEmbedding):
#     """InternLM2RotaryEmbedding extended with Dynamic NTK scaling.

#     Credits to the Reddit users /u/bloc97 and /u/emozilla.
#     """

#     def __init__(self, dim, max_position_embeddings=2048, base=10000,
#         device=None, scaling_factor=1.0):
#         self.scaling_factor = scaling_factor
#         super().__init__(dim, max_position_embeddings, base, device)

#     def _set_cos_sin_cache(self, seq_len, device, dtype):
#         self.max_seq_len_cached = seq_len
#         if seq_len > self.max_position_embeddings:
#             base = self.base * (self.scaling_factor * seq_len / self.
#                 max_position_embeddings - (self.scaling_factor - 1)) ** (self
#                 .dim / (self.dim - 2))
#             inv_freq = 1.0 / base ** (paddle.arange(start=0, end=self.dim,
#                 step=2).astype(dtype='float32') / self.dim)
#             self.register_buffer(name='inv_freq', tensor=inv_freq,
#                 persistable=False)
#         t = paddle.arange(dtype=self.inv_freq.dtype, end=self.
#             max_seq_len_cached)
#         freqs = paddle.einsum('i,j->ij', t, self.inv_freq)
#         emb = paddle.concat(x=(freqs, freqs), axis=-1)
#         self.register_buffer(name='cos_cached', tensor=emb.cos().to(dtype),
#             persistable=False)
#         self.register_buffer(name='sin_cached', tensor=emb.sin().to(dtype),
#             persistable=False)


# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., :x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2:]
#     return paddle.concat(x=(-x2, x1), axis=-1)


# def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
#     cos = cos.squeeze(axis=1).squeeze(axis=0)
#     sin = sin.squeeze(axis=1).squeeze(axis=0)
#     cos = cos.unsqueeze(axis=0).unsqueeze(axis=0).expand(shape=[len(
#         position_ids), -1, -1, -1])
#     sin = sin.unsqueeze(axis=0).unsqueeze(axis=0).expand(shape=[len(
#         position_ids), -1, -1, -1])
#     if q.shape[2] == 1:
#         q_embed = q * cos[:, :, -1:, :] + rotate_half(q) * sin[:, :, -1:, :]
#     else:
#         q_embed = q * cos + rotate_half(q) * sin
#     if k.shape[2] == 1:
#         k_embed = k * cos[:, :, -1:, :] + rotate_half(k) * sin[:, :, -1:, :]
#     else:
#         k_embed = k * cos + rotate_half(k) * sin
#     return q_embed, k_embed


# class InternLM2MLP(paddle.nn.Layer):

#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.intermediate_size = config.intermediate_size
#         self.w1 = PLoRA(self.hidden_size, self.intermediate_size, bias=
#             False, lora_r=256, lora_alpha=256, lora_len=576)
#         self.w3 = PLoRA(self.hidden_size, self.intermediate_size, bias=
#             False, lora_r=256, lora_alpha=256, lora_len=576)
#         self.w2 = PLoRA(self.intermediate_size, self.hidden_size, bias=
#             False, lora_r=256, lora_alpha=256, lora_len=576)
#         self.act_fn = ACT2FN[config.hidden_act]

#     def forward(self, x, im_mask):
#         down_proj = self.w2(self.act_fn(self.w1(x, im_mask)) * self.w3(x,
#             im_mask), im_mask)
#         return down_proj


# def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) ->paddle.Tensor:
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(shape=[batch,
#         num_key_value_heads, n_rep, slen, head_dim])
#     return hidden_states.reshape([batch, num_key_value_heads * n_rep, slen,
#         head_dim])


# class InternLM2Attention(paddle.nn.Layer):
#     """Multi-headed attention from 'Attention Is All You Need' paper."""

#     def __init__(self, config: InternLM2Config):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.num_heads = config.num_attention_heads
#         self.head_dim = self.hidden_size // self.num_heads
#         self.num_key_value_heads = config.num_key_value_heads
#         self.num_key_value_groups = self.num_heads // self.num_key_value_heads
#         self.max_position_embeddings = config.max_position_embeddings
#         self.is_causal = True
#         if self.head_dim * self.num_heads != self.hidden_size:
#             raise ValueError(
#                 f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads}).'
#                 )
#         self.wqkv = PLoRA(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=config.bias, lora_r=256, lora_alpha=256, lora_len=576)
#         self.wo = PLoRA(self.num_heads * self.head_dim, self.hidden_size,
#             bias=config.bias, lora_r=256, lora_alpha=256, lora_len=576)
#         self._init_rope()

#     def _init_rope(self):
#         if self.config.rope_scaling is None:
#             self.rotary_emb = InternLM2RotaryEmbedding(self.head_dim,
#                 max_position_embeddings=self.max_position_embeddings, base=
#                 self.config.rope_theta)
#         else:
#             scaling_type = self.config.rope_scaling['type']
#             scaling_factor = self.config.rope_scaling['factor']
#             if scaling_type == 'dynamic':
#                 self.rotary_emb = InternLM2DynamicNTKScalingRotaryEmbedding(
#                     self.head_dim, max_position_embeddings=self.
#                     max_position_embeddings, base=self.config.rope_theta,
#                     scaling_factor=scaling_factor)
#             else:
#                 raise ValueError(
#                     "Currently we only support rotary embedding's type being 'dynamic'."
#                     )
#         return self.rotary_emb

#     def _shape(self, tensor: paddle.Tensor, seq_len: int, bsz: int):
#         x = tensor.reshape([bsz, seq_len, self.num_heads, self.head_dim])
#         perm_0 = list(range(x.ndim))
#         perm_0[1] = 2
#         perm_0[2] = 1
#         return x.transpose(perm=perm_0)

#     def forward(self, hidden_states: paddle.Tensor, attention_mask:
#         Optional[paddle.Tensor]=None, position_ids: Optional[paddle.Tensor]
#         =None, past_key_value: Optional[Tuple[paddle.Tensor]]=None,
#         output_attentions: bool=False, use_cache: bool=False, im_mask:
#         Optional[Tuple[paddle.Tensor]]=None, **kwargs) ->Tuple[paddle.
#         Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
#         if 'padding_mask' in kwargs:
#             warnings.warn(
#                 'Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`'
#                 )


#         bsz, q_len, _ = hidden_states.shape
#         qkv_states = self.wqkv(hidden_states, im_mask)
#         qkv_states = rearrange(qkv_states, 'b q (h gs d) -> b q h gs d', gs
#             =2 + self.num_key_value_groups, d=self.head_dim)
#         query_states = qkv_states[..., :self.num_key_value_groups, :]
#         query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
#         key_states = qkv_states[..., -2, :]
#         value_states = qkv_states[..., -1, :]
#         x = query_states
#         perm_1 = list(range(x.ndim))
#         perm_1[1] = 2
#         perm_1[2] = 1
#         query_states = x.transpose(perm=perm_1)
#         x = key_states
#         perm_2 = list(range(x.ndim))
#         perm_2[1] = 2
#         perm_2[2] = 1
#         key_states = x.transpose(perm=perm_2)
#         x = value_states
#         perm_3 = list(range(x.ndim))
#         perm_3[1] = 2
#         perm_3[2] = 1
#         value_states = x.transpose(perm=perm_3)
#         kv_seq_len = key_states.shape[-2]
#         if past_key_value is not None:
#             kv_seq_len += past_key_value[0].shape[-2]
#         cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
#         query_states, key_states = apply_rotary_pos_emb(query_states,
#             key_states, cos, sin, position_ids)
#         if past_key_value is not None:
#             key_states = paddle.concat(x=[past_key_value[0], key_states],
#                 axis=2)
#             value_states = paddle.concat(x=[past_key_value[1], value_states
#                 ], axis=2)
#         past_key_value = (key_states, value_states) if use_cache else None
#         key_states = repeat_kv(key_states, self.num_key_value_groups)
#         value_states = repeat_kv(value_states, self.num_key_value_groups)
#         x = key_states
#         perm_4 = list(range(x.ndim))
#         perm_4[2] = 3
#         perm_4[3] = 2
#         attn_weights = paddle.matmul(x=query_states, y=x.transpose(perm=perm_4)
#             ) / math.sqrt(self.head_dim)
#         if attn_weights.shape != [bsz, self.num_heads, q_len, kv_seq_len]:
#             raise ValueError(
#                 f'Attention weights should be of size {bsz, self.num_heads, q_len, kv_seq_len}, but is {attn_weights.shape}'
#                 )
#         if attention_mask is not None:
#             if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
#                 raise ValueError(
#                     f'Attention mask should be of size {bsz, 1, q_len, kv_seq_len}, but is {attention_mask.shape}'
#                     )
#             attn_weights = attn_weights + attention_mask
#         attn_weights = paddle.nn.functional.softmax(x=attn_weights, axis=-1,
#             dtype='float32').to(query_states.dtype)
#         attn_output = paddle.matmul(x=attn_weights, y=value_states)
#         if attn_output.shape != [bsz, self.num_heads, q_len, self.head_dim]:
#             raise ValueError(
#                 f'`attn_output` should be of size {bsz, self.num_heads, q_len, self.head_dim}, but is {attn_output.shape}'
#                 )
#         x = attn_output
#         perm_5 = list(range(x.ndim))
#         perm_5[1] = 2
#         perm_5[2] = 1
#         attn_output = x.transpose(perm=perm_5)
#         attn_output = attn_output.reshape([bsz, q_len, self.hidden_size])
#         attn_output = self.wo(attn_output, im_mask)
#         if not output_attentions:
#             attn_weights = None
#         return attn_output, attn_weights, past_key_value


# class InternLM2FlashAttention2(InternLM2Attention):
#     """InternLM2 flash attention module.

#     This module inherits from `InternLM2Attention` as the weights of the module
#     stays untouched. The only required change would be on the forward pass
#     where it needs to correctly call the public API of flash attention and deal
#     with padding tokens in case the input contains any of them.
#     """

#     def forward(self, hidden_states: paddle.Tensor, attention_mask:
#         Optional[paddle.Tensor]=None, position_ids: Optional[paddle.Tensor]
#         =None, past_key_value: Optional[Tuple[paddle.Tensor]]=None,
#         output_attentions: bool=False, use_cache: bool=False, im_mask:
#         Optional[Tuple[paddle.Tensor]]=None, **kwargs) ->Tuple[paddle.
#         Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
#         if 'padding_mask' in kwargs:
#             warnings.warn(
#                 'Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`'
#                 )
#             attention_mask = kwargs.pop('padding_mask')
#         output_attentions = False
#         bsz, q_len, _ = hidden_states.shape
#         qkv_states = self.wqkv(hidden_states, im_mask)
#         qkv_states = rearrange(qkv_states, 'b q (h gs d) -> b q h gs d', gs
#             =self.num_heads + 2 * self.num_key_value_heads, d=self.head_dim,
#             q=q_len)
#         query_states = qkv_states[..., :self.num_key_value_groups, :]
#         query_states = rearrange(query_states, 'b q h gs d -> b q (h gs) d')
#         key_states = qkv_states[..., -2, :]
#         value_states = qkv_states[..., -1, :]
#         kv_seq_len = key_states.shape[-2]
#         if past_key_value is not None:
#             kv_seq_len += past_key_value[0].shape[-2]
#         cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
#         query_states, key_states = apply_rotary_pos_emb(query_states,
#             key_states, cos, sin, position_ids)
#         if past_key_value is not None:
#             key_states = paddle.concat(x=[past_key_value[0], key_states],
#                 axis=2)
#             value_states = paddle.concat(x=[past_key_value[1], value_states
#                 ], axis=2)
#         past_key_value = (key_states, value_states) if use_cache else None
#         x = query_states
#         perm_6 = list(range(x.ndim))
#         perm_6[1] = 2
#         perm_6[2] = 1
#         query_states = x.transpose(perm=perm_6)
#         x = key_states
#         perm_7 = list(range(x.ndim))
#         perm_7[1] = 2
#         perm_7[2] = 1
#         key_states = x.transpose(perm=perm_7)
#         x = value_states
#         perm_8 = list(range(x.ndim))
#         perm_8[1] = 2
#         perm_8[2] = 1
#         value_states = x.transpose(perm=perm_8)
#         dropout_rate = 0.0 if not self.training else self.attention_dropout
#         input_dtype = query_states.dtype
#         if input_dtype == 'float32':
#             if hasattr(self.config, '_pre_quantization_dtype'):
#                 target_dtype = self.config._pre_quantization_dtype
#             else:
#                 target_dtype = self.q_proj.weight.dtype
#             logger.warning_once(
#                 f'The input hidden states seems to be silently casted in float32, this might be related to the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in {target_dtype}.'
#                 )
#             query_states = query_states.to(target_dtype)
#             key_states = key_states.to(target_dtype)
#             value_states = value_states.to(target_dtype)
#         attn_output = self._flash_attention_forward(query_states,
#             key_states, value_states, attention_mask, q_len, dropout=
#             dropout_rate)
#         attn_output = attn_output.reshape([bsz, q_len, self.hidden_size])
#         attn_output = self.wo(attn_output, im_mask)
#         if not output_attentions:
#             attn_weights = None
#         return attn_output, attn_weights, past_key_value


# class InternLM2DecoderLayer(paddle.nn.Layer):

#     def __init__(self, config: InternLM2Config):
#         super().__init__()
#         self.hidden_size = config.hidden_size
#         self.attention = InternLM2Attention(config=config) if not getattr(config, '_flash_attn_2_enabled', False) else InternLM2FlashAttention2(config=config)
#         self.feed_forward = InternLM2MLP(config)
#         self.attention_norm = InternLM2RMSNorm(config.hidden_size, eps=
#             config.rms_norm_eps)
#         self.ffn_norm = InternLM2RMSNorm(config.hidden_size, eps=config.
#             rms_norm_eps)

#     def forward(self, hidden_states: paddle.Tensor, attention_mask:
#         Optional[paddle.Tensor]=None, position_ids: Optional[paddle.Tensor]
#         =None, past_key_value: Optional[Tuple[paddle.Tensor]]=None,
#         output_attentions: Optional[bool]=False, use_cache: Optional[bool]=
#         False, im_mask: Optional[Tuple[paddle.Tensor]]=None, **kwargs) ->Tuple[
#         paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
#         if 'padding_mask' in kwargs:
#             warnings.warn(
#                 'Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`'
#                 )
#         residual = hidden_states
#         hidden_states = self.attention_norm(hidden_states)
#         hidden_states, self_attn_weights, present_key_value = self.attention(
#             hidden_states=hidden_states, attention_mask=attention_mask,
#             position_ids=position_ids, past_key_value=past_key_value,
#             output_attentions=output_attentions, use_cache=use_cache,
#             im_mask=im_mask, **kwargs)
#         hidden_states = residual + hidden_states
#         residual = hidden_states
#         hidden_states = self.ffn_norm(hidden_states)
#         hidden_states = self.feed_forward(hidden_states, im_mask)
#         hidden_states = residual + hidden_states
#         outputs = (hidden_states,)
#         if output_attentions:
#             outputs += self_attn_weights,
#         if use_cache:
#             outputs += present_key_value,
#         return outputs




# class InternLM2PretrainedModel(PretrainedModel):
#     config_class = InternLM2Config
#     base_model_prefix = 'model'
#     supports_gradient_checkpointing = True
#     _no_split_modules = ['InternLM2DecoderLayer']
#     _skip_keys_device_placement = 'past_key_values'
#     _supports_flash_attn_2 = True


# class InternLM2Model(InternLM2PretrainedModel):
#     """Transformer decoder consisting of *config.num_hidden_layers* layers.
#     Each layer is a [`InternLM2DecoderLayer`]

#     Args:
#         config: InternLM2Config
#     """
#     _auto_class = 'AutoModel'

#     def __init__(self, config: InternLM2Config):
#         super().__init__(config)
#         self.padding_idx = config.pad_token_id
#         self.vocab_size = config.vocab_size
#         self.tok_embeddings = paddle.nn.Embedding(num_embeddings=config.
#             vocab_size, embedding_dim=config.hidden_size, padding_idx=self.
#             padding_idx)
#         self.layers = paddle.nn.LayerList(sublayers=[InternLM2DecoderLayer(
#             config) for _ in range(config.num_hidden_layers)])
#         self.norm = InternLM2RMSNorm(config.hidden_size, eps=config.
#             rms_norm_eps)
#         self.gradient_checkpointing = False

#     def get_input_embeddings(self):
#         return self.tok_embeddings

#     def set_input_embeddings(self, value):
#         self.tok_embeddings = value

#     def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
#         combined_attention_mask = None
#         if input_shape[-1] > 1:
#             combined_attention_mask = _make_causal_mask(
#                 input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
#             )
#         if attention_mask is not None:
#             expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
#             combined_attention_mask = (
#                 expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
#             )
#         return combined_attention_mask

#     def forward(self, 
#                 input_ids: paddle.Tensor=None, 
#                 attention_mask: Optional[paddle.Tensor]=None, 
#                 position_ids: Optional[paddle.Tensor]=None, 
#                 past_key_values: Optional[List[paddle.Tensor]]=None,
#                 inputs_embeds: Optional[paddle.Tensor]=None, 
#                 use_cache: Optional[bool]=None, 
#                 output_attentions: Optional[bool]=None,
#                 output_hidden_states: Optional[bool]=None, 
#                 return_dict: Optional[bool]=None, **kwargs) ->Union[Tuple, BaseModelOutputWithPast]:
#         im_mask = kwargs.get('im_mask', None)
#         output_attentions = (output_attentions if output_attentions is not
#             None else self.config.output_attentions)
#         output_hidden_states = (output_hidden_states if 
#             output_hidden_states is not None else self.config.
#             output_hidden_states)
#         use_cache = (use_cache if use_cache is not None else self.config.
#             use_cache)
#         return_dict = (return_dict if return_dict is not None else self.
#             config.use_return_dict)
#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError(
#                 'You cannot specify both input_ids and inputs_embeds at the same time'
#                 )
#         elif input_ids is not None:
#             batch_size, seq_length = input_ids.shape[:2]
#         elif inputs_embeds is not None:
#             batch_size, seq_length = inputs_embeds.shape[:2]
#         else:
#             raise ValueError(
#                 'You have to specify either input_ids or inputs_embeds')
#         seq_length_with_past = seq_length
#         past_key_values_length = 0
#         if past_key_values is not None:
#             past_key_values_length = past_key_values[0][0].shape[2]
#             seq_length_with_past = (seq_length_with_past +
#                 past_key_values_length)
#         if position_ids is None:
#             position_ids = paddle.arange(start=past_key_values_length, end=
#                 seq_length + past_key_values_length, dtype='int64')
#             position_ids = position_ids.unsqueeze(axis=0)
#         if inputs_embeds is None:
#             inputs_embeds = self.tok_embeddings(input_ids)
#             im_mask = paddle.zeros(shape=inputs_embeds.shape[:2]).astype(dtype='bool')
#         if attention_mask is None:
#             attention_mask = paddle.ones(shape=(batch_size,
#                 seq_length_with_past), dtype='bool')
#         attention_mask = self._prepare_decoder_attention_mask(attention_mask,
#             (batch_size, seq_length), inputs_embeds, past_key_values_length)
#         hidden_states = inputs_embeds
#         if self.gradient_checkpointing and self.training:
#             if use_cache:
#                 logger.warning_once(
#                     '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
#                     )
#                 use_cache = False
#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         next_decoder_cache = () if use_cache else None
#         for idx, decoder_layer in enumerate(self.layers):
#             if output_hidden_states:
#                 all_hidden_states += hidden_states,
#             past_key_value = past_key_values[idx
#                 ] if past_key_values is not None else None
#             if self.gradient_checkpointing and self.training:

#                 # def create_custom_forward(module):

#                 #     def custom_forward(*inputs):
#                 #         return module(*inputs, output_attentions, None, im_mask
#                 #             )
#                 #     return custom_forward
#                 # layer_outputs = torch.utils.checkpoint.checkpoint(
#                 #     create_custom_forward(decoder_layer), hidden_states,
#                 #     attention_mask, position_ids, None)
#                 pass
#                 # todo: checkpoint api in paddle?
#             else:
#                 layer_outputs = decoder_layer(hidden_states, attention_mask
#                     =attention_mask, position_ids=position_ids,
#                     past_key_value=past_key_value, output_attentions=
#                     output_attentions, use_cache=use_cache, im_mask=im_mask)
#             hidden_states = layer_outputs[0]
#             if use_cache:
#                 next_decoder_cache += (layer_outputs[2 if output_attentions else 1], )
#             if output_attentions:
#                 all_self_attns += layer_outputs[1],
#         hidden_states = self.norm(hidden_states)
#         if output_hidden_states:
#             all_hidden_states += hidden_states,
#         next_cache = next_decoder_cache if use_cache else None
#         if not return_dict:
#             return tuple(v for v in [hidden_states, next_cache,
#                 all_hidden_states, all_self_attns] if v is not None)
#         return BaseModelOutputWithPast(
#             last_hidden_state=hidden_states, past_key_values=next_cache,
#             hidden_states=all_hidden_states, attentions=all_self_attns)
