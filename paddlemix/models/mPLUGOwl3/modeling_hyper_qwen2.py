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

from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddlenlp
from einops import rearrange, repeat
from paddle.nn import MultiHeadAttention
from paddlenlp.transformers.qwen2.modeling import Qwen2Attention

from paddlemix.utils.log import logger

from ...activations import ACT2FN
from .configuration_hyper_qwen2 import HyperQwen2Config


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


class Qwen2RMSNorm(paddle.nn.Layer):
    def __init__(self, hidden_size, eps=1e-06):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.ones(shape=hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to("float32")
        variance = hidden_states.pow(y=2).mean(axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(x=variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2RotaryEmbedding(nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / self.base ** (
            paddle.arange(start=0, end=self.dim, step=2, dtype="int64").astype(dtype="float32") / self.dim
        )
        self.register_buffer(name="inv_freq", tensor=inv_freq, persistable=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, dtype=paddle.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(dtype="int64", end=self.max_seq_len_cached).astype(dtype=self.inv_freq.dtype)
        freqs = paddle.outer(x=t, y=self.inv_freq)
        emb = paddle.concat(x=(freqs, freqs), axis=-1)
        self.register_buffer(name="cos_cached", tensor=emb.cos().to(dtype), persistable=False)
        self.register_buffer(name="sin_cached", tensor=emb.sin().to(dtype), persistable=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class RotaryEmbedding(paddle.nn.Layer):
    def __init__(self, dim, base=10000, use_fp32=False, use_outer_in_rope=False):
        super().__init__()
        self.dim = dim
        self.base = base
        self.use_fp32 = use_fp32
        if use_fp32:
            self.inv_freq = 1.0 / base ** (paddle.arange(start=0, end=dim, step=2).astype(dtype="float32") / dim)
        else:
            inv_freq = 1.0 / base ** (paddle.arange(start=0, end=dim, step=2).astype(dtype="float32") / dim)
            self.register_buffer(name="inv_freq", tensor=inv_freq)

        self._rotary_pos_emb_cache = None
        self._seq_len_cached = 0
        self.use_outer_in_rope = use_outer_in_rope
        self._ntk_alpha_cached = 1.0

    def update_rotary_pos_emb_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
        seqlen = max_seq_len + offset
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / base ** (
                paddle.arange(start=0, end=self.dim, step=2).astype(dtype="float32") / self.dim
            )
            self._seq_len_cached = seqlen
            self._ntk_alpha_cached = ntk_alpha
            seq = paddle.arange(end=seqlen)
            if 1:  # self.use_outer_in_rope:
                freqs = paddle.outer(x=seq.astype(dtype=self.inv_freq.dtype), y=self.inv_freq)
            # else:
            #    freqs = einsum("i , j -> i j", seq.astype(dtype=self.inv_freq.dtype), self.inv_freq)
            emb = paddle.concat(x=(freqs, freqs), axis=-1)
            # emb [seq_length, .., dim]
            self._rotary_pos_emb_cache = rearrange(emb, "n d -> n 1 1 d")

    def forward(self, max_seq_len, offset=0, ntk_alpha=1.0):
        self.update_rotary_pos_emb_cache(max_seq_len, offset, ntk_alpha)
        return self._rotary_pos_emb_cache[offset : offset + max_seq_len]


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : tuple(x.shape)[-1] // 2]
    x2 = x[..., tuple(x.shape)[-1] // 2 :]
    return paddle.concat(x=(-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
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
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2MLP(nn.Layer):
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
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = tuple(hidden_states.shape)
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(shape=[batch, num_key_value_heads, n_rep, slen, head_dim])
    return hidden_states.reshape([batch, num_key_value_heads * n_rep, slen, head_dim])


def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(axis=-2)
    return paddle.concat(x=(-x2, x1), axis=-1)


def apply_rotary_pos_emb_core(t, freqs, use_fp32=False, debug=False):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    # if use_flash_rotary and use_fp32:
    #     t_ = rearrange(t, "s b ... -> b s ...")
    #     if use_fp32:
    #         t_ = t_.astype(dtype="float32")
    #     freqs = freqs.squeeze(axis=1).squeeze(axis=1)
    #     cos = freqs[:, :freqs.shape[-1] // 2].cos()
    #     sin = freqs[:, :freqs.shape[-1] // 2].sin()
    #     output = apply_rotary_emb_func(t_, cos, sin).astype(dtype=t.dtype) # TODO
    #     return rearrange(output, 'b s ... -> s b ...')

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]

    if use_fp32:
        t_ = t_.astype(dtype="float32")
        t_pass_ = t_pass_.astype(dtype="float32")
    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t_ = (t_ * freqs.cos()) + (_rotate_half(t_) * freqs.sin())
    return paddle.concat(x=(t_, t_pass_), axis=-1).astype(dtype=t.dtype)


class HyperQwen2Attention(nn.Layer):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: HyperQwen2Config, layer_idx: Optional[int] = None, is_hyper_enabled=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias_attr=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias_attr=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias_attr=False)

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.rotary_emb_core = RotaryEmbedding(
            self.head_dim, base=self.rope_theta, use_fp32=True, use_outer_in_rope=True
        )
        # Hyper Attention Modules
        self.is_hyper_enabled = is_hyper_enabled
        if self.is_hyper_enabled:
            self.v_kv_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim * 2, bias_attr=True)

            self.visual_cache = {}

        self.use_flexattention = True

    def apply_mi_rope(self, key_layer, image_pos, length_each_img):
        # input shape should be [s b h d]
        key_layer = rearrange(key_layer, "b h s d -> s b h d")
        rotary_pos_emb_max_seq_len = self.config.max_position_embeddings
        ntk_alpha = 1
        rotary_pos_emb = self.rotary_emb_core(rotary_pos_emb_max_seq_len, ntk_alpha=ntk_alpha)
        assert rotary_pos_emb is not None

        if isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = rotary_pos_emb
        else:
            rotary_pos_emb = (rotary_pos_emb,) * 2

        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            k_pos_emb = repeat(k_pos_emb[image_pos], "N_img b h d -> (N_img L) b h d", L=length_each_img)  # N_img, dim

            key_layer = apply_rotary_pos_emb_core(key_layer, k_pos_emb, use_fp32=True)  # TODO difference
        key_layer = rearrange(key_layer, "s b h d -> b h s d")
        return key_layer


class HyperQwen2SdpaAttention(HyperQwen2Attention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def hyperattention(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        image_embeds=None,
        media_offset=None,
        past_key_value: Optional[MultiHeadAttention.Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape  # (1, 74, 28, 128) bsz, q_len, self.num_heads, self.head_dim

        try:
            query_states = self.q_proj(hidden_states)
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

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [1, 28, 1, 128] [1, 4, 1, 128]

        if past_key_value is not None:
            key_states = paddle.concat([past_key_value[0], key_states], axis=2)
            value_states = paddle.concat([past_key_value[1], value_states], axis=2)
        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # add visual to kv
        length_each_img = image_embeds.shape[1]
        try:
            image_embeds = self.v_kv_proj(image_embeds)
        except:
            image_embeds = self.v_kv_proj(image_embeds.astype(self.v_kv_proj.weight.dtype))
        image_start = 0
        context_layer = []
        for bi, media_starts in enumerate(media_offset):
            num_images = media_starts.shape[0]
            if num_images > 0:
                if q_len == 1:
                    full_mask = paddle.ones((1, 1, 1, num_images * length_each_img + kv_seq_len)).astype(paddle.bool)
                else:
                    causal_mask = paddle.tril(paddle.ones([q_len, kv_seq_len])).astype(paddle.bool)
                    # 扩展维度以匹配 (bsz, 1, q_len, kv_seq_len)
                    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

                    matrix = paddle.arange(q_len).reshape([-1, 1])
                    t2vmask = ~(matrix < media_starts.reshape([1, -1]))
                    t2vmask = repeat(t2vmask, "seq_t seq_v -> 1 1 seq_t (seq_v v_token)", v_token=length_each_img)
                    full_mask = paddle.concat(
                        [t2vmask, causal_mask], axis=3
                    )  # unsqueeze batch dim (batch, 1, seq_q, seq_k)

                curr_query_layer = query_states[bi : bi + 1]
                # order is sbhd
                curr_visual_key_layer, curr_visual_value_layer = rearrange(
                    image_embeds[image_start : image_start + num_images],
                    "BL Lv (H KV D) -> KV 1 H (BL Lv) D",
                    KV=2,
                    H=self.num_key_value_heads,
                )  # b h s d
                image_start += num_images

                curr_visual_key_layer = self.apply_mi_rope(
                    curr_visual_key_layer, media_starts, length_each_img=length_each_img
                )

                curr_visual_key_layer = repeat_kv(curr_visual_key_layer, self.num_key_value_groups)
                curr_visual_value_layer = repeat_kv(curr_visual_value_layer, self.num_key_value_groups)

                curr_key_layer = paddle.concat([curr_visual_key_layer, key_states[bi : bi + 1]], axis=2)
                curr_value_layer = paddle.concat([curr_visual_value_layer, value_states[bi : bi + 1]], axis=2)
                is_causal = False
            else:
                # 执行无图attention
                curr_query_layer = query_states[bi : bi + 1]
                curr_key_layer = key_states[bi : bi + 1]
                curr_value_layer = value_states[bi : bi + 1]
                is_causal = True if q_len > 1 else False
                if is_causal:
                    full_mask = None
                else:
                    causal_mask = paddle.tril(paddle.ones([q_len, kv_seq_len])).astype(paddle.bool)
                    # 扩展维度以匹配 (bsz, 1, q_len, kv_seq_len)
                    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
                    full_mask = causal_mask

            # Note: 注意paddle的scaled_dot_product_attention 中q k v维度与torch不同
            attn_output = paddle.nn.functional.scaled_dot_product_attention(
                curr_query_layer.transpose(
                    [0, 2, 1, 3]
                ),  # (batch, ..., sequence, dim) # [1, 74, 28, 128], torch [1, 28, 74, 128] sum 18304.
                curr_key_layer.transpose(
                    [0, 2, 1, 3]
                ),  # [1, 5177, 28, 128], torch [1, 28, 5177, 128] sum 1044480   mean 0.05615234  torch sum 1036288. mean 0.0559
                curr_value_layer.transpose([0, 2, 1, 3]),  # [1, 5177, 28, 128] , torch [1, 28, 5177, 128] sum -158720
                attn_mask=full_mask.cast(
                    curr_query_layer.dtype
                ),  # (N, ..., L, S) A boolean mask where a value of True indicates that the element *should* take part in attention.
                dropout_p=self.attention_dropout if self.training else 0.0,
                # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
                is_causal=is_causal,
            )  # -> (N, ..., L, Ev)
            # torch attn_output.shape [1, 28, 72, 128]
            attn_output = attn_output.transpose([0, 2, 1, 3])
            assert attn_output.shape[0] == 1
            context_layer.append(attn_output)
        attn_output = context_layer = paddle.concat(context_layer, axis=0)

        attn_output = attn_output.transpose([0, 2, 1, 3])
        attn_output = attn_output.reshape([bsz, q_len, self.hidden_size])

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        image_embeds=None,
        media_offset=None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        # TODO:
        # if output_attentions:
        #     # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        #     logger.warning_once(
        #         "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
        #         'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        #     )
        #     return super().forward(
        #         hidden_states=hidden_states,
        #         attention_mask=attention_mask,
        #         position_ids=position_ids,
        #         past_key_value=past_key_value,
        #         output_attentions=output_attentions,
        #         use_cache=use_cache,
        #     )

        if self.is_hyper_enabled and image_embeds is not None:
            return self.hyperattention(
                hidden_states,
                attention_mask,
                position_ids,
                image_embeds,
                media_offset,
                past_key_value,
                output_attentions,
                use_cache,
            )

        bsz, q_len, _ = hidden_states.shape

        try:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        except:
            hidden_states = hidden_states.astype(self.q_proj.weight.dtyp)
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        key_states = key_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose([0, 2, 1, 3])
        value_states = value_states.reshape([bsz, q_len, self.num_key_value_heads, self.head_dim]).transpose(
            [0, 2, 1, 3]
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = paddle.concat([past_key_value[0], key_states], axis=2)
            value_states = paddle.concat([past_key_value[1], value_states], axis=2)
        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if tuple(attention_mask.shape) != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {bsz, 1, q_len, kv_seq_len}, but is {tuple(attention_mask.shape)}"
                )

        # Note: 注意paddle的scaled_dot_product_attention 中q k v维度与torch不同
        attn_output = paddle.nn.functional.scaled_dot_product_attention(
            query_states.transpose([0, 2, 1, 3]),  # [1, 28, 74, 128] sum 21632.
            key_states.transpose([0, 2, 1, 3]),  # [1, 28, 74, 128] sum 335872.
            value_states.transpose([0, 2, 1, 3]),  # [1, 28, 74, 128] sum 1680.
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )
        # [1, 74, 28, 128] sum 1408.
        attn_output = attn_output.reshape([bsz, q_len, self.hidden_size])

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


# Original Attention of Qwen2
# PaddleNLP only has Qwen2Attention
QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention,
    "flash_attention_2": Qwen2Attention,  # Qwen2FlashAttention2,
    "sdpa": Qwen2Attention,  # Qwen2SdpaAttention,
}


class HyperQwen2DecoderLayer(nn.Layer):
    def __init__(self, config: HyperQwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.is_hyper_enabled = (layer_idx + 1) in config.hyper_layers
        # TODO: 若使用Qwen2Attention则回答结果不对，若都使用HyperQwen2SdpaAttention回答结果也对，但需check一下
        if 1:  # self.is_hyper_enabled:
            self.self_attn = HyperQwen2SdpaAttention(config, layer_idx, is_hyper_enabled=self.is_hyper_enabled)
        else:
            # self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
            self.self_attn = QWEN2_ATTENTION_CLASSES["flash_attention_2"](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        image_embeds=None,
        media_offset=None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        """
        Args:
            hidden_states (`paddle.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`paddle.Tensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(paddle.Tensor)`, *optional*): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Shared LayerNorm
        if image_embeds is not None and self.is_hyper_enabled:
            image_embeds = self.input_layernorm(image_embeds)
            media_kwargs = {"image_embeds": image_embeds, "media_offset": media_offset}
        else:
            image_embeds = media_offset = None
            media_kwargs = {}

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(  # -704. 2080. (48128., 240.)
            hidden_states=hidden_states.cast(self.mlp.gate_proj.weight.dtype),  # [1, 74, 3584] sum -704.
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=True,  # TODO, paddlenlp默认是False，但是不返回self_attn_weights。output_attentions全局是false，这里改成True是无影响的
            use_cache=use_cache,
            **media_kwargs,  # {}
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        try:
            hidden_states = self.mlp(hidden_states.cast(self.mlp.gate_proj.weight.dtype))
        except:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Qwen2PreTrainedModel(paddlenlp.transformers.model_utils.PretrainedModel):
    config_class = HyperQwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HyperQwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    # _supports_flash_attn_2 = True
    # _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, layer):
        std = self.config.initializer_range
        if isinstance(layer, (paddle.nn.Linear, paddle.nn.Conv3D)):
            paddle.nn.initializer.Normal(mean=0.0, std=std)(layer.weight)
            if layer.bias is not None:
                paddle.nn.initializer.Constant(0.0)(layer.bias)
        elif isinstance(layer, paddle.nn.Embedding):
            paddle.nn.initializer.Normal(mean=0.0, std=std)(layer.weight)
            if layer._padding_idx is not None:
                with paddle.no_grad():
                    layer.weight[layer._padding_idx] = 0.0


class HyperQwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: HyperQwen2Config
    """

    def __init__(self, config: HyperQwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.LayerList(
            [HyperQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = "flash_attention_2"  # config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        # self.post_init()

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
        image_embeds=None,
        media_offset=None,
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

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
        # NOTE: to make cache can be clear in-time
        past_key_values = list(past_key_values)

        past_key_values_length = seq_length
        cache_length = 0
        if past_key_values[0] is not None:
            cache_length = past_key_values[0][0].shape[1]  #
            past_key_values_length += cache_length

        if position_ids is None:
            position_ids = paddle.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=paddle.int64
            )
            position_ids = position_ids.unsqueeze(0).reshape([-1, seq_length])
        else:
            position_ids = position_ids.reshape([-1, seq_length]).astype(dtype="int64")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = None  #

        hidden_states = inputs_embeds

        # beam search
        if batch_size != len(media_offset):
            # The model is performing beamsearch, repeat the visual content
            beam_factor = batch_size // len(media_offset)
            assert batch_size % len(media_offset) == 0
            media_offset = media_offset * beam_factor
            image_embeds = repeat(image_embeds, "B L D -> (factor B) L D", factor=beam_factor)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = ()  # not None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_embeds=image_embeds,
                media_offset=media_offset,
                past_key_value=past_key_value,  # not past_key_values
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            # NOTE: clear outdate cache after it has been used for memory saving
            past_key_value = past_key_values[idx] = None

            hidden_states = layer_outputs[0]

            next_decoder_cache = next_decoder_cache + (layer_outputs[-1],) if use_cache else None

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return paddlenlp.transformers.model_outputs.BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class HyperQwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = HyperQwen2Model(config)
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
        image_embeds=None,
        media_offset=None,
        labels: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, paddlenlp.transformers.model_outputs.CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,  # [1, 74] # [1, 1]
            attention_mask=attention_mask,  # [1, 74] # [1, 75]
            position_ids=position_ids,  # [1, 74] # [1, 1]
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,  # none
            image_embeds=image_embeds,  # [7, 729, 3584] sum 134144.
            media_offset=media_offset,  # [[18, 24, 30, 36, 42, 48, 54]]
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,  #
        )

        hidden_states = outputs[0]  # sum 6656 mean 0.02502441
        try:
            logits = self.lm_head(hidden_states)
        except:
            logits = self.lm_head(hidden_states.cast(self.lm_head.weight.dtype))
        logits = logits.cast(paddle.float32)  # sum -5314405 mean -0.47356287

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.reshape([-1, self.config.vocab_size])
            shift_labels = shift_labels.reshape([-1])
            # Enable model parallelism
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
        # 以下这段参考PaddleNLP的 Qwen2ForCausalLM 的写法，与torch的mPLUG-owl3不同
        batch_size, seq_length = input_ids.shape
        position_ids = kwargs.get("position_ids", paddle.arange(seq_length).expand((batch_size, seq_length)))
        attention_mask = kwargs.get("attention_mask", None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(axis=-1)
            position_ids = position_ids[:, -1].unsqueeze(-1)

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
                "image_embeds": kwargs.get("image_embeds"),
                "media_offset": kwargs.get("media_offset"),
            }
        )
        return model_inputs
