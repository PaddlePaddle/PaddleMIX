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

"""Paddle InternLMXComposer2 model."""
import copy
import math
import queue
import threading
import warnings
from typing import List, Optional, Tuple, Union

import paddle
import paddlenlp.transformers as transformers
from einops import rearrange
from paddlenlp.transformers.model_outputs import CausalLMOutputWithPast
from PIL import Image

try:
    from transformers.generation.streamers import BaseStreamer
except:  # noqa # pylint: disable=bare-except
    BaseStreamer = None

from paddlenlp.transformers.activations import ACT2FN
from paddlenlp.transformers.model_outputs import BaseModelOutputWithPast
from paddlenlp.transformers.model_utils import PretrainedModel

from ppdiffusers.utils import logging

from .configuration import InternLMXcomposer2Config as InternLM2Config

_CONFIG_FOR_DOC = "InternLM2Config"
logger = logging.get_logger(__name__)


def build_vision_tower(config):
    vision_tower = config.vision_tower
    return CLIPVisionTower(vision_tower)


def build_vision_projector(config):
    mm_hidden_size = config["mm_hidden_size"]
    hidden_size = config["hidden_size"]
    mlp_depth = config["mlp_depth"]
    if mlp_depth > 0:
        modules = [paddle.nn.Linear(in_features=mm_hidden_size, out_features=hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(paddle.nn.GELU())
            modules.append(paddle.nn.Linear(in_features=hidden_size, out_features=hidden_size))
        return paddle.nn.Sequential(*modules)
    return IdentityMap()  # mlp_depth == 0


class IdentityMap(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class CLIPVisionTower(paddle.nn.Layer):
    def __init__(self, vision_tower):
        super().__init__()
        self.is_loaded = False
        self.is_resize_pos = False
        self.vision_tower_name = vision_tower
        self.select_layer = -1
        self.select_feature = "patch"
        self.load_model()
        self.resize_pos()

    def load_model(self):
        self.vision_tower = transformers.CLIPVisionModel.from_pretrained(self.vision_tower_name)
        for p in self.vision_tower.parameters():
            p.stop_gradient = True
        self.is_loaded = True
        if self.training:
            self.vision_tower.vision_model.ln_post = paddle.nn.Identity()

    def resize_pos(self):
        pos_embed_checkpoint = self.vision_tower.vision_model.positional_embedding.weight
        pos_embed_checkpoint = pos_embed_checkpoint.unsqueeze(axis=0)
        orig_size = 24
        new_size = 16
        if pos_embed_checkpoint.shape[1] == new_size**2 + 1:
            self.is_resize_pos = True
        else:
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_extra_tokens = 1
            new_num = new_size**2 + num_extra_tokens
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape([-1, orig_size, orig_size, embedding_size]).transpose(perm=[0, 3, 1, 2])
            pos_tokens = paddle.nn.functional.interpolate(
                x=pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
            )
            pos_tokens = pos_tokens.transpose(perm=[0, 2, 3, 1]).flatten(start_axis=1, stop_axis=2)
            new_pos_embed = paddle.concat(x=(extra_tokens, pos_tokens), axis=1)
            new_pos_embed = new_pos_embed.squeeze(axis=0)
            self.vision_tower.vision_model.positional_embedding = paddle.nn.Embedding(
                num_embeddings=new_num, embedding_dim=1024
            )
            out_1 = paddle.create_parameter(
                shape=new_pos_embed.to(pos_embed_checkpoint.dtype).shape,
                dtype=new_pos_embed.to(pos_embed_checkpoint.dtype).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(new_pos_embed.to(pos_embed_checkpoint.dtype)),
            )
            out_1.stop_gradient = not True
            self.vision_tower.vision_model.positional_embedding.weight = out_1
            self.vision_tower.vision_model.position_ids = paddle.arange(end=new_num).expand(shape=(1, -1))
            self.is_resize_pos = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward(self, images):
        if not self.is_loaded:
            self.load_model()
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.unsqueeze(axis=0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images, output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs)
        return image_features

    @property
    def dummy_feature(self):
        return paddle.zeros(shape=[1, self.hidden_size])

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class PLoRA(paddle.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_len=0,
        **kwargs
    ) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias_attr=bias)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_len = lora_len
        if lora_dropout > 0.0:
            self.lora_dropout = paddle.nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.lora_scaling = self.lora_alpha / self.lora_r
        self.Plora_A = paddle.nn.Linear(in_features=in_features, out_features=self.lora_r, bias_attr=False)
        self.Plora_B = paddle.nn.Linear(in_features=self.lora_r, out_features=out_features, bias_attr=False)
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            init_KaimingUniform = paddle.nn.initializer.KaimingUniform(
                negative_slope=math.sqrt(5), nonlinearity="leaky_relu"
            )
            init_KaimingUniform(self.lora_A.weight)
            init_Constant = paddle.nn.initializer.Constant(value=0.0)
            init_Constant(self.lora_B.weight)

    def forward(self, x, im_mask=None):
        res = super().forward(x)
        if im_mask is not None:
            if paddle.sum(x=im_mask) > 0:
                part_x = x[im_mask]
                res[im_mask] += self.Plora_B(self.Plora_A(self.lora_dropout(part_x))) * self.lora_scaling
            else:
                part_x = x[:, :1]
                res[:, :1] += self.Plora_B(self.Plora_A(self.lora_dropout(part_x))) * 0
        return res


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def _make_causal_mask(input_ids_shape: list, dtype: paddle.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = paddle.full(shape=(tgt_len, tgt_len), fill_value=paddle.finfo(dtype).min)
    mask_cond = paddle.arange(end=mask.shape[-1])

    mask = masked_fill(mask, mask_cond < (mask_cond + 1).reshape([mask.shape[-1], 1]), 0)
    mask = mask.astype(dtype)
    if past_key_values_length > 0:
        mask = paddle.concat(x=[paddle.zeros(shape=[tgt_len, past_key_values_length], dtype=dtype), mask], axis=-1)
    return mask[(None), (None), :, :].expand(shape=[bsz, 1, tgt_len, tgt_len + past_key_values_length])


def _expand_mask(mask: paddle.Tensor, dtype: paddle.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, (None), (None), :].expand(shape=[bsz, 1, tgt_len, src_len]).astype(dtype)
    inverted_mask = 1.0 - expanded_mask

    return masked_fill(inverted_mask, inverted_mask.astype("bool"), paddle.finfo(dtype).min)


class InternLM2RMSNorm(paddle.nn.Layer):
    def __init__(self, hidden_size, eps=1e-06):
        """InternLM2RMSNorm is equivalent to T5LayerNorm."""
        super().__init__()
        out_2 = paddle.create_parameter(
            shape=paddle.ones(shape=hidden_size).shape,
            dtype=paddle.ones(shape=hidden_size).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=hidden_size)),
        )
        out_2.stop_gradient = not True
        self.weight = out_2
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states
        variance = hidden_states.pow(y=2).mean(axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(x=variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class InternLM2RotaryEmbedding(paddle.nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / self.base ** (paddle.arange(start=0, end=self.dim, step=2).astype(dtype="float32") / self.dim)
        self.register_buffer(name="inv_freq", tensor=inv_freq, persistable=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device="cuda", dtype=paddle.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(dtype=self.inv_freq.dtype, end=self.max_seq_len_cached)
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        emb = paddle.concat(x=(freqs, freqs), axis=-1)
        self.register_buffer(name="cos_cached", tensor=emb.cos().to(dtype), persistable=False)
        self.register_buffer(name="sin_cached", tensor=emb.sin().to(dtype), persistable=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device="cuda", dtype=x.dtype)
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)


class InternLM2LinearScalingRotaryEmbedding(InternLM2RotaryEmbedding):
    """InternLM2RotaryEmbedding extended with linear scaling.

    Credits to the Reddit user /u/kaiokendev
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(dtype=self.inv_freq.dtype, end=self.max_seq_len_cached)
        t = t / self.scaling_factor
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        emb = paddle.concat(x=(freqs, freqs), axis=-1)
        self.register_buffer(name="cos_cached", tensor=emb.cos().to(dtype), persistable=False)
        self.register_buffer(name="sin_cached", tensor=emb.sin().to(dtype), persistable=False)


class InternLM2DynamicNTKScalingRotaryEmbedding(InternLM2RotaryEmbedding):
    """InternLM2RotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla.
    """

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                self.scaling_factor * seq_len / self.max_position_embeddings - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / base ** (paddle.arange(start=0, end=self.dim, step=2) / self.dim)
            self.register_buffer(name="inv_freq", tensor=inv_freq, persistable=False)
        t = paddle.arange(dtype=self.inv_freq.dtype, end=self.max_seq_len_cached)
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        emb = paddle.concat(x=(freqs, freqs), axis=-1)
        self.register_buffer(name="cos_cached", tensor=emb.cos().to(dtype), persistable=False)
        self.register_buffer(name="sin_cached", tensor=emb.sin().to(dtype), persistable=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat(x=(-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(axis=1).squeeze(axis=0)
    sin = sin.squeeze(axis=1).squeeze(axis=0)
    cos = cos.unsqueeze(axis=0).unsqueeze(axis=0).expand(shape=[len(position_ids), -1, -1, -1])
    sin = sin.unsqueeze(axis=0).unsqueeze(axis=0).expand(shape=[len(position_ids), -1, -1, -1])
    if q.shape[2] == 1:
        q_embed = q * cos[:, :, -1:, :] + rotate_half(q) * sin[:, :, -1:, :]
    else:
        q_embed = q * cos + rotate_half(q) * sin
    if k.shape[2] == 1:
        k_embed = k * cos[:, :, -1:, :] + rotate_half(k) * sin[:, :, -1:, :]
    else:
        k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class InternLM2MLP(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.w1 = PLoRA(self.hidden_size, self.intermediate_size, bias=False, lora_r=256, lora_alpha=256, lora_len=576)
        self.w3 = PLoRA(self.hidden_size, self.intermediate_size, bias=False, lora_r=256, lora_alpha=256, lora_len=576)
        self.w2 = PLoRA(self.intermediate_size, self.hidden_size, bias=False, lora_r=256, lora_alpha=256, lora_len=576)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, im_mask):
        down_proj = self.w2(self.act_fn(self.w1(x, im_mask)) * self.w3(x, im_mask), im_mask)
        return down_proj


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(shape=[batch, num_key_value_heads, n_rep, slen, head_dim])
    return hidden_states.reshape([batch, num_key_value_heads * n_rep, slen, head_dim])


class InternLM2Attention(paddle.nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, config: InternLM2Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.is_causal = True
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads})."
            )
        self.wqkv = PLoRA(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.bias,
            lora_r=256,
            lora_alpha=256,
            lora_len=576,
        )
        self.wo = PLoRA(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.bias,
            lora_r=256,
            lora_alpha=256,
            lora_len=576,
        )
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = InternLM2RotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.config.rope_theta
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "dynamic":
                self.rotary_emb = InternLM2DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.config.rope_theta,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError("Currently we only support rotary embedding's type being 'dynamic'.")
        return self.rotary_emb

    def _shape(self, tensor: paddle.Tensor, seq_len: int, bsz: int):
        x = tensor.reshape([bsz, seq_len, self.num_heads, self.head_dim])
        perm_0 = list(range(x.ndim))
        perm_0[1] = 2
        perm_0[2] = 1
        return x.transpose(perm=perm_0)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        im_mask: Optional[Tuple[paddle.Tensor]] = None,
        **kwargs
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.shape
        qkv_states = self.wqkv(hidden_states, im_mask)
        qkv_states = rearrange(
            qkv_states, "b q (h gs d) -> b q h gs d", gs=2 + self.num_key_value_groups, d=self.head_dim
        )
        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]
        x = query_states
        perm_1 = list(range(x.ndim))
        perm_1[1] = 2
        perm_1[2] = 1
        query_states = x.transpose(perm=perm_1)
        x = key_states
        perm_2 = list(range(x.ndim))
        perm_2[1] = 2
        perm_2[2] = 1
        key_states = x.transpose(perm=perm_2)
        x = value_states
        perm_3 = list(range(x.ndim))
        perm_3[1] = 2
        perm_3[2] = 1
        value_states = x.transpose(perm=perm_3)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            key_states = paddle.concat(x=[past_key_value[0], key_states], axis=2)
            value_states = paddle.concat(x=[past_key_value[1], value_states], axis=2)
        past_key_value = (key_states, value_states) if use_cache else None
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        x = key_states
        perm_4 = list(range(x.ndim))
        perm_4[2] = 3
        perm_4[3] = 2
        attn_weights = paddle.matmul(x=query_states, y=x.transpose(perm=perm_4)) / math.sqrt(self.head_dim)
        if attn_weights.shape != [bsz, self.num_heads, q_len, kv_seq_len]:
            raise ValueError(
                f"Attention weights should be of size {bsz, self.num_heads, q_len, kv_seq_len}, but is {attn_weights.shape}"
            )
        if attention_mask is not None:
            if attention_mask.shape != [bsz, 1, q_len, kv_seq_len]:
                raise ValueError(
                    f"Attention mask should be of size {bsz, 1, q_len, kv_seq_len}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights + attention_mask
        attn_weights = paddle.nn.functional.softmax(x=attn_weights, axis=-1).to(query_states.dtype)
        attn_output = paddle.matmul(x=attn_weights, y=value_states)
        if attn_output.shape != [bsz, self.num_heads, q_len, self.head_dim]:
            raise ValueError(
                f"`attn_output` should be of size {bsz, self.num_heads, q_len, self.head_dim}, but is {attn_output.shape}"
            )
        x = attn_output
        perm_5 = list(range(x.ndim))
        perm_5[1] = 2
        perm_5[2] = 1
        attn_output = x.transpose(perm=perm_5)
        attn_output = attn_output.reshape([bsz, q_len, self.hidden_size])
        attn_output = self.wo(attn_output, im_mask)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class InternLM2FlashAttention2(InternLM2Attention):
    """InternLM2 flash attention module.

    This module inherits from `InternLM2Attention` as the weights of the module
    stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal
    with padding tokens in case the input contains any of them.
    """

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        im_mask: Optional[Tuple[paddle.Tensor]] = None,
        **kwargs
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
            attention_mask = kwargs.pop("padding_mask")
        output_attentions = False
        bsz, q_len, _ = hidden_states.shape
        qkv_states = self.wqkv(hidden_states, im_mask)
        qkv_states = rearrange(
            qkv_states,
            "b q (h gs d) -> b q h gs d",
            gs=self.num_heads + 2 * self.num_key_value_heads,
            d=self.head_dim,
            q=q_len,
        )
        query_states = qkv_states[..., : self.num_key_value_groups, :]
        query_states = rearrange(query_states, "b q h gs d -> b q (h gs) d")
        key_states = qkv_states[..., -2, :]
        value_states = qkv_states[..., -1, :]
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            key_states = paddle.concat(x=[past_key_value[0], key_states], axis=2)
            value_states = paddle.concat(x=[past_key_value[1], value_states], axis=2)
        past_key_value = (key_states, value_states) if use_cache else None
        x = query_states
        perm_6 = list(range(x.ndim))
        perm_6[1] = 2
        perm_6[2] = 1
        query_states = x.transpose(perm=perm_6)
        x = key_states
        perm_7 = list(range(x.ndim))
        perm_7[1] = 2
        perm_7[2] = 1
        key_states = x.transpose(perm=perm_7)
        x = value_states
        perm_8 = list(range(x.ndim))
        perm_8[1] = 2
        perm_8[2] = 1
        value_states = x.transpose(perm=perm_8)
        dropout_rate = 0.0 if not self.training else self.attention_dropout
        input_dtype = query_states.dtype
        if input_dtype == "float32":
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )
        attn_output = attn_output.reshape([bsz, q_len, self.hidden_size])
        attn_output = self.wo(attn_output, im_mask)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class InternLM2DecoderLayer(paddle.nn.Layer):
    def __init__(self, config: InternLM2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention = (
            InternLM2Attention(config=config)
            if not getattr(config, "_flash_attn_2_enabled", False)
            else InternLM2FlashAttention2(config=config)
        )
        self.feed_forward = InternLM2MLP(config)
        self.attention_norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        im_mask: Optional[Tuple[paddle.Tensor]] = None,
        **kwargs
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            im_mask=im_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states, im_mask)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class InternLM2PretrainedModel(PretrainedModel):
    config_class = InternLM2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["InternLM2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True


class InternLM2Model(InternLM2PretrainedModel):
    """Transformer decoder consisting of *config.num_hidden_layers* layers.
    Each layer is a [`InternLM2DecoderLayer`]

    Args:
        config: InternLM2Config
    """

    _auto_class = "AutoModel"

    def __init__(self, config: InternLM2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.tok_embeddings = paddle.nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.hidden_size)
        self.layers = paddle.nn.LayerList(
            sublayers=[InternLM2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = InternLM2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.tok_embeddings

    def set_input_embeddings(self, value):
        self.tok_embeddings = value

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
        **kwargs
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        im_mask = kwargs.get("im_mask", None)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            position_ids = paddle.arange(
                start=past_key_values_length, end=seq_length + past_key_values_length, dtype="int64"
            )
            position_ids = position_ids.unsqueeze(axis=0)
        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)
            im_mask = paddle.zeros(shape=inputs_embeds.shape[:2]).astype(dtype="bool")
        if attention_mask is None:
            attention_mask = paddle.ones(shape=(batch_size, seq_length_with_past), dtype="bool")
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        hidden_states = inputs_embeds
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
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
                im_mask=im_mask,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class InternLMXComposer2ForCausalLM(InternLM2PretrainedModel):
    _auto_class = "AutoModelForCausalLM"
    _tied_weights_keys = ["output.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = InternLM2Model(config)
        self.vocab_size = config.vocab_size
        self.output = paddle.nn.Linear(in_features=config.hidden_size, out_features=config.vocab_size, bias_attr=False)
        self.tokenizer = None
        self.max_length = config.max_length
        print(f"Set max length to {self.max_length}")
        self.vit = build_vision_tower(config)
        self.vision_proj = build_vision_projector(config.vision_projector)
        self.vis_processor = paddle.vision.transforms.Compose(
            [
                paddle.vision.transforms.Resize((config.img_size, config.img_size), interpolation="bicubic"),
                paddle.vision.transforms.ToTensor(),
                paddle.vision.transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                ),
            ]
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, InternLM2Model):
            module.gradient_checkpointing = value
        if value:
            (self.vit.vision_tower.vision_model.encoder.gradient_checkpointing) = value

    def get_input_embeddings(self):
        return self.model.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.tok_embeddings = value

    def get_output_embeddings(self):
        return self.output

    def set_output_embeddings(self, new_embeddings):
        self.output = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def encode_text(self, text, add_special_tokens=False):
        token = self.tokenizer(text, return_tensors="pd", add_special_tokens=add_special_tokens).input_ids
        embs = self.model.tok_embeddings(token)
        return embs

    def encode_img(self, image):
        if image is None:
            return None
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
            image = self.vis_processor(image).unsqueeze(axis=0)
        else:
            assert isinstance(image, paddle.Tensor)
        img_embeds, atts_img, img_target = self.img2emb(image)
        return img_embeds

    def img2emb(self, image):
        img_embeds = self.vision_proj(self.vit(image))
        atts_img = paddle.ones(shape=img_embeds.shape[:-1], dtype="int64")
        img_target = paddle.ones(shape=img_embeds.shape[:2], dtype="int64") * -100
        return img_embeds, atts_img, img_target

    def prompt_wrap(self, img_embeds, prompt):
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split("<ImageHere>")
        p_before_tokens = self.tokenizer(p_before, return_tensors="pd", add_special_tokens=True)
        p_before_embeds = self.model.tok_embeddings(p_before_tokens.input_ids).expand(shape=[batch_size, -1, -1])
        wrapped_img_embeds = paddle.concat(x=[p_before_embeds, img_embeds], axis=1)
        wrapped_atts_img = paddle.ones(shape=wrapped_img_embeds.shape[:-1], dtype="int64")
        wrapped_target = paddle.ones(shape=[batch_size, wrapped_img_embeds.shape[1]], dtype="int64") * -100
        return wrapped_img_embeds, wrapped_atts_img, wrapped_target

    def text2emb(self, to_regress_tokens, add_special=False):
        targets = self.mask_human_targets(to_regress_tokens["input_ids"])
        return to_regress_tokens, targets

    def interleav_wrap_chat(self, tokenizer, query, image, history, meta_instruction):
        prompt = ""
        if meta_instruction:
            prompt += f"""[UNUSED_TOKEN_146]system\n{meta_instruction}[UNUSED_TOKEN_145]\n"""
        for record in history:
            prompt += f"""[UNUSED_TOKEN_146]user\n{record[0]}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{record[1]}[UNUSED_TOKEN_145]\n"""
        prompt += f"""[UNUSED_TOKEN_146]user\n{query}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""

        im_len = image.shape[1]
        image_nums = len(image)
        parts = prompt.split("<ImageHere>")
        wrap_embeds, wrap_im_mask = [], []
        temp_len = 0
        if len(parts) != image_nums + 1:
            raise ValueError("Invalid <ImageHere> prompt format.")
        for idx, part in enumerate(parts):
            if len(part) > 0:
                part_tokens = tokenizer(part, return_tensors="pd")
                part_embeds = self.model.tok_embeddings(part_tokens.input_ids)
                wrap_embeds.append(part_embeds)
                wrap_im_mask.append(paddle.zeros(shape=part_embeds.shape[:2]))
                temp_len += part_embeds.shape[1]
            if idx < image_nums:
                wrap_embeds.append(image[idx].unsqueeze(axis=0))
                wrap_im_mask.append(paddle.ones(shape=[1, image[idx].shape[0]]))
                temp_len += im_len
            if temp_len > self.max_length:
                break
        wrap_embeds = paddle.concat(x=wrap_embeds, axis=1)
        wrap_im_mask = paddle.concat(x=wrap_im_mask, axis=1)
        wrap_embeds = wrap_embeds[:, : self.max_length]
        wrap_im_mask = wrap_im_mask[:, : self.max_length].astype(dtype="bool")
        inputs = {"inputs_embeds": wrap_embeds}
        return inputs, wrap_im_mask

    def interleav_wrap(self, img_list, text_tokens_list):
        wrap_embeds_list, wrap_atts_list = [], []
        wrap_target_list, wrap_im_mask_list = [], []
        for image, text_tokens in zip(img_list, text_tokens_list):
            img_embeds, atts_img, img_target = self.img2emb(image)
            wrap_tokens, wrap_embeds, wrap_atts, wrap_im_mask = [], [], [], []
            temp_len = 0
            image_nums, im_len = img_embeds.shape[:2]
            need_bos = True
            for idx, part_tokens in enumerate(text_tokens):
                if len(part_tokens) > 0:
                    # part_tokens = self.tokenizer(
                    #     part,
                    #     return_tensors='pd',
                    #     padding='longest',
                    #     add_special_tokens=need_bos)
                    if need_bos:
                        need_bos = False
                    wrap_tokens.append(part_tokens["input_ids"])
                    part_embeds = self.model.tok_embeddings(part_tokens["input_ids"])
                    wrap_embeds.append(part_embeds)
                    wrap_atts.append(part_tokens["attention_mask"])
                    wrap_im_mask.append(paddle.zeros(shape=part_embeds.shape[:2]).to("float32"))
                    temp_len += part_embeds.shape[1]
                if idx < image_nums:
                    wrap_tokens.append(img_target[idx].unsqueeze(axis=0))
                    wrap_embeds.append(img_embeds[idx].unsqueeze(axis=0))
                    wrap_atts.append(atts_img[idx].unsqueeze(axis=0))
                    wrap_im_mask.append(paddle.ones_like(x=atts_img[idx].unsqueeze(axis=0)).to("float32"))
                    temp_len += im_len
                if temp_len > self.max_length:
                    break
            wrap_tokens = paddle.concat(x=wrap_tokens, axis=1)
            wrap_embeds = paddle.concat(x=wrap_embeds, axis=1)
            wrap_atts = paddle.concat(x=wrap_atts, axis=1)
            wrap_im_mask = paddle.concat(x=wrap_im_mask, axis=1)
            wrap_target = self.mask_human_targets(wrap_tokens)
            wrap_embeds = wrap_embeds[:, : self.max_length]
            wrap_atts = wrap_atts[:, : self.max_length]
            wrap_target = wrap_target[:, : self.max_length]
            wrap_im_mask = wrap_im_mask[:, : self.max_length]
            wrap_embeds_list.append(wrap_embeds)
            wrap_atts_list.append(wrap_atts)
            wrap_target_list.append(wrap_target)
            wrap_im_mask_list.append(wrap_im_mask)
        wrap_embeds = paddle.concat(x=wrap_embeds_list)
        wrap_atts = paddle.concat(x=wrap_atts_list)
        wrap_target = paddle.concat(x=wrap_target_list)
        wrap_im_mask = paddle.concat(x=wrap_im_mask_list)
        return wrap_embeds, wrap_atts, wrap_target, wrap_im_mask

    def mask_human_targets(self, input_ids, pure=False):
        target_batch = []
        for bs in range(input_ids.shape[0]):
            ids = input_ids[bs]
            targets = copy.deepcopy(ids)
            end_count = 0
            last_eoa = 0
            for i, temp_id in enumerate(ids):
                if temp_id == 92542:
                    if end_count % 2 == 0:
                        targets[last_eoa : i + 6] = -100
                    else:
                        last_eoa = i + 1
                    end_count += 1
                elif temp_id == 2:
                    targets[i + 1 :] = -100
                    break
            if temp_id != 2 and end_count % 2 == 0:
                targets[last_eoa + 1 :] = -100
            target_batch.append(targets.unsqueeze(axis=0))
        target_batch = paddle.concat(x=target_batch, axis=0)
        return target_batch

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
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Args:
            labels (`paddle.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        samples = kwargs.get("samples", None)
        if samples:
            if "images" in samples:
                has_img = True
            else:
                has_img = False
            input_tokens = samples["input_tokens"]

            if has_img:
                image = samples["images"]
                to_regress_embeds, attention_mask, targets, im_mask = self.interleav_wrap(image, input_tokens)
            else:
                input_tokens = {  # [{'input_ids': xxx}, {'input_ids': yyy}] to {input_ids: [xxx, yyy]}
                    "input_ids": paddle.concat([tokens["input_ids"] for tokens in input_tokens]),
                    "attention_mask": paddle.concat([tokens["attention_mask"] for tokens in input_tokens]),
                }
                to_regress_tokens, targets = self.text2emb(input_tokens, add_special=True)
                to_regress_embeds = self.model.tok_embeddings(to_regress_tokens["input_ids"])
                attention_mask = to_regress_tokens["attention_mask"]
                im_mask = paddle.zeros(shape=to_regress_embeds.shape[:2])
            inputs_embeds = to_regress_embeds[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]
            targets = targets[:, : self.max_length]
            im_mask = im_mask[:, : self.max_length].astype(dtype="bool")
            labels = targets
        else:
            im_mask = kwargs.get("im_mask", None)
            if im_mask is None and inputs_embeds is not None:
                im_mask = paddle.zeros(shape=inputs_embeds.shape[:2])
                im_mask = im_mask.astype(dtype="bool")
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
            im_mask=im_mask,
        )
        hidden_states = outputs[0]
        logits = self.output(hidden_states)
        logits = logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss_fct = paddle.nn.CrossEntropyLoss()
            shift_logits = shift_logits.reshape([-1, self.config.vocab_size])
            shift_labels = shift_labels.reshape([-1])
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

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, im_mask=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.astype(dtype="int64").cumsum(axis=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        im_mask = im_mask
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "im_mask": im_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(axis=0, index=beam_idx) for past_state in layer_past),)
        return reordered_past

    @paddle.no_grad()
    def generate(self, **kwargs):
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if attention_mask is None:
            attention_mask = paddle.ones(inputs_embeds.shape[:2], dtype="int64")
            batch_size, seq_length = attention_mask.shape

        return super().generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

    @staticmethod
    def prepare_input_ids_for_generation(bos_token_id, encoder_output=None):
        batch_size = 1
        if bos_token_id is None:
            raise ValueError("`bos_token_id` should be defined when no " "`input_ids` are provided.")
        if encoder_output is not None:
            batch_size = encoder_output.shape[0]
        return paddle.ones([batch_size, 1], dtype="int64") * bos_token_id

    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = [], meta_instruction=""):
        prompt = ""
        if meta_instruction:
            prompt += f"""<s>[UNUSED_TOKEN_146]system\n{meta_instruction}[UNUSED_TOKEN_145]\n"""
        else:
            prompt += "<s>"
        for record in history:
            prompt += f"""[UNUSED_TOKEN_146]user\n{record[0]}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n{record[1]}[UNUSED_TOKEN_145]\n"""
        prompt += f"""[UNUSED_TOKEN_146]user\n{query}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n"""
        return tokenizer([prompt], return_tensors="pd")

    @paddle.no_grad()
    def chat(
        self,
        tokenizer,
        query: str,
        image: paddle.Tensor = None,
        history: List[Tuple[str, str]] = [],
        streamer: Optional[BaseStreamer] = None,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.8,
        repetition_penalty: float = 1.005,
        meta_instruction: str = "You are an AI assistant whose name is InternLM-XComposer (浦语·灵笔).\n"
        "- InternLM-XComposer (浦语·灵笔) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM-XComposer (浦语·灵笔) can understand and communicate fluently in the language chosen by the user such as English and 中文.",
        **kwargs
    ):
        if image is None:
            inputs = self.build_inputs(tokenizer, query, history, meta_instruction)
            im_mask = paddle.zeros(shape=inputs["input_ids"].shape[:2]).astype(dtype="bool")
        else:
            image = self.encode_img(image)
            inputs, im_mask = self.interleav_wrap_chat(tokenizer, query, image, history, meta_instruction)
        inputs = {k: v for k, v in inputs.items() if paddle.is_tensor(x=v)}
        eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["[UNUSED_TOKEN_145]"])[0]]
        outputs, _ = self.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            im_mask=im_mask,
            **kwargs,
        )
        if image is None:
            outputs = outputs[0].cpu().tolist()[len(inputs["input_ids"][0]) :]
        else:
            outputs = outputs[0].cpu().tolist()
        response = tokenizer.decode(outputs, skip_special_tokens=True)
        response = response.split("[UNUSED_TOKEN_145]")[0]
        history = history + [(query, response)]
        return response, history

    @paddle.no_grad()
    def stream_chat(
        self,
        tokenizer,
        query: str,
        history: List[Tuple[str, str]] = [],
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.8,
        **kwargs
    ):
        """Return a generator in format: (response, history) Eg.

        ('你好，有什么可以帮助您的吗', [('你好', '你好，有什么可以帮助您的吗')]) ('你好，有什么可以帮助您的吗？', [('你好',
        '你好，有什么可以帮助您的吗？')])
        """
        if BaseStreamer is None:
            raise ModuleNotFoundError(
                "The version of `transformers` is too low. Please make sure that you have installed `transformers>=4.28.0`."
            )
        response_queue = queue.Queue(maxsize=20)

        class ChatStreamer(transformers.generation.streamers.BaseStreamer):
            def __init__(self, tokenizer) -> None:
                super().__init__()
                self.tokenizer = tokenizer
                self.queue = response_queue
                self.query = query
                self.history = history
                self.response = ""
                self.received_inputs = False
                self.queue.put((self.response, history + [(self.query, self.response)]))

            def put(self, value):
                if len(value.shape) > 1 and value.shape[0] > 1:
                    raise ValueError("ChatStreamer only supports batch size 1")
                elif len(value.shape) > 1:
                    value = value[0]
                if not self.received_inputs:
                    self.received_inputs = True
                    return
                token = self.tokenizer.decode([value[-1]], skip_special_tokens=True)
                if token.strip() != "[UNUSED_TOKEN_145]":
                    self.response = self.response + token
                    history = self.history + [(self.query, self.response)]
                    self.queue.put((self.response, history))

            def end(self):
                self.queue.put(None)

        def stream_producer():
            return self.chat(
                tokenizer=tokenizer,
                query=query,
                streamer=ChatStreamer(tokenizer=tokenizer),
                history=history,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

        def consumer():
            producer = threading.Thread(target=stream_producer)
            producer.start()
            while True:
                res = response_queue.get()
                if res is None:
                    return
                yield res

        return consumer()
