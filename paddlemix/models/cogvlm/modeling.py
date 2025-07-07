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

import paddle

"""largely copy from llama and adapt for CogAgent"""
import math
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from einops import rearrange
from paddlenlp import transformers
from paddlenlp.transformers.activations import ACT2FN
from paddlenlp.transformers.model_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from paddlenlp.transformers.model_utils import PretrainedModel

from .configuration import CogModelConfig
from .visual import CrossVisionModel, EVA2CLIPModel

if TYPE_CHECKING:
    logger = transformers.utils.logging.get_logger(__name__)
LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def _make_causal_mask(input_ids_shape: list, dtype: paddle.dtype, device: str, past_key_values_length: int = 0):
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


class RMSNorm(paddle.nn.Layer):
    def __init__(self, hidden_size, eps=1e-06):
        super().__init__()
        out_10 = paddle.create_parameter(
            shape=paddle.ones(shape=hidden_size).shape,
            dtype=paddle.ones(shape=hidden_size).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=hidden_size)),
        )
        out_10.stop_gradient = not True
        self.weight = out_10
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to("float32")
        variance = hidden_states.pow(y=2).mean(axis=-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(x=variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class MLP(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.intermediate_size, bias_attr=False
        )
        self.up_proj = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.intermediate_size, bias_attr=False
        )
        self.down_proj = paddle.nn.Linear(
            in_features=self.intermediate_size, out_features=self.hidden_size, bias_attr=False
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if x.shape[0] == 0:
            return x
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def get_expert_mask(token_type_ids):
    vision_token_mask = paddle.zeros_like(x=token_type_ids, dtype="bool")
    if vision_token_mask.shape[1] != 1:  # to avoid error
        vision_token_mask[:, :-1] = (token_type_ids[:, :-1] == VISION_TOKEN_TYPE) & (
            token_type_ids[:, 1:] == VISION_TOKEN_TYPE
        )
    language_token_mask = ~vision_token_mask
    return vision_token_mask, language_token_mask


class VisionExpertMLP(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.language_mlp = MLP(config)
        self.vision_mlp = MLP(config)

    def forward(self, hidden_states, token_type_ids):
        output = paddle.empty(shape=hidden_states.shape, dtype=hidden_states.dtype)
        vision_token_mask, language_token_mask = get_expert_mask(token_type_ids)
        output[vision_token_mask] = self.vision_mlp(hidden_states[vision_token_mask])
        output[language_token_mask] = self.language_mlp(hidden_states[language_token_mask])
        return output


def attention_fn(
    query_layer,
    key_layer,
    value_layer,
    attention_mask,
    *,
    scaling_attention_score: bool = True,
    attention_dropout: paddle.nn.Layer = None
):
    attention_mask == 0

    if scaling_attention_score:
        query_layer = query_layer / math.sqrt(query_layer.shape[-1])
    x = key_layer
    perm_3 = list(range(x.ndim))
    perm_3[-1] = -2
    perm_3[-2] = -1
    attention_scores = paddle.matmul(x=query_layer, y=x.transpose(perm=perm_3))
    attention_scores = attention_scores + attention_mask.astype(attention_scores.dtype)
    attention_scores = paddle.nn.functional.softmax(x=attention_scores, axis=-1, dtype="float32").to(query_layer.dtype)
    if attention_dropout is not None:
        attention_scores = attention_dropout(attention_scores)
    context_layer = paddle.matmul(x=attention_scores, y=value_layer)
    return context_layer


class RotaryEmbedding(paddle.nn.Layer):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer(name="inv_freq", tensor=inv_freq)
        self.max_seq_len_cached = 0

    def _compute_inv_freq(self, device=None):
        return 1.0 / self.base ** (paddle.arange(start=0, end=self.dim, step=2) / self.dim)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = paddle.arange(dtype=self.inv_freq.dtype, end=self.max_seq_len_cached)
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        emb = paddle.concat(x=(freqs, freqs), axis=-1)
        self.register_buffer(name="cos_cached", tensor=emb.cos()[:, None, :].to(dtype), persistable=False)
        self.register_buffer(name="sin_cached", tensor=emb.sin()[:, None, :].to(dtype), persistable=False)

    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.place, dtype=x.dtype)
        return self.cos_cached[:seq_len, ...].to("float32"), self.sin_cached[:seq_len, ...].to(dtype="float32")


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return paddle.concat(x=(-x2, x1), axis=x1.ndim - 1)


def apply_rotary_pos_emb_index_bhs(q, k, cos, sin, position_id):
    input_dtype = q.dtype
    cos, sin = paddle.nn.functional.embedding(x=position_id, weight=cos.squeeze(axis=1)).unsqueeze(
        axis=1
    ), paddle.nn.functional.embedding(x=position_id, weight=sin.squeeze(axis=1)).unsqueeze(axis=1)
    q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
    q = q.astype(input_dtype)
    k = k.astype(input_dtype)
    return q, k


class VisionExpertAttention(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.vision_expert_query_key_value = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size * 3, bias_attr=False
        )
        self.vision_expert_dense = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size, bias_attr=False
        )
        self.language_expert_query_key_value = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size * 3, bias_attr=False
        )
        self.language_expert_dense = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.hidden_size, bias_attr=False
        )

    def _transpose_for_scores(self, tensor):
        new_tensor_shape = tuple(tensor.shape[:-1]) + (self.num_heads, self.head_dim)
        tensor = tensor.reshape(new_tensor_shape)
        return tensor.transpose(perm=[0, 2, 1, 3])

    def forward(
        self,
        hidden_states: paddle.Tensor,
        token_type_ids: paddle.Tensor,
        position_ids: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape
        vision_token_mask, language_token_mask = get_expert_mask(token_type_ids)
        shape = list(hidden_states.shape)
        shape[-1] = shape[-1] * 3
        mixed_raw_layer = paddle.empty(shape=shape, dtype=hidden_states.dtype)
        if hidden_states[vision_token_mask].shape[0] != 0:
            mixed_raw_layer[vision_token_mask] = self.vision_expert_query_key_value(hidden_states[vision_token_mask])
        if hidden_states[language_token_mask].shape[0] != 0:
            mixed_raw_layer[language_token_mask] = self.language_expert_query_key_value(
                hidden_states[language_token_mask]
            )
        query_states, key_states, value_states = paddle.split(x=mixed_raw_layer, num_or_sections=3, axis=-1)
        query_states = self._transpose_for_scores(query_states)
        key_states = self._transpose_for_scores(key_states)
        value_states = self._transpose_for_scores(value_states)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max() + 1)
        query_states, key_states = apply_rotary_pos_emb_index_bhs(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            key_states = paddle.concat(x=[past_key_value[0], key_states], axis=2)
            value_states = paddle.concat(x=[past_key_value[1], value_states], axis=2)
        past_key_value = (key_states, value_states) if use_cache else None
        context_layer = attention_fn(
            query_layer=query_states,
            key_layer=key_states,
            value_layer=value_states,
            attention_mask=attention_mask,
            scaling_attention_score=True,
            attention_dropout=None,
        )
        if context_layer.shape != [bsz, self.num_heads, q_len, self.head_dim]:
            raise ValueError(
                f"`attn_output` should be of size {bsz, self.num_heads, q_len, self.head_dim}, but is {context_layer.shape}"
            )
        x = context_layer
        perm_4 = list(range(x.ndim))
        perm_4[1] = 2
        perm_4[2] = 1
        context_layer = x.transpose(perm=perm_4).reshape([bsz, q_len, self.hidden_size])
        attn_output = paddle.empty(shape=context_layer.shape, dtype=hidden_states.dtype)
        if context_layer[vision_token_mask].shape[0] != 0:
            attn_output[vision_token_mask] = self.vision_expert_dense(context_layer[vision_token_mask])
        if context_layer[language_token_mask].shape[0] != 0:
            attn_output[language_token_mask] = self.language_expert_dense(context_layer[language_token_mask])
        if output_attentions:
            warnings.warn("output_attentions is not implemented.")
        return attn_output, None, past_key_value


class CrossAttention(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.cross_hidden_size = config.cross_hidden_size
        self.cross_compute_hidden_size = config.cross_compute_hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.cross_head_dim = self.cross_compute_hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.query = paddle.nn.Linear(
            in_features=self.hidden_size, out_features=self.cross_compute_hidden_size, bias_attr=False
        )
        self.key_value = paddle.nn.Linear(
            in_features=self.cross_hidden_size, out_features=self.cross_compute_hidden_size * 2, bias_attr=False
        )
        self.dense = paddle.nn.Linear(
            in_features=self.cross_compute_hidden_size, out_features=self.hidden_size, bias_attr=False
        )

    def _transpose_for_scores(self, tensor):
        new_tensor_shape = tuple(tensor.shape[:-1]) + (self.num_heads, self.cross_head_dim)
        tensor = tensor.reshape(new_tensor_shape)
        return tensor.transpose(perm=[0, 2, 1, 3])

    def forward(
        self,
        hidden_states: paddle.Tensor,
        encoder_outputs: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape
        shape = list(hidden_states.shape)
        shape[-1] = shape[-1] * 3
        mixed_query_layer = self.query(hidden_states)
        if past_key_value is None:
            mixed_x_layer = self.key_value(encoder_outputs)
            mixed_key_layer, mixed_value_layer = paddle.split(x=mixed_x_layer, num_or_sections=2, axis=-1)
            key_states = self._transpose_for_scores(mixed_key_layer)
            value_states = self._transpose_for_scores(mixed_value_layer)
        else:
            key_states, value_states = past_key_value
        query_states = self._transpose_for_scores(mixed_query_layer)
        past_key_value = (key_states, value_states) if use_cache else None
        context_layer = attention_fn(
            query_layer=query_states,
            key_layer=key_states,
            value_layer=value_states,
            attention_mask=attention_mask,
            scaling_attention_score=True,
            attention_dropout=None,
        )
        if context_layer.shape != [bsz, self.num_heads, q_len, self.cross_head_dim]:
            raise ValueError(
                f"`cross_attn_output` should be of size {bsz, self.num_heads, q_len, self.cross_head_dim}, but is {context_layer.shape}"
            )
        x = context_layer
        perm_5 = list(range(x.ndim))
        perm_5[1] = 2
        perm_5[2] = 1
        context_layer = x.transpose(perm=perm_5).reshape([bsz, q_len, self.cross_hidden_size])
        attn_output = self.dense(context_layer)
        if output_attentions:
            warnings.warn("output_attentions is not implemented.")
        return attn_output, None, past_key_value


class CogModelDecoderLayer(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.model_type = config.model_type
        self.hidden_size = config.hidden_size
        self.self_attn = VisionExpertAttention(config=config)
        if self.model_type == "cogagent":
            self.cross_attn = CrossAttention(config=config)
            self.post_cross_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = VisionExpertMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        encoder_outputs: paddle.Tensor,
        token_type_ids: paddle.Tensor,
        position_ids: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        cross_attention_mask: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor, Optional[Tuple[paddle.Tensor, paddle.Tensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=None if past_key_value is None else past_key_value[:2],
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        data_type = hidden_states.dtype
        residual = residual.astype(data_type)
        hidden_states = residual + hidden_states
        if self.model_type == "cogagent":
            cross_input = self.post_cross_attention_layernorm(hidden_states)
            (attention_output, self_cross_attn_weights, present_cross_key_value) = self.cross_attn(
                hidden_states=cross_input,
                encoder_outputs=encoder_outputs,
                attention_mask=cross_attention_mask,
                past_key_value=past_key_value[-2:] if past_key_value is not None else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = hidden_states.astype(data_type)
            hidden_states = hidden_states + attention_output
            mlp_input = self.post_attention_layernorm(hidden_states)
            mlp_output = self.mlp(mlp_input, token_type_ids=token_type_ids)
            hidden_states = hidden_states.astype(data_type)
            hidden_states = mlp_output + hidden_states
        elif self.model_type == "cogvlm":
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states, token_type_ids=token_type_ids)
            data_type = hidden_states.dtype
            residual = residual.astype(data_type)
            hidden_states = residual + hidden_states
        else:
            raise ValueError("model_type in config must be cogagent or cogvlm, but got {}".format(self.model_type))
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            if self.model_type == "cogagent":
                outputs += (present_key_value + present_cross_key_value,)
            else:
                outputs += (present_key_value,)
        return outputs


class CogPreTrainedModel(PretrainedModel):
    config_class = CogModelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["CogModelDecoderLayer", "TransformerLayer", "Block"]
    _skip_keys_device_placement = "past_key_values"


def is_empty(images_list: Optional[List[List[paddle.Tensor]]]):
    if images_list is None or len(images_list) == 0:
        return True
    for image_list in images_list:
        if len(image_list):
            return False
    return True


def build_position_ids(x, attention_mask):
    if attention_mask is not None:
        tmp = x.clone()
        tmp[~attention_mask.astype(dtype="bool")] = -1
    else:
        tmp = x.clone()
    is_boi_eoi = paddle.zeros_like(x=x, dtype="bool")
    is_boi_eoi[:, 1:] |= (tmp[:, 1:] == VISION_TOKEN_TYPE) & (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, 0] |= tmp[:, 0] == VISION_TOKEN_TYPE
    is_boi_eoi[:, :-1] |= (tmp[:, :-1] == VISION_TOKEN_TYPE) & (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, -1] |= tmp[:, -1] == VISION_TOKEN_TYPE
    tmp[is_boi_eoi] = LANGUAGE_TOKEN_TYPE
    y = paddle.zeros_like(x=x, dtype="int64")
    y[:, 1:] = (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE) | (tmp[:, 1:] == VISION_TOKEN_TYPE) & (
        tmp[:, :-1] == LANGUAGE_TOKEN_TYPE
    )
    y = y.cumsum(axis=-1)
    return y


class CogModel(CogPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_type = config.model_type
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = paddle.nn.Embedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, padding_idx=self.padding_idx
        )
        self.layers = paddle.nn.LayerList(
            sublayers=[CogModelDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.vision = EVA2CLIPModel(config)
        self.cross_vision = CrossVisionModel(config)
        self.gradient_checkpointing = False

    def encode_images(self, images: List[List[paddle.Tensor]]) -> paddle.Tensor:
        images_list, images = images, []
        images = []
        for image_list in images_list:
            for image in image_list:
                images.append(image)
        images = paddle.stack(x=images)
        images_features = self.vision(images)
        return images_features

    def encode_cross_images(self, images: List[List[paddle.Tensor]]) -> paddle.Tensor:
        images_list, images = images, []
        images = []
        for image_list in images_list:
            for image in image_list:
                images.append(image)
        images = paddle.stack(x=images)
        encoder_outputs = self.cross_vision(images)
        return encoder_outputs

    def forward(
        self,
        input_ids: paddle.Tensor = None,
        images: List[List[paddle.Tensor]] = None,
        cross_images: List[List[paddle.Tensor]] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        cross_attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """take care of image_encode, token_type_ids, position_ids and (attention_mask = None is fine)"""
        if past_key_values is not None:
            if self.model_type == "cogagent" or self.model_type == "cogvlm":
                encoder_outputs = None
            else:
                raise ValueError("model_type in config must be cogagent or cogvlm, but got {}".format(self.model_type))
        else:
            assert input_ids is not None and inputs_embeds is None, f"{input_ids} {inputs_embeds}"
            if not is_empty(images):
                assert token_type_ids is not None, "multi-modality requires `token_type_ids`!"
                assert len(input_ids) == len(images), f"{len(input_ids)} {len(images)}"
                inputs_embeds = self.embed_tokens(input_ids)
                images_features = self.encode_images(images)
                encoder_outputs = None
                if self.model_type == "cogagent":
                    encoder_outputs = self.encode_cross_images(cross_images)
                elif self.model_type != "cogvlm":
                    raise ValueError(
                        "model_type in config must be cogagent or cogvlm, but got {}".format(self.model_type)
                    )
                images_features = rearrange(images_features, "b n d -> (b n) d")

                images_features = images_features.to(dtype=inputs_embeds.dtype, device=inputs_embeds.place)
                inputs_embeds = inputs_embeds.index_put([token_type_ids == VISION_TOKEN_TYPE], images_features)
            else:
                if token_type_ids is None:
                    token_type_ids = paddle.ones_like(x=input_ids, dtype="int64") * LANGUAGE_TOKEN_TYPE
                assert (
                    not (token_type_ids == VISION_TOKEN_TYPE).astype("bool").any()
                ), f"{(token_type_ids == VISION_TOKEN_TYPE).sum()}"
                inputs_embeds = self.embed_tokens(input_ids)
                encoder_outputs = None
            if position_ids is None:
                position_ids = build_position_ids(token_type_ids, attention_mask)
            input_ids = None

        return self.llm_forward(
            input_ids=input_ids,
            encoder_outputs=encoder_outputs,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            cross_attention_mask=cross_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def llm_forward(
        self,
        input_ids: paddle.Tensor = None,
        encoder_outputs: paddle.Tensor = None,
        token_type_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        cross_attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if position_ids is None:
            position_ids = paddle.arange(
                start=past_key_values_length, end=seq_length + past_key_values_length, dtype="int64"
            )
            position_ids = position_ids.unsqueeze(axis=0).reshape([-1, seq_length])
        else:
            position_ids = position_ids.reshape([-1, seq_length]).astype(dtype="int64")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = paddle.ones(shape=(batch_size, seq_length_with_past), dtype="bool")
        if self.model_type == "cogagent" and cross_attention_mask is None:
            cross_attention_mask = paddle.ones(shape=(batch_size, 1), dtype="bool")

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = decoder_layer(
                hidden_states,
                encoder_outputs=encoder_outputs,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                cross_attention_mask=cross_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
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

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.place,
                past_key_values_length=past_key_values_length,
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask


def vqa_history_to_prompt(history, query):
    prompt = "<EOI>Question: "
    prompt += query + " Short answer:"
    return prompt


def chat_old_history_to_prompt(history, query):
    prompt = "<EOI>Question: "
    for i, (old_query, response) in enumerate(history):
        prompt += old_query + " Answer: " + response + "\nQuestion: "
    prompt += query + " Answer:"
    return prompt


def chat_history_to_prompt(history, query):
    prompt = " [INST] "
    for i, (old_query, response) in enumerate(history):
        prompt += old_query + " [/INST] " + response + " [INST] "
    prompt += query + " [/INST] "
    return prompt


def base_history_to_prompt(history, query):
    prompt = query
    return prompt


_history_to_prompt_for_cogagent = {
    "base": base_history_to_prompt,
    "chat": chat_history_to_prompt,
    "chat_old": chat_old_history_to_prompt,
    "vqa": vqa_history_to_prompt,
}


def _history_to_prompt_for_cogvlm(signal_type, history, query):
    if signal_type == "base":
        return query
    elif signal_type == "vqa":
        answer_format = "Short answer:"
    elif signal_type == "chat":
        answer_format = "Answer:"
    else:
        assert False, f"Unknown signal type {signal_type}"
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "Question: " + old_query + " {} ".format(answer_format) + response + "\n"
    prompt += "Question: {} {}".format(query, answer_format)
    return prompt


class CogModelForCausalLM(CogPreTrainedModel):
    _auto_class = "AutoModelForCausalLM"

    def __init__(self, config):
        super().__init__(config)
        self.model_type = config.model_type
        self.model = CogModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = paddle.nn.Linear(
            in_features=config.hidden_size, out_features=config.vocab_size, bias_attr=False
        )

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
        images: List[List[paddle.Tensor]] = None,
        cross_images: List[List[paddle.Tensor]] = None,
        token_type_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_values: Optional[List[paddle.Tensor]] = None,
        inputs_embeds: Optional[paddle.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            images=images,
            cross_images=cross_images,
            token_type_ids=token_type_ids,
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
            shift_logits = shift_logits.reshape([-1, self.config.vocab_size])
            shift_labels = shift_labels.reshape(-1)
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

    def _prepare_attention_mask_for_generation(
        self, inputs: paddle.Tensor, pad_token_id: Optional[int], eos_token_id: Optional[Union[int, List[int]]]
    ) -> paddle.Tensor:
        return paddle.ones(shape=inputs.shape[:2], dtype="int64")

    def prepare_inputs_for_generation(
        self,
        input_ids,
        token_type_ids,
        images=None,
        cross_images=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        position_ids = kwargs.get("position_ids", None)
        if position_ids is None:
            position_ids = build_position_ids(token_type_ids, attention_mask)
        if past_key_values:
            input_ids = input_ids[:, -1:]
            token_type_ids = token_type_ids[:, -1:]
            position_ids = position_ids[:, -1:]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "token_type_ids": token_type_ids,
                "images": images,
                "cross_images": cross_images,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            new_token_type_ids = (
                paddle.ones(shape=(token_type_ids.shape[0], 1), dtype=token_type_ids.dtype) * LANGUAGE_TOKEN_TYPE
            )
            model_kwargs["token_type_ids"] = paddle.concat(x=[token_type_ids, new_token_type_ids], axis=-1)
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = paddle.concat(
                    x=[attention_mask, paddle.ones(shape=(attention_mask.shape[0], 1), dtype=attention_mask.dtype)],
                    axis=-1,
                )
        elif "decoder_attention_mask" in model_kwargs:
            decoder_attention_mask = model_kwargs["decoder_attention_mask"]
            model_kwargs["decoder_attention_mask"] = paddle.concat(
                x=[
                    decoder_attention_mask,
                    paddle.ones(shape=(decoder_attention_mask.shape[0], 1), dtype=decoder_attention_mask.dtype),
                ],
                axis=-1,
            )
        return model_kwargs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(axis=0, index=beam_idx) for past_state in layer_past),)
        return reordered_past

    def build_conversation_input_ids(
        self,
        tokenizer,
        *,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        images,
        template_version: Optional[Literal["base", "chat", "vqa"]] = None
    ):
        image_size: int = self.config.vision_config["image_size"]
        if self.model_type == "cogagent":
            cross_image_size: int = self.config.cross_image_size
        patch_size: int = self.config.vision_config["patch_size"]
        template_version = template_version or self.config.template_version
        assert images is None or len(images) <= 1, "not support multi images by now."
        history = history or []
        if self.model_type == "cogagent":
            text = _history_to_prompt_for_cogagent[template_version](history, query)
        elif self.model_type == "cogvlm":
            text = _history_to_prompt_for_cogvlm(template_version, history, query)
        else:
            raise ValueError("model_type in config must be cogagent or cogvlm, but got {}".format(self.model_type))

        input_ids = [tokenizer.bos_token_id]
        token_type_ids = [LANGUAGE_TOKEN_TYPE]
        if images is not None and len(images) == 1:
            transform = paddle.vision.transforms.Compose(
                [
                    paddle.vision.transforms.Resize((image_size, image_size), interpolation="bicubic"),
                    paddle.vision.transforms.ToTensor(),
                    paddle.vision.transforms.Normalize(
                        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                    ),
                ]
            )
            if self.model_type == "cogagent":
                ori = images
                images = [transform(ori[0])]
                cross_transform = paddle.vision.transforms.Compose(
                    [
                        paddle.vision.transforms.Resize((cross_image_size, cross_image_size), interpolation="bicubic"),
                        paddle.vision.transforms.ToTensor(),
                        paddle.vision.transforms.Normalize(
                            (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                        ),
                    ]
                )
                cross_images = [cross_transform(ori[0])]
            elif self.model_type == "cogvlm":
                images = [transform(images[0])]
                cross_images = None
            else:
                raise ValueError("model_type in config must be cogagent or cogvlm, but got {}".format(self.model_type))
            vision_token_num = image_size // patch_size * (image_size // patch_size) + 2
            input_ids += [tokenizer.pad_token_id] * vision_token_num
            token_type_ids += [VISION_TOKEN_TYPE] * vision_token_num
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        text_ids = text_ids["input_ids"]

        input_ids += text_ids
        token_type_ids += [LANGUAGE_TOKEN_TYPE] * len(text_ids)
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": paddle.to_tensor(data=input_ids, dtype="int64"),
            "token_type_ids": paddle.to_tensor(data=token_type_ids, dtype="int64"),
            "attention_mask": paddle.to_tensor(data=attention_mask, dtype="int64"),
            "images": images,
            "cross_images": cross_images,
        }
