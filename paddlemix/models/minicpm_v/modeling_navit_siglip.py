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

import os

import paddle

""" PyTorch Siglip model. """
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from paddle import nn
from paddlenlp.transformers import PretrainedConfig
from paddlenlp.transformers.activations import ACT2FN
from paddlenlp.transformers.model_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ModelOutput,
)
from paddlenlp.transformers.model_utils import PretrainedModel

from paddlemix.utils.initializer import _calculate_fan_in_and_fan_out

from .bert_padding import pad_input, unpad_input
from paddlemix.models.flash_attn_utils import has_flash_attn_func

flash_attn_func, flash_attn_varlen_func = has_flash_attn_func()

@dataclass
class PaddleAttentionMaskConverter:
    """
    A utility attention mask class for Paddle that allows one to:
        - Convert a 2d attention mask (batch_size, query_length) to a 4d attention mask
          (batch_size, 1, query_length, key_value_length)
    """

    @staticmethod
    def _expand_mask(mask: paddle.Tensor, dtype: str, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.shape
        tgt_len = tgt_len if tgt_len is not None else src_len

        # Expand dimensions: [bsz, 1, 1, src_len]
        expanded_mask = mask.unsqueeze([1, 2])

        # Broadcast to target shape: [bsz, 1, tgt_len, src_len]
        expanded_mask = paddle.expand(expanded_mask, shape=[bsz, 1, tgt_len, src_len])
        expanded_mask = expanded_mask.astype(dtype)

        # Invert the mask (1.0 for positions to attend to)
        inverted_mask = 1.0 - expanded_mask

        # Replace 1s with large negative values
        min_value = paddle.to_tensor(float("-1e9"), dtype=dtype)
        inverted_mask = paddle.where(inverted_mask.astype("bool"), min_value, paddle.zeros_like(inverted_mask))

        return inverted_mask


def _prepare_4d_attention_mask(mask: paddle.Tensor, dtype: str, tgt_len: Optional[int] = None):
    """
    Creates a 4D attention mask from a 2D mask.

    Args:
        mask (paddle.Tensor): A 2D attention mask of shape (batch_size, key_value_length)
        dtype (str): The dtype the created mask should have
        tgt_len (int, optional): The target length the created mask should have

    Returns:
        paddle.Tensor: A 4D attention mask of shape (batch_size, 1, query_length, key_value_length)
    """
    return PaddleAttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


class SigLipVisionConfig(PretrainedConfig):

    model_type = "siglip_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-06,
        attention_dropout=0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from SigLipConfig
        if config_dict.get("model_type") == "siglip":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


# _CHECKPOINT_FOR_DOC = 'google/siglip-base-patch16-224'


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype="int32")
    paddle.utils.try_import("warnings").warn("Now, the return shape is inconsistent with torch when as_tuple is True")
    indices = paddle.nonzero(x=attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = nn.functional.pad(
        x=paddle.cumsum(x=seqlens_in_batch, axis=0, dtype="int32"), pad=(1, 0), pad_from_left_axis=False
    )
    return indices, cu_seqlens, max_seqlen_in_batch


def _trunc_normal_(tensor, mean, std, a, b):
    # 确保mean是浮点数
    mean = float(mean)
    std = float(std)
    a = float(a)
    b = float(b)

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.",
            stacklevel=2,
        )
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    tensor.uniform_(min=2 * l - 1, max=2 * u - 1)
    if tensor.dtype in ["float16", "bfloat16"]:
        og_dtype = tensor.dtype
        tensor = tensor.to("float32")
        tensor.erfinv_()
        tensor = tensor.to(og_dtype)
    else:
        tensor.erfinv_()
    tensor.multiply_(y=paddle.to_tensor(std * math.sqrt(2.0)))
    tensor.add_(y=paddle.to_tensor(mean))
    if tensor.dtype == "float16":
        tensor = tensor.to("float32")
        tensor.clip_(min=a, max=b)
        tensor = tensor.to("float16")
    else:
        tensor.clip_(min=a, max=b)


def trunc_normal_tf_(
    tensor: paddle.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> paddle.Tensor:
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(	ext{mean}, 	ext{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq 	ext{mean} \\leq b`.
    NOTE: this 'tf' variant behaves closer to Tensorflow / JAX impl where the
    bounds [a, b] are applied when sampling the normal distribution with mean=0, std=1.0
    and the result is subsquently scaled and shifted by the mean and std args.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    """
    with paddle.no_grad():
        _trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.multiply_(y=paddle.to_tensor(std)).add_(y=paddle.to_tensor(mean))


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.8796256610342398)
    elif distribution == "normal":
        with paddle.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        with paddle.no_grad():
            tensor.uniform_(min=-bound, max=bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def default_flax_embed_init(tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="normal")


@dataclass
class SiglipVisionModelOutput(ModelOutput):
    """
    Base class for vision model's outputs that also contains image embeddings of the pooling of the last hidden states.
    Args:
        image_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)` *optional* returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    image_embeds: Optional[paddle.Tensor] = None
    last_hidden_state: paddle.float32 = None
    hidden_states: Optional[Tuple[paddle.Tensor]] = None
    attentions: Optional[Tuple[paddle.Tensor]] = None


class SiglipVisionEmbeddings(nn.Layer):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2D(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )
        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def forward(
        self,
        pixel_values: paddle.Tensor,
        patch_attention_mask: paddle.Tensor,
        tgt_sizes: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose([0, 2, 1])

        max_im_h, max_im_w = pixel_values.shape[2], pixel_values.shape[3]
        max_nb_patches_h, max_nb_patches_w = (max_im_h // self.patch_size, max_im_w // self.patch_size)
        boundaries = paddle.arange(start=1 / self.num_patches_per_side, end=1.0, step=1 / self.num_patches_per_side)
        position_ids = paddle.full(shape=(batch_size, max_nb_patches_h * max_nb_patches_w), fill_value=0)
        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            if tgt_sizes is not None:
                nb_patches_h = tgt_sizes[batch_idx][0]
                nb_patches_w = tgt_sizes[batch_idx][1]
            else:
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = paddle.arange(start=0, end=1 - 1e-06, step=1 / nb_patches_h)
            fractional_coords_w = paddle.arange(start=0, end=1 - 1e-06, step=1 / nb_patches_w)
            bucket_coords_h = paddle.bucketize(x=fractional_coords_h, sorted_sequence=boundaries, right=True)
            bucket_coords_w = paddle.bucketize(x=fractional_coords_w, sorted_sequence=boundaries, right=True)
            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx].scatter_(
                paddle.nonzero(p_attn_mask.reshape([-1]))[:, 0], pos_ids.astype(position_ids.dtype)
            )
        position_ids = position_ids.to(self.position_embedding.weight.place)

        embeddings = embeddings + self.position_embedding(position_ids.cast("int64"))
        return embeddings


class SigLipAttention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape([batch_size, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1])
        key_states = key_states.reshape([batch_size, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1])
        value_states = value_states.reshape([batch_size, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1])

        k_v_seq_len = key_states.shape[-2]
        attn_weights = paddle.matmul(query_states, key_states.transpose([0, 1, 3, 2])) * self.scale

        if attn_weights.shape != [batch_size, self.num_heads, q_len, k_v_seq_len]:
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != [batch_size, 1, q_len, k_v_seq_len]:
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.shape}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, axis=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = paddle.matmul(attn_weights, value_states)

        if attn_output.shape != [batch_size, self.num_heads, q_len, self.head_dim]:
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.transpose([0, 2, 1]).contiguous()
        attn_output = attn_output.reshape([batch_size, q_len, self.embed_dim])

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipFlashAttention2(SigLipAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        position_ids: Optional[paddle.Tensor] = None,
        past_key_value: Optional[Tuple[paddle.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor], Optional[Tuple[paddle.Tensor]]]:
        output_attentions = False
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1])
        key_states = key_states.reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1])
        value_states = value_states.reshape([bsz, q_len, self.num_heads, self.head_dim]).transpose([0, 2, 1])

        kv_seq_len = tuple(key_states.shape)[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        query_states = query_states.transpose([0, 2, 1])
        key_states = key_states.transpose([0, 2, 1])
        value_states = value_states.transpose([0, 2, 1])

        dropout_rate = self.dropout if self.training else 0.0
        input_dtype = query_states.dtype
        if input_dtype == paddle.float32:
            if paddle.amp.is_auto_cast_enabled():
                target_dtype = paddle.amp.get_default_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )
        attn_output = attn_output.reshape(bsz, q_len, self.embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights

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
        # Contains at least one padding token in the sequence
        causal = self.is_causal and query_length != 1

        head_dim = query_states.shape[-1]
        softmax_scale = head_dim**-0.5  # TODO: 需要手动加上

        if attention_mask is not None:
            batch_size = query_states.shape[0]  # [2, 3383, 16, 128]

            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = unpad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(  # TODO: flash_attn_unpadded
                query_states,  # [5998, 16, 128]
                key_states,  # [5998, 8, 128]
                value_states,  # [5998, 8, 128]
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                scale=softmax_scale,  # not softmax_scale=
                dropout=dropout,
                causal=causal,
            )[0]

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                causal=causal,  # no softmax_scale=
            )[0]

        return attn_output


class SigLipMLP(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SigLipEncoderLayer(nn.Layer):
    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SigLipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_eps)
        self.mlp = SigLipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, epsilon=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: paddle.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[paddle.Tensor]:
        """
        Args:
            hidden_states (`paddle.Tensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`paddle.Tensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SigLipPreTrainedModel(PretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SigLipVisionConfig
    base_model_prefix = "siglip"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, SiglipVisionEmbeddings):
            width = self.config.hidden_size
            init_Normal = nn.initializer.Normal(std=1 / np.sqrt(width))
            init_Normal(module.position_embedding.weight)
        elif isinstance(module, nn.Embedding):
            default_flax_embed_init(module.weight)
        elif isinstance(module, SigLipAttention):
            # 初始化投影层权重
            for proj in [module.q_proj, module.k_proj, module.v_proj, module.out_proj]:
                init_Normal = nn.initializer.Normal()
                init_Normal(proj.weight)
                # 使用assign替代原地操作初始化偏置
                if hasattr(proj, "bias") and proj.bias is not None:
                    proj.bias.set_value(paddle.zeros_like(proj.bias))

        elif isinstance(module, SigLipMLP):
            # 初始化FC层权重
            init_Normal = nn.initializer.Normal()
            init_Normal(module.fc1.weight)
            init_Normal(module.fc2.weight)

            # 使用assign初始化偏置
            if hasattr(module.fc1, "bias") and module.fc1.bias is not None:
                module.fc1.bias.set_value(paddle.normal(shape=module.fc1.bias.shape, mean=0.0, std=1e-06))
            if hasattr(module.fc2, "bias") and module.fc2.bias is not None:
                module.fc2.bias.set_value(paddle.normal(shape=module.fc2.bias.shape, mean=0.0, std=1e-06))

        elif isinstance(module, (nn.Linear, nn.Conv2D)):
            lecun_normal_(module.weight)
            if module.bias is not None:
                module.bias.set_value(paddle.zeros_like(module.bias))

        elif isinstance(module, nn.LayerNorm):
            # 使用set_value替代原地操作
            if module.bias is not None:
                module.bias.set_value(paddle.zeros_like(module.bias))
            if module.weight is not None:
                module.weight.set_value(paddle.ones_like(module.weight))


SIGLIP_START_DOCSTRING = """
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`SiglipVisionConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""
SIGLIP_VISION_INPUTS_DOCSTRING = """
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`CLIPImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class SigLipEncoder(nn.Layer):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`SiglipEncoderLayer`].
    Args:
        config: SiglipConfig
    """

    def __init__(self, config: SigLipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.LayerList(sublayers=[SigLipEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__, hidden_states, attention_mask, output_attentions
                )
            else:
                layer_outputs = encoder_layer(hidden_states, attention_mask, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


# Copied from transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_with_cache_position
def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: paddle.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: paddle.dtype,
    min_dtype: float,
    cache_position: paddle.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`paddle.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`paddle.dtype`):
            The dtype to use for the 4D attention mask.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`paddle.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`paddle.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = paddle.full([sequence_length, target_length], fill_value=min_dtype, dtype=dtype)
        if sequence_length != 1:
            causal_mask = paddle.triu(x=causal_mask, diagonal=1)
        causal_mask *= paddle.arange(target_length) > cache_position.reshape([-1, 1])
        causal_mask = causal_mask[None, None, :, :].expand(shape=[batch_size, 1, -1, -1])
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = tuple(attention_mask.shape)[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                mask=padding_mask, value=min_dtype
            )

    return causal_mask


class SigLipVisionTransformer(SigLipPreTrainedModel):
    config_class = SigLipVisionConfig
    main_input_name = "pixel_values"
    _supports_flash_attn_2 = True

    def __init__(self, config: SigLipVisionConfig):
        super().__init__(config)
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SigLipEncoder(config)
        self.post_layernorm = nn.LayerNorm(normalized_shape=embed_dim, epsilon=config.layer_norm_eps)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # self.post_init()

    def get_input_embeddings(self) -> nn.Layer:
        return self.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        patch_attention_mask: Optional[paddle.Tensor] = None,
        tgt_sizes: Optional[paddle.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = pixel_values.shape[0]
        if patch_attention_mask is None:
            patch_attention_mask = paddle.ones(
                shape=(
                    batch_size,
                    pixel_values.shape[2] // self.config.patch_size,
                    pixel_values.shape[3] // self.config.patch_size,
                ),
                dtype="bool",
            )

        hidden_states = self.embeddings(
            pixel_values=pixel_values, patch_attention_mask=patch_attention_mask, tgt_sizes=tgt_sizes
        )
        patch_attention_mask = patch_attention_mask.reshape([batch_size, -1])
        if not paddle.any(x=~patch_attention_mask):
            attention_mask = None
        else:
            attention_mask = (
                _prepare_4d_attention_mask(patch_attention_mask, hidden_states.dtype)
                if not self._use_flash_attention_2
                else patch_attention_mask
            )
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.post_layernorm(last_hidden_state)
        if not return_dict:
            return (last_hidden_state, None) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
