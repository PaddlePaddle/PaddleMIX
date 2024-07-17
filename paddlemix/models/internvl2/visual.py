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
import paddle.nn as nn
import paddle.nn.functional as F
from typing import Callable, Dict, Optional, Union, Tuple
from models.model_utils import MixPretrainedModel
from utils import pad_input, unpad_input
from paddlenlp.transformers.activations import ACT2FN
from paddlenlp.transformers.model_outputs import (BaseModelOutput,
                                                  BaseModelOutputWithPooling)
from .configuration import InternVisionConfig
from functools import partial
from ppdiffusers.utils import logging
from einops import rearrange

try:
    from paddle.nn.functional.flash_attention import flash_attn_varlen_qkvpacked
    has_flash_attn = True
except:
    print('FlashAttention is not installed.')
    has_flash_attn = False

logger = logging.get_logger(__name__)

def _freeze_params(module):
    for param in module.parameters():
        param.stop_gradient = True


##util
def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = paddle.bernoulli(paddle.full(shape, keep_prob, dtype=x.dtype))
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = paddle.divide(random_tensor, paddle.to_tensor(keep_prob))
    return x * random_tensor


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"
##


class FlashAttention(nn.Layer):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert qkv.dtype in [paddle.float16, paddle.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = paddle.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=paddle.int32,
                                          device=qkv.device).place(qkv.place)
                output = flash_attn_varlen_qkvpacked(
                    qkv, cu_seqlens, cu_seqlens, max_s, max_s, self.softmax_scale, 
                    self.dropout_p if self.training else 0.0, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                output_unpad = flash_attn_varlen_qkvpacked(
                    x_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                             indices, batch_size, seqlen),
                                   'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            output = flash_attn_varlen_qkvpacked(
                qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )

        return output, None


class InternRMSNorm(nn.Layer):
    def __init__(self, hidden_size, eps=1e-6):
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
        hidden_states = hidden_states.to(paddle.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class InternVisionEmbeddings(nn.Layer):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = paddle.create_parameter(  
            shape=[1, 1, self.embed_dim],  
            default_initializer=paddle.nn.initializer.Normal(0.0, 1.0),  
            dtype='float32'  
        )

        self.patch_embedding = nn.Conv2D(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = paddle.create_parameter(  
            shape=[1, self.num_positions, self.embed_dim],  
            default_initializer=paddle.nn.initializer.Normal(0.0, 1.0),  
            dtype='float32'  
        )

    def _get_pos_embed(self, pos_embed, H, W):  
        target_dtype = pos_embed.dtype  
        pos_embed = pos_embed.reshape(  
            [1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1]).transpose([0, 3, 1, 2])  
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='BICUBIC', 
                        align_corners=False).reshape([1, -1, H * W]).transpose([0, 2, 1]).astype(target_dtype)  
        return pos_embed

    def forward(self, pixel_values) -> paddle.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.reshape([batch_size, self.embed_dim, -1]).transpose([0, 2, 1])
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).astype(target_dtype)
        embeddings = paddle.concat([class_embeds, patch_embeds], axis=1)
        position_embedding = paddle.concat([  
            self.position_embedding[:, :1, :],  
            self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)  
        ], axis=1) 
        embeddings = embeddings + position_embedding.astype(target_dtype)
        return embeddings

class InternAttention(nn.Layer):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_flash_attn = config.use_flash_attn and has_flash_attn
        if config.use_flash_attn and not has_flash_attn:
            print('Warning: Flash Attention is not available, use_flash_attn is set to False.')
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias_attr=config.qkv_bias)
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.dropout)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        if self.use_flash_attn:
            self.inner_attn = FlashAttention(attention_dropout=config.attention_dropout)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _naive_attn(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose([2, 0, 3, 1, 4])
        q, k, v = paddle.split(qkv, num_or_sections=3, axis=0)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose([0, 2, 1, 3]).reshape([B_, N_, -1])).reshape([B_, N_, H_, D_]).transpose([0, 2, 1, 3])
            k = self.k_norm(k.transpose([0, 2, 1, 3]).reshape([B_, N_, -1])).reshape([B_, N_, H_, D_]).transpose([0, 2, 1, 3])

        attn = ((q * self.scale) @ k.transpose([0, 1, 3, 2]))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose([0, 2, 1, 3]).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)

        if self.qk_normalization:
            q, k, v = paddle.split(qkv, num_or_sections=3, axis=2)

            q = q.squeeze(2)
            k = k.squeeze(2)
            v = v.squeeze(2)
            q = self.q_norm(q.reshape([q.shape[0], q.shape[1], -1])).reshape(q.shape)
            k = self.k_norm(k.reshape([k.shape[0], k.shape[1], -1])).reshape(k.shape)
            qkv = paddle.stack([q, k, v], axis=2)

        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=False
        )
        outs = self.proj(rearrange(context, 'b s h d -> b s (h d)'))
        outs = self.proj_drop(outs)
        return outs

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        x = self._naive_attn(hidden_states) if not self.use_flash_attn else self._flash_attn(hidden_states)
        return x

class InternMLP(nn.Layer):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.act = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternVisionEncoderLayer(nn.Layer):
    def __init__(self, config: InternVisionConfig, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        self.norm1 = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        self.ls1 = paddle.create_parameter(
            shape=[self.embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(config.initializer_factor * paddle.ones([self.embed_dim]))
        )
        
        self.ls2 = paddle.create_parameter(
            shape=[self.embed_dim],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Assign(config.initializer_factor * paddle.ones([self.embed_dim]))
        )
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(
            self,
            hidden_states: paddle.Tensor,
    ):
        hidden_states = hidden_states + self.drop_path1(self.attn(self.norm1(hidden_states)) * self.ls1)

        hidden_states = hidden_states + self.drop_path2(self.mlp(self.norm2(hidden_states)) * self.ls2)

        return hidden_states

class InternVisionEncoder(nn.Layer):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InternEncoderLayer`].

    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        # stochastic depth decay rule
        dpr = [x.item() for x in paddle.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layers = nn.LayerList([
            InternVisionEncoderLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

    def forward(
            self,
            inputs_embeds,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            
            layer_outputs = encoder_layer(
                hidden_states,
            )
            hidden_states = layer_outputs

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states
        )



class InternVisionModel(MixPretrainedModel):
    main_input_name = 'pixel_values'
    config_class = InternVisionConfig
    _no_split_modules = ['InternVisionEncoderLayer']

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pos_emb = self.embeddings.position_embedding
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size, old_size // patch_size, -1).transpose(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb.float(), size=new_size // patch_size, mode='bicubic', align_corners=False)
        pos_emb = pos_emb.astype(cls_emb.dtype).reshape(1, embed_dim, -1).transpose(0, 2, 1)
        pos_emb = paddle.concat([cls_emb, pos_emb], axis=1)
        self.embeddings.position_embedding = self.create_parameter(
            shape=pos_emb.shape,
            attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(pos_emb))
        )
        logger.info('Resized position embeddings from {} to {}'.format(old_size, new_size))

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
            self,
            pixel_values = None,
            output_hidden_states = None,
            return_dict = None,
            pixel_embeds = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and pixel_embeds is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) == 4:
                hidden_states = self.embeddings(pixel_values)
            else:
                raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )