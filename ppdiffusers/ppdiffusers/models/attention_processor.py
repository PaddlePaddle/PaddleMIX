# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import Optional, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..utils import deprecate, is_ppxformers_available, logging
from .lora import LoRALinearLayer

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Attention(nn.Layer):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block=False,
        processor: Optional["AttnProcessor"] = None,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = heads
        self.head_dim = dim_head
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, epsilon=eps)
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(f_channels=query_dim, zq_channels=spatial_norm_dim)
        else:
            self.spatial_norm = None

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, epsilon=1e-5
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        self.to_q = nn.Linear(query_dim, inner_dim, bias_attr=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias_attr=bias)
            self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias_attr=bias)
        else:
            self.to_k = None
            self.to_v = None

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, inner_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, inner_dim)

        self.to_out = nn.LayerList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim, bias_attr=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        # set attention processor
        if processor is None:
            processor = AttnProcessor2_5() if is_ppxformers_available() else AttnProcessor()
        self.set_processor(processor)

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[str] = None
    ):
        is_lora = hasattr(self, "processor") and isinstance(
            self.processor,
            LORA_ATTENTION_PROCESSORS,
        )
        is_custom_diffusion = hasattr(self, "processor") and isinstance(
            self.processor, (CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor)
        )
        is_added_kv_processor = hasattr(self, "processor") and isinstance(
            self.processor,
            (
                AttnAddedKVProcessor,
                AttnAddedKVProcessor2_5,
                SlicedAttnAddedKVProcessor,
                XFormersAttnAddedKVProcessor,
                LoRAAttnAddedKVProcessor,
            ),
        )
        if use_memory_efficient_attention_xformers:
            if is_added_kv_processor and (is_lora or is_custom_diffusion):
                raise NotImplementedError(
                    f"Memory efficient attention is currently not supported for LoRA or custom diffuson for attention processor type {self.processor}"
                )
            if not is_ppxformers_available():
                raise NotImplementedError(
                    "requires the scaled_dot_product_attention but your PaddlePaddle donot have this. Checkout the instructions on the installation page: https://www.paddlepaddle.org.cn/install/quick and follow the ones that match your environment."
                )
            else:
                try:
                    # Make sure we can run the memory efficient attention
                    _ = F.scaled_dot_product_attention_(
                        paddle.ones((1, 1, 2, 40), dtype=paddle.float16),
                        paddle.ones((1, 1, 2, 40), dtype=paddle.float16),
                        paddle.ones((1, 1, 2, 40), dtype=paddle.float16),
                        attention_op=attention_op,
                    )
                except Exception as e:
                    raise e
            # remove this PR: https://github.com/PaddlePaddle/Paddle/pull/56045
            # if self.head_size > 128 and attention_op == "flash":
            #     attention_op = "cutlass"
            if is_lora:
                processor = LoRAXFormersAttnProcessor(
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    rank=self.processor.rank,
                    attention_op=attention_op,
                )
                # we must cast dtype
                processor.to(dtype=self.dtype)
                processor.load_dict(self.processor.state_dict())
            elif is_custom_diffusion:
                processor = CustomDiffusionXFormersAttnProcessor(
                    train_kv=self.processor.train_kv,
                    train_q_out=self.processor.train_q_out,
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    attention_op=attention_op,
                )
                # we must cast dtype
                processor.to(dtype=self.dtype)
                processor.load_dict(self.processor.state_dict())
            elif is_added_kv_processor:
                # TODO(Patrick, Suraj, William) - currently xformers doesn't work for UnCLIP
                # which uses this type of cross attention ONLY because the attention mask of format
                # [0, ..., -10.000, ..., 0, ...,] is not supported
                # throw warning
                logger.info(
                    "Memory efficient attention with `xformers` might currently not work correctly if an attention mask is required for the attention operation."
                )
                processor = XFormersAttnAddedKVProcessor(attention_op=attention_op)
            else:
                processor = XFormersAttnProcessor(attention_op=attention_op)
        else:
            if is_lora:
                attn_processor_class = LoRAAttnProcessor2_5 if is_ppxformers_available() else LoRAAttnProcessor
                processor = attn_processor_class(
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    rank=self.processor.rank,
                )
                # we must cast dtype
                processor.to(dtype=self.dtype)
                processor.load_dict(self.processor.state_dict())
            elif is_custom_diffusion:
                processor = CustomDiffusionAttnProcessor(
                    train_kv=self.processor.train_kv,
                    train_q_out=self.processor.train_q_out,
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                )
                # we must cast dtype
                processor.to(dtype=self.dtype)
                processor.load_dict(self.processor.state_dict())
            else:
                processor = AttnProcessor2_5() if is_ppxformers_available() else AttnProcessor()

        self.set_processor(processor)

    def set_attention_slice(self, slice_size):
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        if slice_size is not None and self.added_kv_proj_dim is not None:
            processor = SlicedAttnAddedKVProcessor(slice_size)
        elif slice_size is not None:
            processor = SlicedAttnProcessor(slice_size)
        elif self.added_kv_proj_dim is not None:
            processor = AttnAddedKVProcessor()
        else:
            processor = AttnProcessor2_5() if is_ppxformers_available() else AttnProcessor()

        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor"):
        # if current processor is in `self._sub_layers` and if passed `processor` is not, we need to
        # pop `processor` from `self._sub_layers`
        if hasattr(self, "processor") and isinstance(self.processor, nn.Layer) and not isinstance(processor, nn.Layer):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._sub_layers.pop("processor")

        self.processor = processor

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor, transpose=True, in_dim=4):
        if in_dim == 3:
            head_size = self.heads
            batch_size, seq_len, dim = tensor.shape
            tensor = tensor.reshape([batch_size // head_size, head_size, seq_len, dim])
        if transpose:
            tensor = tensor.transpose([0, 2, 1, 3])
        tensor = tensor.reshape([0, 0, tensor.shape[2] * tensor.shape[3]])
        return tensor

    def head_to_batch_dim(self, tensor, transpose=True, out_dim=4):
        tensor = tensor.reshape([0, 0, self.heads, self.head_dim])
        if transpose or out_dim == 3:
            tensor = tensor.transpose([0, 2, 1, 3])
        if out_dim == 3:
            tensor = tensor.flatten(0, 1)
        return tensor

    def get_attention_scores(self, query, key, attention_mask=None):
        if self.upcast_softmax or self.upcast_attention:
            dtype = query.dtype

        if self.upcast_attention:
            query = query.cast(paddle.float32)
            key = key.cast(paddle.float32)

        attention_scores = paddle.matmul(query, key, transpose_y=True) * self.scale

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if self.upcast_softmax:
            attention_scores = attention_scores.cast(paddle.float32)

        attention_probs = F.softmax(attention_scores, axis=-1)

        if self.upcast_softmax or self.upcast_attention:
            attention_probs = attention_probs.cast(dtype)

        return attention_probs

    def prepare_attention_mask(self, attention_mask, target_length, batch_size=None, out_dim=4, transpose=True):
        if batch_size is None:
            deprecate(
                "batch_size=None",
                "0.0.15",
                message=(
                    "Not passing the `batch_size` parameter to `prepare_attention_mask` can lead to incorrect"
                    " attention mask preparation and is deprecated behavior. Please make sure to pass `batch_size` to"
                    " `prepare_attention_mask` when preparing the attention_mask."
                ),
            )
            batch_size = 1

        num_heads = self.heads
        if attention_mask is None:
            return attention_mask

        current_length = attention_mask.shape[-1]
        if current_length != target_length:
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0, data_format="NCL")
        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * num_heads:
                attention_mask = attention_mask.repeat_interleave(num_heads, axis=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            if attention_mask.shape[0] < batch_size * num_heads:
                attention_mask = attention_mask.repeat_interleave(num_heads, axis=1)
            attention_mask = paddle.reshape(attention_mask, [batch_size, num_heads, -1, attention_mask.shape[-1]])

        if attention_mask.ndim == 4:
            if not transpose:
                attention_mask = attention_mask.transpose([0, 2, 1, 3])
        return attention_mask

    def norm_encoder_hidden_states(self, encoder_hidden_states):
        assert self.norm_cross is not None, "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, nn.LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.transpose([0, 2, 1])
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.transpose([0, 2, 1])
        else:
            assert False

        return encoder_hidden_states


class AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        **cross_attention_kwargs
    ):
        if attn.residual_connection:
            residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.reshape([batch_size, channel, height * width]).transpose([0, 2, 1])

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose([0, 2, 1])).transpose([0, 2, 1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = paddle.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose([0, 2, 1]).reshape([batch_size, channel, height, width])

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class LoRAAttnProcessor(nn.Layer):
    r"""
    Processor for implementing the LoRA attention mechanism.

    Args:
        hidden_size (`int`, *optional*):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, rank=4, network_alpha=None, **kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank

        q_rank = kwargs.pop("q_rank", None)
        q_hidden_size = kwargs.pop("q_hidden_size", None)
        q_rank = q_rank if q_rank is not None else rank
        q_hidden_size = q_hidden_size if q_hidden_size is not None else hidden_size

        v_rank = kwargs.pop("v_rank", None)
        v_hidden_size = kwargs.pop("v_hidden_size", None)
        v_rank = v_rank if v_rank is not None else rank
        v_hidden_size = v_hidden_size if v_hidden_size is not None else hidden_size

        out_rank = kwargs.pop("out_rank", None)
        out_hidden_size = kwargs.pop("out_hidden_size", None)
        out_rank = out_rank if out_rank is not None else rank
        out_hidden_size = out_hidden_size if out_hidden_size is not None else hidden_size

        self.to_q_lora = LoRALinearLayer(q_hidden_size, q_hidden_size, q_rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or v_hidden_size, v_hidden_size, v_rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(out_hidden_size, out_hidden_size, out_rank, network_alpha)

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        scale=1.0,
        temb=None,
        **cross_attention_kwargs
    ):
        if attn.residual_connection:
            residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.reshape([batch_size, channel, height * width]).transpose([0, 2, 1])

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose([0, 2, 1])).transpose([0, 2, 1])

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = paddle.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose([0, 2, 1]).reshape([batch_size, channel, height, width])

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CustomDiffusionAttnProcessor(nn.Layer):
    r"""
    Processor for implementing attention for the Custom Diffusion method.

    Args:
        train_kv (`bool`, defaults to `True`):
            Whether to newly train the key and value matrices corresponding to the text features.
        train_q_out (`bool`, defaults to `True`):
            Whether to newly train query matrices corresponding to the latent image features.
        hidden_size (`int`, *optional*, defaults to `None`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*, defaults to `None`):
            The number of channels in the `encoder_hidden_states`.
        out_bias (`bool`, defaults to `True`):
            Whether to include the bias parameter in `train_q_out`.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
    """

    def __init__(
        self,
        train_kv=True,
        train_q_out=True,
        hidden_size=None,
        cross_attention_dim=None,
        out_bias=True,
        dropout=0.0,
    ):
        super().__init__()
        self.train_kv = train_kv
        self.train_q_out = train_q_out

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        # `_custom_diffusion` id for easy serialization and loading.
        if self.train_kv:
            self.to_k_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias_attr=False)
            self.to_v_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias_attr=False)
        if self.train_q_out:
            self.to_q_custom_diffusion = nn.Linear(hidden_size, hidden_size, bias_attr=False)
            self.to_out_custom_diffusion = nn.LayerList([])
            self.to_out_custom_diffusion.append(nn.Linear(hidden_size, hidden_size, bias_attr=out_bias))
            self.to_out_custom_diffusion.append(nn.Dropout(dropout))

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if self.train_q_out:
            query = self.to_q_custom_diffusion(hidden_states).cast(dtype=attn.to_q.weight.dtype)
        else:
            query = attn.to_q(hidden_states).cast(dtype=attn.to_q.weight.dtype)

        if encoder_hidden_states is None:
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self.train_kv:
            key = self.to_k_custom_diffusion(encoder_hidden_states.cast(self.to_k_custom_diffusion.weight.dtype))
            value = self.to_v_custom_diffusion(encoder_hidden_states.cast(self.to_v_custom_diffusion.weight.dtype))
            key = key.cast(attn.to_q.weight.dtype)
            value = value.cast(attn.to_q.weight.dtype)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        if crossattn:
            detach = paddle.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.0
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = paddle.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if self.train_q_out:
            # linear proj
            hidden_states = self.to_out_custom_diffusion[0](hidden_states)
            # dropout
            hidden_states = self.to_out_custom_diffusion[1](hidden_states)
        else:
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class AttnAddedKVProcessor:
    r"""
    Processor for performing attention-related computations with extra learnable key and value matrices for the text
    encoder.
    """

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs
    ):
        residual = hidden_states
        hidden_states = hidden_states.reshape([hidden_states.shape[0], hidden_states.shape[1], -1]).transpose(
            [0, 2, 1]
        )
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        hidden_states = attn.group_norm(hidden_states.transpose([0, 2, 1])).transpose([0, 2, 1])

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)

        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key = paddle.concat([encoder_hidden_states_key_proj, key], axis=2)
            value = paddle.concat([encoder_hidden_states_value_proj, value], axis=2)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = paddle.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose([0, 2, 1]).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states


class LoRAAttnAddedKVProcessor(nn.Layer):
    r"""
    Processor for implementing the LoRA attention mechanism with extra learnable key and value matrices for the text
    encoder.

    Args:
        hidden_size (`int`, *optional*):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*, defaults to `None`):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.

    """

    def __init__(self, hidden_size, cross_attention_dim=None, rank=4, network_alpha=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.add_k_proj_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.add_v_proj_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank, network_alpha)

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        scale=1.0,
        **cross_attention_kwargs
    ):
        residual = hidden_states
        hidden_states = hidden_states.reshape([hidden_states.shape[0], hidden_states.shape[1], -1]).transpose(
            [0, 2, 1]
        )
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        hidden_states = attn.group_norm(hidden_states.transpose([0, 2, 1])).transpose([0, 2, 1])

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states) + scale * self.add_k_proj_lora(
            encoder_hidden_states
        )
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states) + scale * self.add_v_proj_lora(
            encoder_hidden_states
        )
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)

        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states) + scale * self.to_k_lora(hidden_states)
            value = attn.to_v(hidden_states) + scale * self.to_v_lora(hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key = paddle.concat([encoder_hidden_states_key_proj, key], axis=1)
            value = paddle.concat([encoder_hidden_states_value_proj, value], axis=1)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = paddle.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose([0, 2, 1]).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states


class XFormersAttnAddedKVProcessor:
    r"""
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`str`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[str] = None):
        assert attention_op in [None, "auto", "cutlass", "flash"]
        self.attention_op = attention_op

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs
    ):
        residual = hidden_states
        hidden_states = hidden_states.reshape([hidden_states.shape[0], hidden_states.shape[1], -1]).transpose(
            [0, 2, 1]
        )
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size, transpose=False)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        hidden_states = attn.group_norm(hidden_states.transpose([0, 2, 1])).transpose([0, 2, 1])

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query, transpose=False)

        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj, transpose=False)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj, transpose=False)

        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key, transpose=False)
            value = attn.head_to_batch_dim(value, transpose=False)
            key = paddle.concat([encoder_hidden_states_key_proj, key], axis=1)
            value = paddle.concat([encoder_hidden_states_value_proj, value], axis=1)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj

        hidden_states = F.scaled_dot_product_attention_(
            query,
            key,
            value,
            attn_mask=attention_mask,
            scale=attn.scale,
            dropout_p=0.0,
            training=attn.training,
            attention_op=self.attention_op,
        )
        hidden_states = attn.batch_to_head_dim(hidden_states, transpose=False)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose([0, 2, 1]).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states


class XFormersAttnProcessor:
    r"""
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[str] = None):
        assert attention_op in [None, "auto", "cutlass", "flash"]
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        temb: Optional[paddle.Tensor] = None,
        **cross_attention_kwargs
    ):
        if attn.residual_connection:
            residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.reshape([batch_size, channel, height * width]).transpose([0, 2, 1])

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size, transpose=False)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand([-1, query_tokens, -1, -1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose([0, 2, 1])).transpose([0, 2, 1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # if transpose = False, query's shape will be [batch_size, seq_len, num_head, head_dim]
        query = attn.head_to_batch_dim(query, transpose=False)
        key = attn.head_to_batch_dim(key, transpose=False)
        value = attn.head_to_batch_dim(value, transpose=False)

        hidden_states = F.scaled_dot_product_attention_(
            query,
            key,
            value,
            attn_mask=attention_mask,
            scale=attn.scale,
            dropout_p=0.0,
            training=attn.training,
            attention_op=self.attention_op,
        )

        # hidden_states = hidden_states.cast(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states, transpose=False)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose([0, 2, 1]).reshape([batch_size, channel, height, width])

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class LoRAXFormersAttnProcessor(nn.Layer):
    r"""
    Processor for implementing the LoRA attention mechanism with memory efficient attention using xFormers.

    Args:
        hidden_size (`int`, *optional*):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.

    """

    def __init__(
        self,
        hidden_size,
        cross_attention_dim,
        rank=4,
        attention_op: Optional[str] = None,
        network_alpha=None,
        **kwargs,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        self.attention_op = attention_op

        q_rank = kwargs.pop("q_rank", None)
        q_hidden_size = kwargs.pop("q_hidden_size", None)
        q_rank = q_rank if q_rank is not None else rank
        q_hidden_size = q_hidden_size if q_hidden_size is not None else hidden_size

        v_rank = kwargs.pop("v_rank", None)
        v_hidden_size = kwargs.pop("v_hidden_size", None)
        v_rank = v_rank if v_rank is not None else rank
        v_hidden_size = v_hidden_size if v_hidden_size is not None else hidden_size

        out_rank = kwargs.pop("out_rank", None)
        out_hidden_size = kwargs.pop("out_hidden_size", None)
        out_rank = out_rank if out_rank is not None else rank
        out_hidden_size = out_hidden_size if out_hidden_size is not None else hidden_size

        self.to_q_lora = LoRALinearLayer(q_hidden_size, q_hidden_size, q_rank, network_alpha)
        self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or v_hidden_size, v_hidden_size, v_rank, network_alpha)
        self.to_out_lora = LoRALinearLayer(out_hidden_size, out_hidden_size, out_rank, network_alpha)

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        scale=1.0,
        temb=None,
        **cross_attention_kwargs
    ):
        if attn.residual_connection:
            residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.reshape([batch_size, channel, height * width]).transpose([0, 2, 1])

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size, transpose=False)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose([0, 2, 1])).transpose([0, 2, 1])

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query, transpose=False)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)

        key = attn.head_to_batch_dim(key, transpose=False)
        value = attn.head_to_batch_dim(value, transpose=False)

        hidden_states = F.scaled_dot_product_attention_(
            query,
            key,
            value,
            attn_mask=attention_mask,
            scale=attn.scale,
            dropout_p=0.0,
            training=attn.training,
            attention_op=self.attention_op,
        )

        hidden_states = attn.batch_to_head_dim(hidden_states, transpose=False)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose([0, 2, 1]).reshape([batch_size, channel, height, width])

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CustomDiffusionXFormersAttnProcessor(nn.Layer):
    r"""
    Processor for implementing memory efficient attention using xFormers for the Custom Diffusion method.

    Args:
    train_kv (`bool`, defaults to `True`):
        Whether to newly train the key and value matrices corresponding to the text features.
    train_q_out (`bool`, defaults to `True`):
        Whether to newly train query matrices corresponding to the latent image features.
    hidden_size (`int`, *optional*, defaults to `None`):
        The hidden size of the attention layer.
    cross_attention_dim (`int`, *optional*, defaults to `None`):
        The number of channels in the `encoder_hidden_states`.
    out_bias (`bool`, defaults to `True`):
        Whether to include the bias parameter in `train_q_out`.
    dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability to use.
    attention_op (`Callable`, *optional*, defaults to `None`):
        The base
        [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to use
        as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best operator.
    """

    def __init__(
        self,
        train_kv=True,
        train_q_out=False,
        hidden_size=None,
        cross_attention_dim=None,
        out_bias=True,
        dropout=0.0,
        attention_op: Optional[str] = None,
    ):
        super().__init__()
        assert attention_op in [None, "auto", "cutlass", "flash"]
        self.train_kv = train_kv
        self.train_q_out = train_q_out

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.attention_op = attention_op

        # `_custom_diffusion` id for easy serialization and loading.
        if self.train_kv:
            self.to_k_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias_attr=False)
            self.to_v_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias_attr=False)
        if self.train_q_out:
            self.to_q_custom_diffusion = nn.Linear(hidden_size, hidden_size, bias_attr=False)
            self.to_out_custom_diffusion = nn.LayerList([])
            self.to_out_custom_diffusion.append(nn.Linear(hidden_size, hidden_size, bias_attr=out_bias))
            self.to_out_custom_diffusion.append(nn.Dropout(dropout))

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size, transpose=False)

        if self.train_q_out:
            query = self.to_q_custom_diffusion(hidden_states).cast(attn.to_q.weight.dtype)
        else:
            query = attn.to_q(hidden_states).cast(attn.to_q.weight.dtype)

        if encoder_hidden_states is None:
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self.train_kv:
            key = self.to_k_custom_diffusion(encoder_hidden_states.cast(self.to_k_custom_diffusion.weight.dtype))
            value = self.to_v_custom_diffusion(encoder_hidden_states.cast(self.to_v_custom_diffusion.weight.dtype))
            key = key.cast(attn.to_q.weight.dtype)
            value = value.cast(attn.to_q.weight.dtype)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        if crossattn:
            detach = paddle.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.0
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()

        # if transpose = False, query's shape will be [batch_size, seq_len, num_head, head_dim]
        query = attn.head_to_batch_dim(query, transpose=False)
        key = attn.head_to_batch_dim(key, transpose=False)
        value = attn.head_to_batch_dim(value, transpose=False)

        hidden_states = F.scaled_dot_product_attention_(
            query,
            key,
            value,
            attn_mask=attention_mask,
            scale=attn.scale,
            dropout_p=0.0,
            training=attn.training,
            attention_op=self.attention_op,
        )
        # hidden_states = hidden_states.cast(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states, transpose=False)

        if self.train_q_out:
            # linear proj
            hidden_states = self.to_out_custom_diffusion[0](hidden_states)
            # dropout
            hidden_states = self.to_out_custom_diffusion[1](hidden_states)
        else:
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class SlicedAttnProcessor:
    r"""
    Processor for implementing sliced attention.

    Args:
        slice_size (`int`, *optional*):
            The number of steps to compute attention. Uses as many slices as `attention_head_dim // slice_size`, and
            `attention_head_dim` must be a multiple of the `slice_size`.
    """

    def __init__(self, slice_size):
        self.slice_size = slice_size

    def __call__(
        self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs
    ):
        if attn.residual_connection:
            residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.reshape([batch_size, channel, height * width]).transpose([0, 2, 1])
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size, out_dim=3)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose([0, 2, 1])).transpose([0, 2, 1])

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        query = query.flatten(0, 1)
        key = key.flatten(0, 1)
        value = value.flatten(0, 1)

        batch_size_attention = query.shape[0]
        query_len = query.shape[1]
        hidden_states = paddle.zeros((batch_size_attention, query_len, attn.head_dim), dtype=query.dtype)
        for i in range(batch_size_attention // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)

            attn_slice = paddle.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape back to [bs, num_heads, seqlen, head_dim]
        hidden_states = hidden_states.reshape([-1, attn.heads, query_len, attn.head_dim])

        hidden_states = attn.batch_to_head_dim(hidden_states)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose([0, 2, 1]).reshape([batch_size, channel, height, width])

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SlicedAttnAddedKVProcessor:
    r"""
    Processor for implementing sliced attention with extra learnable key and value matrices for the text encoder.

    Args:
        slice_size (`int`, *optional*):
            The number of steps to compute attention. Uses as many slices as `attention_head_dim // slice_size`, and
            `attention_head_dim` must be a multiple of the `slice_size`.
    """

    def __init__(self, slice_size):
        self.slice_size = slice_size

    def __call__(
        self,
        attn: "Attention",
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        **cross_attention_kwargs
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        hidden_states = hidden_states.reshape([hidden_states.shape[0], hidden_states.shape[1], -1]).transpose(
            [0, 2, 1]
        )

        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size, out_dim=3)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        hidden_states = attn.group_norm(hidden_states.transpose([0, 2, 1])).transpose([0, 2, 1])

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)

        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key = paddle.concat([encoder_hidden_states_key_proj, key], axis=2)
            value = paddle.concat([encoder_hidden_states_value_proj, value], axis=2)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj

        # flatten
        query = query.flatten(0, 1)
        key = key.flatten(0, 1)
        value = value.flatten(0, 1)

        batch_size_attention = query.shape[0]
        query_len = query.shape[1]
        hidden_states = paddle.zeros((batch_size_attention, query_len, attn.head_dim), dtype=query.dtype)

        for i in range(batch_size_attention // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)

            attn_slice = paddle.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape back to [bs, num_heads, seqlen, head_dim]
        hidden_states = hidden_states.reshape([-1, attn.heads, query_len, attn.head_dim])

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose([0, 2, 1]).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states


AttnProcessor2_5 = XFormersAttnProcessor
AttnAddedKVProcessor2_5 = XFormersAttnAddedKVProcessor
LoRAAttnProcessor2_5 = LoRAXFormersAttnProcessor
AttentionProcessor = Union[
    AttnProcessor,
    AttnProcessor2_5,
    XFormersAttnProcessor,
    SlicedAttnProcessor,
    AttnAddedKVProcessor,
    SlicedAttnAddedKVProcessor,
    AttnAddedKVProcessor2_5,
    XFormersAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAXFormersAttnProcessor,
    LoRAAttnProcessor2_5,
    LoRAAttnAddedKVProcessor,
    CustomDiffusionAttnProcessor,
    CustomDiffusionXFormersAttnProcessor,
]

LORA_ATTENTION_PROCESSORS = (
    LoRAAttnProcessor,
    LoRAAttnProcessor2_5,
    LoRAXFormersAttnProcessor,
    LoRAAttnAddedKVProcessor,
)


class SpatialNorm(nn.Layer):
    """
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002
    """

    def __init__(
        self,
        f_channels,
        zq_channels,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(
            num_channels=f_channels,
            num_groups=32,
            epsilon=1e-6,
        )
        self.conv_y = nn.Conv2D(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2D(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f, zq):
        f_size = f.shape[-2:]
        zq = F.interpolate(zq, size=f_size, mode="nearest")
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f
