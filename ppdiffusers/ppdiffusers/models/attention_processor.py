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
import os
from importlib import import_module
from typing import Optional, Union

import paddle
import paddle.nn
import paddle.nn.functional as F
from paddle import einsum, nn

from ..utils import USE_PEFT_BACKEND, deprecate, logging
from ..utils.import_utils import is_ppxformers_available
from ..utils.paddle_utils import maybe_allow_in_graph
from .lora import LoRACompatibleLinear, LoRALinearLayer

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class Attention(nn.Layer):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the softmax computation to `float32`.
        cross_attention_norm (`str`, *optional*, defaults to `None`):
            The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.
        cross_attention_norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the group norm in the cross attention.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the group norm in the attention.
        spatial_norm_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the spatial normalization.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
        eps (`float`, *optional*, defaults to 1e-5):
            An additional value added to the denominator in group normalization that is used for numerical stability.
        rescale_output_factor (`float`, *optional*, defaults to 1.0):
            A factor to rescale the output by dividing it with this value.
        residual_connection (`bool`, *optional*, defaults to `False`):
            Set to `True` to add the residual connection to the output.
        _from_deprecated_attn_block (`bool`, *optional*, defaults to `False`):
            Set to `True` if the attention block is loaded from a deprecated state dict.
        processor (`AttnProcessor`, *optional*, defaults to `None`):
            The attention processor to use. If `None`, defaults to `AttnProcessor2_5` if `torch 2.x` is used and
            `AttnProcessor` otherwise.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        elementwise_affine: bool = True,
    ):
        super().__init__()

        # To prevent circular import.
        from .normalization import FP32LayerNorm, LpNorm, RMSNorm

        self.inner_dim = dim_head * heads
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
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

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            norm_elementwise_affine_kwargs = dict(weight_attr=elementwise_affine, bias_attr=elementwise_affine)
            self.norm_q = nn.LayerNorm(dim_head, epsilon=eps, **norm_elementwise_affine_kwargs)
            self.norm_k = nn.LayerNorm(dim_head, epsilon=eps, **norm_elementwise_affine_kwargs)
        elif qk_norm == "fp32_layer_norm":
            norm_elementwise_affine_kwargs = dict(weight_attr=False, bias_attr=False)
            self.norm_q = FP32LayerNorm(dim_head, epsilon=eps, **norm_elementwise_affine_kwargs)
            self.norm_k = FP32LayerNorm(dim_head, epsilon=eps, **norm_elementwise_affine_kwargs)
        elif qk_norm == "layer_norm_across_heads":
            # Lumina applies qk norm across all heads
            self.norm_q = nn.LayerNorm(dim_head * heads, epsilon=eps)
            self.norm_k = nn.LayerNorm(dim_head * kv_heads, epsilon=eps)
        elif qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, epsilon=eps)
            self.norm_k = RMSNorm(dim_head, epsilon=eps)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim_head * heads, epsilon=eps)
            self.norm_k = RMSNorm(dim_head * kv_heads, epsilon=eps)
        elif qk_norm == "l2":
            self.norm_q = LpNorm(p=2, dim=-1, epsilon=eps)
            self.norm_k = LpNorm(p=2, dim=-1, epsilon=eps)
        else:
            raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None,'layer_norm','fp32_layer_norm','rms_norm'")

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, epsilon=1e-5
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        if USE_PEFT_BACKEND:
            linear_cls = nn.Linear
        else:
            linear_cls = LoRACompatibleLinear

        self.to_q = linear_cls(query_dim, self.inner_dim, bias_attr=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = linear_cls(self.cross_attention_dim, self.inner_dim, bias_attr=bias)
            self.to_v = linear_cls(self.cross_attention_dim, self.inner_dim, bias_attr=bias)
        else:
            self.to_k = None
            self.to_v = None

        self.added_proj_bias = added_proj_bias
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = linear_cls(added_kv_proj_dim, self.inner_dim, bias_attr=self.added_proj_bias)
            self.add_v_proj = linear_cls(added_kv_proj_dim, self.inner_dim, bias_attr=self.added_proj_bias)
            if self.context_pre_only is not None:
                self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias_attr=self.added_proj_bias)
        else:
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None

        self.to_out = nn.LayerList([])
        self.to_out.append(linear_cls(self.inner_dim, query_dim, bias_attr=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, self.out_dim, bias_attr=out_bias)

        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "fp32_layer_norm":
                self.norm_added_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, epsilon=eps)
                self.norm_added_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, epsilon=eps)
            elif qk_norm == "rms_norm":
                self.norm_added_q = RMSNorm(dim_head, epsilon=eps)
                self.norm_added_k = RMSNorm(dim_head, epsilon=eps)
            elif qk_norm == "rms_norm_across_heads":
                # Wan applies qk norm across all heads
                # Wan also doesn't apply a q norm
                self.norm_added_q = None
                self.norm_added_k = RMSNorm(dim_head * kv_heads, epsilon=eps)
            else:
                raise ValueError(
                    f"unknown qk_norm: {qk_norm}. Should be one of `None,'layer_norm','fp32_layer_norm','rms_norm'`"
                )
        else:
            self.norm_added_q = None
            self.norm_added_k = None

        # set attention processor
        # We use the AttnProcessor2_5 by default when paddle 2.5 is used which uses
        # paddle.nn.functional.scaled_dot_product_attention_ for native Flash/memory_efficient_attention
        if processor is None:
            processor = AttnProcessor2_5() if is_ppxformers_available() else AttnProcessor()
        self.set_processor(processor)

    @property
    def dtype(self: nn.Layer) -> paddle.dtype:
        try:
            return next(self.named_parameters())[1].dtype
        except StopIteration:
            try:
                return next(self.named_buffers())[1].dtype
            except StopIteration:
                return self._dtype

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[str] = None
    ) -> None:
        r"""
        Set whether to use memory efficient attention from `xformers` or not.

        Args:
            use_memory_efficient_attention_xformers (`bool`):
                Whether to use memory efficient attention from `xformers` or not.
            attention_op (`Callable`, *optional*):
                The attention operation to use. Defaults to `None` which uses the default attention operation from
                `xformers`.
        """
        is_lora = hasattr(self, "processor") and isinstance(
            self.processor,
            LORA_ATTENTION_PROCESSORS,
        )
        is_custom_diffusion = hasattr(self, "processor") and isinstance(
            self.processor,
            (CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor, CustomDiffusionAttnProcessor2_5),
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
                    f"Memory efficient attention is currently not supported for LoRA or custom diffusion for attention processor type {self.processor}"
                )
            if not is_ppxformers_available():
                raise NotImplementedError(
                    "requires the scaled_dot_product_attention but your PaddlePaddle do not have this. Checkout the instructions on the installation page: https://www.paddlepaddle.org.cn/install/quick and follow the ones that match your environment."
                )

            if is_lora:
                # TODO (sayakpaul): should we throw a warning if someone wants to use the xformers
                # variant when using PT 2.0 now that we have LoRAAttnProcessor2_5?
                with paddle.dtype_guard(self.dtype):
                    # we must cast dtype
                    processor = LoRAXFormersAttnProcessor(
                        hidden_size=self.processor.hidden_size,
                        cross_attention_dim=self.processor.cross_attention_dim,
                        rank=self.processor.rank,
                        attention_op=attention_op,
                    )
                processor.load_dict(self.processor.state_dict())
            elif is_custom_diffusion:
                with paddle.dtype_guard(self.dtype):
                    # we must cast dtype
                    processor = CustomDiffusionXFormersAttnProcessor(
                        train_kv=self.processor.train_kv,
                        train_q_out=self.processor.train_q_out,
                        hidden_size=self.processor.hidden_size,
                        cross_attention_dim=self.processor.cross_attention_dim,
                        attention_op=attention_op,
                    )
                processor.load_dict(self.processor.state_dict())
            elif is_added_kv_processor:
                # TODO(Patrick, Suraj, William) - currently xformers doesn't work for UnCLIP
                # which uses this type of cross attention ONLY because the attention mask of format
                # [0, ..., -10.000, ..., 0, ...,] is not supported
                # throw warning
                logger.info(
                    "Memory efficient attention with `ppxformers` might currently not work correctly if an attention mask is required for the attention operation."
                )
                processor = XFormersAttnAddedKVProcessor(attention_op=attention_op)
            else:
                processor = XFormersAttnProcessor(attention_op=attention_op)
        else:
            if is_lora:
                attn_processor_class = LoRAAttnProcessor2_5 if is_ppxformers_available() else LoRAAttnProcessor
                with paddle.dtype_guard(self.dtype):
                    # we must cast dtype
                    processor = attn_processor_class(
                        hidden_size=self.processor.hidden_size,
                        cross_attention_dim=self.processor.cross_attention_dim,
                        rank=self.processor.rank,
                    )
                processor.load_dict(self.processor.state_dict())
            elif is_custom_diffusion:
                attn_processor_class = (
                    CustomDiffusionAttnProcessor2_5 if is_ppxformers_available() else CustomDiffusionAttnProcessor
                )
                with paddle.dtype_guard(self.dtype):
                    # we must cast dtype
                    processor = attn_processor_class(
                        train_kv=self.processor.train_kv,
                        train_q_out=self.processor.train_q_out,
                        hidden_size=self.processor.hidden_size,
                        cross_attention_dim=self.processor.cross_attention_dim,
                    )
                processor.load_dict(self.processor.state_dict())
            else:
                # set attention processor
                # We use the AttnProcessor2_5 by default when paddle 2.5 is used which uses
                # paddle.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
                processor = AttnProcessor2_5() if is_ppxformers_available() else AttnProcessor()

        self.set_processor(processor)

    def set_attention_slice(self, slice_size: int) -> None:
        r"""
        Set the slice size for attention computation.

        Args:
            slice_size (`int`):
                The slice size for attention computation.
        """
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        if slice_size is not None and self.added_kv_proj_dim is not None:
            processor = SlicedAttnAddedKVProcessor(slice_size)
        elif slice_size is not None:
            processor = SlicedAttnProcessor(slice_size)
        elif self.added_kv_proj_dim is not None:
            processor = AttnAddedKVProcessor()
        else:
            # set attention processor
            # We use the AttnProcessor2_5 by default when paddle 2.5 is used which uses
            # paddle.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
            processor = AttnProcessor2_5() if is_ppxformers_available() else AttnProcessor()

        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor", _remove_lora: bool = False) -> None:
        r"""
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
            _remove_lora (`bool`, *optional*, defaults to `False`):
                Set to `True` to remove LoRA layers from the model.
        """
        if not USE_PEFT_BACKEND and hasattr(self, "processor") and _remove_lora and self.to_q.lora_layer is not None:
            deprecate(
                "set_processor to offload LoRA",
                "0.45.0",
                "In detail, removing LoRA layers via calling `set_default_attn_processor` is deprecated. Please make sure to call `pipe.unload_lora_weights()` instead.",
            )
            # TODO(Patrick, Sayak) - this can be deprecated once PEFT LoRA integration is complete
            # We need to remove all LoRA layers
            # Don't forget to remove ALL `_remove_lora` from the codebase
            for module in self.sublayers(include_self=True):
                if hasattr(module, "set_lora_layer"):
                    module.set_lora_layer(None)

        # if current processor is in `self._sub_layers` and if passed `processor` is not, we need to
        # pop `processor` from `self._sub_layers`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, paddle.nn.Layer)
            and not isinstance(processor, paddle.nn.Layer)
        ):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._sub_layers.pop("processor")

        self.processor = processor

    def get_processor(self, return_deprecated_lora: bool = False) -> "AttentionProcessor":
        r"""
        Get the attention processor in use.

        Args:
            return_deprecated_lora (`bool`, *optional*, defaults to `False`):
                Set to `True` to return the deprecated LoRA attention processor.

        Returns:
            "AttentionProcessor": The attention processor in use.
        """
        if not return_deprecated_lora:
            return self.processor

        # TODO(Sayak, Patrick). The rest of the function is needed to ensure backwards compatible
        # serialization format for LoRA Attention Processors. It should be deleted once the integration
        # with PEFT is completed.
        is_lora_activated = {
            name: module.lora_layer is not None
            for name, module in self.named_sublayers(include_self=True)
            if hasattr(module, "lora_layer")
        }

        # 1. if no layer has a LoRA activated we can return the processor as usual
        if not any(is_lora_activated.values()):
            return self.processor

        # If doesn't apply LoRA do `add_k_proj` or `add_v_proj`
        is_lora_activated.pop("add_k_proj", None)
        is_lora_activated.pop("add_v_proj", None)
        # 2. else it is not possible that only some layers have LoRA activated
        if not all(is_lora_activated.values()):
            raise ValueError(
                f"Make sure that either all layers or no layers have LoRA activated, but have {is_lora_activated}"
            )

        # 3. And we need to merge the current LoRA layers into the corresponding LoRA attention processor
        non_lora_processor_cls_name = self.processor.__class__.__name__
        lora_processor_cls = getattr(import_module(__name__), "LoRA" + non_lora_processor_cls_name)

        hidden_size = self.inner_dim

        # now create a LoRA attention processor from the LoRA layers
        if lora_processor_cls in [LoRAAttnProcessor, LoRAAttnProcessor2_5, LoRAXFormersAttnProcessor]:
            kwargs = {
                "cross_attention_dim": self.cross_attention_dim,
                "rank": self.to_q.lora_layer.rank,
                "network_alpha": self.to_q.lora_layer.network_alpha,
                "q_rank": self.to_q.lora_layer.rank,
                "q_hidden_size": self.to_q.lora_layer.out_features,
                "k_rank": self.to_k.lora_layer.rank,
                "k_hidden_size": self.to_k.lora_layer.out_features,
                "v_rank": self.to_v.lora_layer.rank,
                "v_hidden_size": self.to_v.lora_layer.out_features,
                "out_rank": self.to_out[0].lora_layer.rank,
                "out_hidden_size": self.to_out[0].lora_layer.out_features,
            }

            if hasattr(self.processor, "attention_op"):
                kwargs["attention_op"] = self.processor.attention_op
            with paddle.dtype_guard(self.dtype):
                lora_processor = lora_processor_cls(hidden_size, **kwargs)
            lora_processor.to_q_lora.load_dict(self.to_q.lora_layer.state_dict())
            lora_processor.to_k_lora.load_dict(self.to_k.lora_layer.state_dict())
            lora_processor.to_v_lora.load_dict(self.to_v.lora_layer.state_dict())
            lora_processor.to_out_lora.load_dict(self.to_out[0].lora_layer.state_dict())
        elif lora_processor_cls == LoRAAttnAddedKVProcessor:
            with paddle.dtype_guard(self.dtype):
                lora_processor = lora_processor_cls(
                    hidden_size,
                    cross_attention_dim=self.add_k_proj.weight.shape[0],
                    rank=self.to_q.lora_layer.rank,
                    network_alpha=self.to_q.lora_layer.network_alpha,
                )
            lora_processor.to_q_lora.load_dict(self.to_q.lora_layer.state_dict())
            lora_processor.to_k_lora.load_dict(self.to_k.lora_layer.state_dict())
            lora_processor.to_v_lora.load_dict(self.to_v.lora_layer.state_dict())
            lora_processor.to_out_lora.load_dict(self.to_out[0].lora_layer.state_dict())

            # only save if used
            if self.add_k_proj.lora_layer is not None:
                lora_processor.add_k_proj_lora.load_dict(self.add_k_proj.lora_layer.state_dict())
                lora_processor.add_v_proj_lora.load_dict(self.add_v_proj.lora_layer.state_dict())
            else:
                lora_processor.add_k_proj_lora = None
                lora_processor.add_v_proj_lora = None
        else:
            raise ValueError(f"{lora_processor_cls} does not exist.")

        return lora_processor

    def forward(
        self,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        **cross_attention_kwargs,
    ) -> paddle.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`paddle.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`paddle.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`paddle.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `paddle.Tensor`: The output of the attention layer.
        """
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

    def batch_to_head_dim(self, tensor: paddle.Tensor, in_dim: int = 4, transpose: bool = True) -> paddle.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`paddle.Tensor`): The tensor to reshape.

        Returns:
            `paddle.Tensor`: The reshaped tensor.
        """
        if in_dim == 3:
            head_size = self.heads
            batch_size, seq_len, dim = tensor.shape
            tensor = tensor.reshape([batch_size // head_size, head_size, seq_len, dim])
        if transpose:
            tensor = tensor.transpose([0, 2, 1, 3])
        tensor = tensor.reshape([0, 0, tensor.shape[2] * tensor.shape[3]])
        return tensor

    def head_to_batch_dim(self, tensor: paddle.Tensor, out_dim: int = 4, transpose: bool = True) -> paddle.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`paddle.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `paddle.Tensor`: The reshaped tensor.
        """
        tensor = tensor.reshape([0, 0, self.heads, self.head_dim])
        if transpose or out_dim == 3:
            tensor = tensor.transpose([0, 2, 1, 3])
        if out_dim == 3:
            tensor = tensor.flatten(0, 1)
        return tensor

    def get_attention_scores(
        self, query: paddle.Tensor, key: paddle.Tensor, attention_mask: paddle.Tensor = None
    ) -> paddle.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`paddle.Tensor`): The query tensor.
            key (`paddle.Tensor`): The key tensor.
            attention_mask (`paddle.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `paddle.Tensor`: The attention probabilities/scores.
        """
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

    def prepare_attention_mask(
        self,
        attention_mask: paddle.Tensor,
        target_length: int,
        batch_size: int,
        out_dim: int = 4,
        transpose: bool = True,
    ) -> paddle.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`paddle.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `paddle.Tensor`: The prepared attention mask.
        """
        num_heads = self.heads
        if attention_mask is None:
            return attention_mask

        ori_type = attention_mask.dtype
        attention_mask = attention_mask.to(paddle.float32)

        current_length = attention_mask.shape[-1]
        if current_length != target_length:
            # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
            #       we want to instead pad by (0, remaining_length), where remaining_length is:
            #       remaining_length: int = target_length - current_length
            # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0, data_format="NCL")

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * num_heads:
                attention_mask = attention_mask.repeat_interleave(num_heads, axis=0)
        elif out_dim == 4:
            if attention_mask.shape[0] < batch_size * num_heads:
                attention_mask = attention_mask.repeat_interleave(num_heads, axis=0)
            attention_mask = attention_mask.reshape([batch_size, num_heads, -1, attention_mask.shape[-1]])

        # do not need transpose
        # if attention_mask.ndim == 4:
        #     if not transpose:
        #         attention_mask = attention_mask.transpose([0, 2, 1, 3])
        return attention_mask.to(ori_type)

    def norm_encoder_hidden_states(self, encoder_hidden_states: paddle.Tensor) -> paddle.Tensor:
        r"""
        Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
        `Attention` class.

        Args:
            encoder_hidden_states (`paddle.Tensor`): Hidden states of the encoder.

        Returns:
            `paddle.Tensor`: The normalized encoder hidden states.
        """
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
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        temb: Optional[paddle.Tensor] = None,
        scale: float = 1.0,
        **kwargs,
    ) -> paddle.Tensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

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

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = paddle.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
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
        train_kv: bool = True,
        train_q_out: bool = True,
        hidden_size: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        dropout: float = 0.0,
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
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> paddle.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if self.train_q_out:
            query = self.to_q_custom_diffusion(hidden_states).cast(attn.to_q.weight.dtype)
        else:
            query = attn.to_q(hidden_states.cast(attn.to_q.weight.dtype))

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
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        scale: float = 1.0,
        **kwargs,
    ) -> paddle.Tensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

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

        query = attn.to_q(hidden_states, *args)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states, *args)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states, *args)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)

        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states, *args)
            value = attn.to_v(hidden_states, *args)
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
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose([0, 2, 1]).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states


class JointAttnProcessor2_5:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_5 requires Paddle version >= 2.5, to use it, please upgrade Paddle.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        *args,
        **kwargs,
    ) -> paddle.Tensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.reshape([batch_size, -1, attn.heads, head_dim])
        key = key.reshape([batch_size, -1, attn.heads, head_dim])
        value = value.reshape([batch_size, -1, attn.heads, head_dim])

        if attn.norm_q is not None:
            query = attn.norm_q(query, begin_norm_axis=3)
        if attn.norm_k is not None:
            key = attn.norm_k(key, begin_norm_axis=3)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.reshape(
                [batch_size, -1, attn.heads, head_dim]
            )
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.reshape(
                [batch_size, -1, attn.heads, head_dim]
            )
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.reshape(
                [batch_size, -1, attn.heads, head_dim]
            )

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj, begin_norm_axis=3
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj, begin_norm_axis=3)

            query = paddle.concat([query, encoder_hidden_states_query_proj], axis=1)
            key = paddle.concat([key, encoder_hidden_states_key_proj], axis=1)
            value = paddle.concat([value, encoder_hidden_states_value_proj], axis=1)

        hidden_states = hidden_states = F.scaled_dot_product_attention_(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.reshape([batch_size, -1, attn.heads * head_dim])
        hidden_states = hidden_states.astype(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class FusedJointAttnProcessor2_5:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, attention_op=None):
        assert attention_op in [None, "math", "auto", "flash", "cutlass", "memory_efficient"]
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_5 requires Paddle >= 2.5, to use it, please upgrade Paddle.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        *args,
        **kwargs,
    ) -> paddle.Tensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.reshape([batch_size, channel, height * width]).transpose([0, 2, 1])
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.reshape([batch_size, channel, height * width]).transpose(
                [0, 2, 1]
            )

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        qkv = attn.to_qkv(hidden_states)
        split_size = qkv.shape[-1] // 3
        query, key, value = paddle.split(qkv, num_or_sections=[split_size, split_size, split_size], axis=-1)

        # `context` projections.
        encoder_qkv = attn.to_added_qkv(encoder_hidden_states)
        split_size = encoder_qkv.shape[-1] // 3
        (
            encoder_hidden_states_query_proj,
            encoder_hidden_states_key_proj,
            encoder_hidden_states_value_proj,
        ) = paddle.split(encoder_qkv, num_or_sections=[split_size, split_size, split_size], axis=-1)

        # attention
        query = paddle.concat([query, encoder_hidden_states_query_proj], axis=1)
        key = paddle.concat([key, encoder_hidden_states_key_proj], axis=1)
        value = paddle.concat([value, encoder_hidden_states_value_proj], axis=1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.reshape([batch_size, -1, attn.heads, head_dim])
        key = key.reshape([batch_size, -1, attn.heads, head_dim])
        value = value.reshape([batch_size, -1, attn.heads, head_dim])

        hidden_states = hidden_states = F.scaled_dot_product_attention_(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.reshape([batch_size, -1, attn.heads * head_dim])
        hidden_states = hidden_states.astype(query.dtype)

        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose([0, 1, 3, 2]).reshape([batch_size, channel, height, width])
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose([0, 1, 3, 2]).reshape(
                [batch_size, channel, height, width]
            )

        return hidden_states, encoder_hidden_states


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
        assert attention_op in [None, "math", "auto", "flash", "cutlass", "memory_efficient"]
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        **cross_attention_kwarg,
    ) -> paddle.Tensor:
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

        # if transpose = False, query's shape will be [batch_size, seq_len, num_head, head_dim]
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
        hidden_states = hidden_states.cast(query.dtype)
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
        attention_op (`str`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[str] = None):
        assert attention_op in [None, "math", "auto", "flash", "cutlass", "memory_efficient"]
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        temb: Optional[paddle.Tensor] = None,
        scale: float = 1.0,
        **kwargs,
    ) -> paddle.Tensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.reshape([batch_size, channel, height * width]).transpose([0, 2, 1])

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch,            1, heads, key_tokens] ->
            #   [batch, query_tokens, heads, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch, query_tokens, heads, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand([-1, -1, query_tokens, -1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose([0, 2, 1])).transpose([0, 2, 1])

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query, transpose=False)
        key = attn.head_to_batch_dim(key, transpose=False)
        value = attn.head_to_batch_dim(value, transpose=False)

        #  adapt the scaled_dot_product_attention_ when attention_mask is a bool tensor
        if attention_mask is not None and attention_mask.dtype == paddle.bool:
            L, S = query.shape[1], key.shape[1]
            attention_mask_tmp = paddle.zeros([1, 1, L, S], dtype=query.dtype)
            attention_mask_tmp = attention_mask_tmp.masked_fill(attention_mask.logical_not(), float("-inf"))
            attention_mask = attention_mask_tmp

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

        hidden_states = hidden_states.cast(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states, transpose=False)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
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
        train_kv: bool = True,
        train_q_out: bool = False,
        hidden_size: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        dropout: float = 0.0,
        attention_op: Optional[str] = None,
    ):
        super().__init__()
        assert attention_op in [None, "math", "auto", "flash", "cutlass", "memory_efficient"]
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
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> paddle.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if self.train_q_out:
            query = self.to_q_custom_diffusion(hidden_states).cast(attn.to_q.weight.dtype)
        else:
            query = attn.to_q(hidden_states.cast(attn.to_q.weight.dtype))

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
        hidden_states = hidden_states.cast(query.dtype)
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

    def __init__(self, slice_size: int):
        self.slice_size = slice_size

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> paddle.Tensor:
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
        query = attn.head_to_batch_dim(query, out_dim=3)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key, out_dim=3)
        value = attn.head_to_batch_dim(value, out_dim=3)

        batch_size_attention, query_tokens, _ = query.shape
        hidden_states = paddle.zeros((batch_size_attention, query_tokens, attn.head_dim), dtype=query.dtype)

        for i in range(batch_size_attention // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)

            attn_slice = paddle.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        hidden_states = attn.batch_to_head_dim(hidden_states, in_dim=3)

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
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        temb: Optional[paddle.Tensor] = None,
        **kwargs,
    ) -> paddle.Tensor:
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
        query = attn.head_to_batch_dim(query, out_dim=3)

        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj, out_dim=3)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj, out_dim=3)

        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key, out_dim=3)
            value = attn.head_to_batch_dim(value, out_dim=3)
            key = paddle.concat([encoder_hidden_states_key_proj, key], axis=1)
            value = paddle.concat([encoder_hidden_states_value_proj, value], axis=1)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj

        batch_size_attention, query_tokens, _ = query.shape
        hidden_states = paddle.zeros((batch_size_attention, query_tokens, attn.head_dim), dtype=query.dtype)

        for i in range(batch_size_attention // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)

            attn_slice = paddle.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        hidden_states = attn.batch_to_head_dim(hidden_states, in_dim=3)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose([0, 2, 1]).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states


class SpatialNorm(nn.Layer):
    """
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=32, epsilon=1e-6)
        self.conv_y = nn.Conv2D(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2D(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f: paddle.Tensor, zq: paddle.Tensor) -> paddle.Tensor:
        f_size = f.shape[-2:]
        zq = F.interpolate(zq, size=f_size, mode="nearest")
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


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
        kwargs (`dict`):
            Additional keyword arguments to pass to the `LoRALinearLayer` layers.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        rank: int = 4,
        network_alpha: Optional[int] = None,
        **kwargs,
    ):
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

    def __call__(self, attn: Attention, hidden_states: paddle.Tensor, *args, **kwargs) -> paddle.Tensor:
        self_cls_name = self.__class__.__name__
        deprecate(
            self_cls_name,
            "0.45.0",
            (
                f"Make sure use {self_cls_name[4:]} instead by setting"
                "LoRA layers to `self.{to_q,to_k,to_v,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
                " `LoraLoaderMixin.load_lora_weights`"
            ),
        )
        attn.to_q.lora_layer = self.to_q_lora
        attn.to_k.lora_layer = self.to_k_lora
        attn.to_v.lora_layer = self.to_v_lora
        attn.to_out[0].lora_layer = self.to_out_lora

        attn._sub_layers.pop("processor")
        attn.processor = AttnProcessor()
        return attn.processor(attn, hidden_states, *args, **kwargs)


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
        kwargs (`dict`):
            Additional keyword arguments to pass to the `LoRALinearLayer` layers.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        rank: int = 4,
        attention_op: Optional[str] = None,
        network_alpha: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        assert attention_op in [None, "math", "auto", "flash", "cutlass", "memory_efficient"]
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

    def __call__(self, attn: Attention, hidden_states: paddle.Tensor, *args, **kwargs) -> paddle.Tensor:
        self_cls_name = self.__class__.__name__
        deprecate(
            self_cls_name,
            "0.45.0",
            (
                f"Make sure use {self_cls_name[4:]} instead by setting"
                "LoRA layers to `self.{to_q,to_k,to_v,add_k_proj,add_v_proj,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
                " `LoraLoaderMixin.load_lora_weights`"
            ),
        )
        attn.to_q.lora_layer = self.to_q_lora
        attn.to_k.lora_layer = self.to_k_lora
        attn.to_v.lora_layer = self.to_v_lora
        attn.to_out[0].lora_layer = self.to_out_lora

        attn._sub_layers.pop("processor")
        attn.processor = XFormersAttnProcessor()
        return attn.processor(attn, hidden_states, *args, **kwargs)


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
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
        kwargs (`dict`):
            Additional keyword arguments to pass to the `LoRALinearLayer` layers.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        rank: int = 4,
        network_alpha: Optional[int] = None,
    ):
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

    def __call__(self, attn: Attention, hidden_states: paddle.Tensor, *args, **kwargs) -> paddle.Tensor:
        self_cls_name = self.__class__.__name__
        deprecate(
            self_cls_name,
            "0.45.0",
            (
                f"Make sure use {self_cls_name[4:]} instead by setting"
                "LoRA layers to `self.{to_q,to_k,to_v,add_k_proj,add_v_proj,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
                " `LoraLoaderMixin.load_lora_weights`"
            ),
        )
        attn.to_q.lora_layer = self.to_q_lora
        attn.to_k.lora_layer = self.to_k_lora
        attn.to_v.lora_layer = self.to_v_lora
        attn.to_out[0].lora_layer = self.to_out_lora

        attn._sub_layers.pop("processor")
        attn.processor = AttnAddedKVProcessor()
        return attn.processor(attn, hidden_states, *args, **kwargs)


class IPAdapterAttnProcessor(nn.Layer):
    r"""
    Attention processor for IP-Adapter.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        num_tokens (`int`, defaults to 4):
            The context length of the image features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, num_tokens=4, scale=1.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias_attr=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias_attr=False)

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        temb: Optional[paddle.Tensor] = None,
        scale: float = 1.0,
        **kwargs,
    ):
        if scale != 1.0:
            logger.warning("`scale` of IPAttnProcessor should be set with `set_ip_adapter_scale`.")
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

        # split hidden states
        end_pos = encoder_hidden_states.shape[1] - self.num_tokens
        encoder_hidden_states, ip_hidden_states = (
            encoder_hidden_states[:, :end_pos, :],
            encoder_hidden_states[:, end_pos:, :],
        )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = paddle.matmul(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key)
        ip_value = attn.head_to_batch_dim(ip_value)

        ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
        ip_hidden_states = paddle.matmul(ip_attention_probs, ip_value)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states)

        hidden_states = hidden_states + self.scale * ip_hidden_states

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


class IPAdapterXFormersAttnProcessor(nn.Layer):
    r"""
    Attention processor for IP-Adapter for using xFormers.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        num_tokens (`int`, defaults to 4):
            The context length of the image features.
        scale (`float`, defaults to 1.0):
            the weight scale of image prompt.
        attention_op (`str`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(
        self, hidden_size, cross_attention_dim=None, num_tokens=4, scale=1.0, attention_op: Optional[str] = None
    ):
        super().__init__()
        assert attention_op in [None, "math", "auto", "flash", "cutlass", "memory_efficient"]
        self.attention_op = attention_op
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens
        self.scale = scale

        self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias_attr=False)
        self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias_attr=False)

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        temb: Optional[paddle.Tensor] = None,
        scale: float = 1.0,
        **kwargs,
    ):
        if scale != 1.0:
            logger.warning("`scale` of IPAttnProcessor should be set by `set_ip_adapter_scale`.")
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

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
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

        # split hidden states
        end_pos = encoder_hidden_states.shape[1] - self.num_tokens
        encoder_hidden_states, ip_hidden_states = (
            encoder_hidden_states[:, :end_pos, :],
            encoder_hidden_states[:, end_pos:, :],
        )

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

        hidden_states = hidden_states.cast(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states, transpose=False)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = attn.head_to_batch_dim(ip_key, transpose=False)
        ip_value = attn.head_to_batch_dim(ip_value, transpose=False)

        ip_hidden_states = F.scaled_dot_product_attention_(
            query,
            ip_key,
            ip_value,
            attn_mask=None,
            scale=attn.scale,
            dropout_p=0.0,
            training=attn.training,
            attention_op=self.attention_op,
        )

        ip_hidden_states = ip_hidden_states.cast(query.dtype)
        ip_hidden_states = attn.batch_to_head_dim(ip_hidden_states, transpose=False)

        hidden_states = hidden_states + self.scale * ip_hidden_states

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


# TODO(Yiyi): This class should not exist, we can replace it with a normal attention processor I believe
# this way torch.compile and co. will work as well
class Kandi3AttnProcessor:
    r"""
    Default kandinsky3 processor for performing attention-related computations.
    """

    @staticmethod
    def _reshape(hid_states, h):
        b, n, f = hid_states.shape
        d = f // h
        return hid_states.unsqueeze(-1).reshape([b, n, h, d]).transpose([0, 2, 1, 3])

    def __call__(
        self,
        attn: Attention,
        x,
        context,
        context_mask=None,
        **kwargs,
    ):
        query = self._reshape(attn.to_q(x), h=attn.heads)  # num_heads
        key = self._reshape(attn.to_k(context), h=attn.heads)  # num_heads
        value = self._reshape(attn.to_v(context), h=attn.heads)  # num_heads

        attention_matrix = einsum("b h i d, b h j d -> b h i j", query, key)

        if context_mask is not None:
            max_neg_value = -paddle.finfo(attention_matrix.dtype).max
            context_mask = context_mask.unsqueeze(1).unsqueeze(1)
            attention_matrix = paddle.where(~(context_mask != 0), attention_matrix, max_neg_value)
        attention_matrix = F.softmax(attention_matrix * attn.scale, axis=-1)

        out = einsum("b h i j, b h j d -> b h i d", attention_matrix, value)
        out = out.transpose([0, 2, 1, 3]).reshape([out.shape[0], out.shape[2], -1])
        out = attn.to_out[0](out)
        return out


class FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        image_rotary_emb: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.reshape([batch_size, -1, attn.heads, head_dim]).transpose([0, 2, 1, 3])
        key = key.reshape([batch_size, -1, attn.heads, head_dim]).transpose([0, 2, 1, 3])
        value = value.reshape([batch_size, -1, attn.heads, head_dim]).transpose([0, 2, 1, 3])

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.reshape(
                [batch_size, -1, attn.heads, head_dim]
            ).transpose([0, 2, 1, 3])
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.reshape(
                [batch_size, -1, attn.heads, head_dim]
            ).transpose([0, 2, 1, 3])
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.reshape(
                [batch_size, -1, attn.heads, head_dim]
            ).transpose([0, 2, 1, 3])

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = paddle.concat([encoder_hidden_states_query_proj, query], axis=2)
            key = paddle.concat([encoder_hidden_states_key_proj, key], axis=2)
            value = paddle.concat([encoder_hidden_states_value_proj, value], axis=2)

        if image_rotary_emb is not None:
            from .embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention_(
            query.transpose([0, 2, 1, 3]),
            key.transpose([0, 2, 1, 3]),
            value.transpose([0, 2, 1, 3]),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.reshape([batch_size, -1, attn.heads * head_dim])
        hidden_states = hidden_states.astype(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class FusedFluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FusedFluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        image_rotary_emb: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        qkv = attn.to_qkv(hidden_states)
        # split_size = qkv.shape[-1] // 3
        query, key, value = paddle.split(qkv, 3, axis=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.reshape([batch_size, -1, attn.heads, head_dim])
        key = key.reshape([batch_size, -1, attn.heads, head_dim])
        value = value.reshape([batch_size, -1, attn.heads, head_dim])

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_qkv = attn.to_added_qkv(encoder_hidden_states)
            # split_size = encoder_qkv.shape[-1] // 3
            (
                encoder_hidden_states_query_proj,
                encoder_hidden_states_key_proj,
                encoder_hidden_states_value_proj,
            ) = paddle.split(encoder_qkv, 3, dim=-1)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.reshape(
                [batch_size, -1, attn.heads, head_dim]
            )
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.reshape(
                [batch_size, -1, attn.heads, head_dim]
            )
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.reshape(
                [batch_size, -1, attn.heads, head_dim]
            )

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = paddle.concat([encoder_hidden_states_query_proj, query], axis=1)
            key = paddle.concat([encoder_hidden_states_key_proj, key], axis=1)
            value = paddle.concat([encoder_hidden_states_value_proj, value], axis=1)

        if image_rotary_emb is not None:
            from .embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention_(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.reshape([batch_size, -1, attn.heads * head_dim])
        hidden_states = hidden_states.astype(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class CogVideoXAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        image_rotary_emb: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        text_seq_length = encoder_hidden_states.shape[1]

        hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.reshape([batch_size, attn.heads, -1, attention_mask.shape[-1]])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.reshape([batch_size, -1, attn.heads, head_dim]).transpose([0, 2, 1, 3])
        key = key.reshape([batch_size, -1, attn.heads, head_dim]).transpose([0, 2, 1, 3])
        value = value.reshape([batch_size, -1, attn.heads, head_dim])
        inference_optimize = os.getenv("INFERENCE_OPTIMIZE") == "True"
        if inference_optimize:
            import paddlemix

            if (
                attn.norm_q is not None
                and attn.norm_k is not None
                and image_rotary_emb is not None
                and not attn.is_cross_attention
            ):
                text_seq_length_tensor = paddle.empty([text_seq_length])
                query, key = paddlemix.triton_ops.ln_partial_rotary_emb(
                    query,
                    key,
                    text_seq_length_tensor,
                    image_rotary_emb[0],
                    image_rotary_emb[1],
                    attn.norm_q.weight.cuda(),
                    attn.norm_q.bias.cuda(),
                    attn.norm_k.weight.cuda(),
                    attn.norm_k.bias.cuda(),
                    norm_eps=1e-5,
                )
            elif attn.norm_q is None or attn.norm_k is None:
                if attn.norm_q is not None:
                    query = attn.norm_q(query)
                if attn.norm_k is not None:
                    key = attn.norm_k(key)
                if image_rotary_emb is not None and not attn.is_cross_attention:
                    text_seq_length_tensor = paddle.empty([text_seq_length])
                    query, key = paddlemix.triton_ops.partial_rotary_emb(
                        query, key, text_seq_length_tensor, image_rotary_emb[0], image_rotary_emb[1]
                    )
        else:
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            # Apply RoPE if needed
            if image_rotary_emb is not None:
                from .embeddings import apply_rotary_emb

                query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
                if not attn.is_cross_attention:
                    key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        # NOTE: There is diff between paddle's and torch's sdpa
        # paddle needs input: [batch_size, seq_len, num_heads, head_dim]
        # torch needs input: [batch_size, num_heads, seq_len, head_dim]
        hidden_states = F.scaled_dot_product_attention_(
            query.transpose([0, 2, 1, 3]),
            key.transpose([0, 2, 1, 3]),
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )

        hidden_states = hidden_states.reshape([batch_size, -1, attn.heads * head_dim])

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.shape[1] - text_seq_length], axis=1
        )

        return hidden_states, encoder_hidden_states


class FusedCogVideoXAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: paddle.Tensor,
        encoder_hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        image_rotary_emb: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        text_seq_length = encoder_hidden_states.shape[1]

        hidden_states = paddle.concat([encoder_hidden_states, hidden_states], axis=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.reshape([batch_size, attn.heads, -1, attention_mask.shape[-1]])

        qkv = attn.to_qkv(hidden_states)
        split_size = qkv.shape[-1] // 3
        query, key, value = paddle.split(qkv, split_size, axis=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.reshape([batch_size, -1, attn.heads, head_dim]).premute([0, 2, 1, 3])
        key = key.reshape([batch_size, -1, attn.heads, head_dim]).premute([0, 2, 1, 3])
        value = value.reshape([batch_size, -1, attn.heads, head_dim]).premute([0, 2, 1, 3])

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from .embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.premute([0, 2, 1, 3]).reshape([batch_size, -1, attn.heads * head_dim])

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.shape[1] - text_seq_length], axis=1
        )
        return hidden_states, encoder_hidden_states


LoRAAttnProcessor2_5 = LoRAXFormersAttnProcessor
AttnAddedKVProcessor2_5 = XFormersAttnAddedKVProcessor
AttnProcessor2_5 = XFormersAttnProcessor
IPAdapterAttnProcessor2_5 = IPAdapterXFormersAttnProcessor
CustomDiffusionAttnProcessor2_5 = CustomDiffusionXFormersAttnProcessor

LORA_ATTENTION_PROCESSORS = (
    LoRAAttnProcessor,
    LoRAAttnProcessor2_5,
    LoRAXFormersAttnProcessor,
    LoRAAttnAddedKVProcessor,
)

ADDED_KV_ATTENTION_PROCESSORS = (
    AttnAddedKVProcessor,
    SlicedAttnAddedKVProcessor,
    AttnAddedKVProcessor2_5,
    XFormersAttnAddedKVProcessor,
    LoRAAttnAddedKVProcessor,
)

CROSS_ATTENTION_PROCESSORS = (
    AttnProcessor,
    AttnProcessor2_5,
    XFormersAttnProcessor,
    SlicedAttnProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_5,
    LoRAXFormersAttnProcessor,
    IPAdapterAttnProcessor,
    IPAdapterXFormersAttnProcessor,
    IPAdapterAttnProcessor2_5,
    Kandi3AttnProcessor,
)

AttentionProcessor = Union[
    AttnProcessor,
    AttnProcessor2_5,
    XFormersAttnProcessor,
    SlicedAttnProcessor,
    AttnAddedKVProcessor,
    SlicedAttnAddedKVProcessor,
    AttnAddedKVProcessor2_5,
    XFormersAttnAddedKVProcessor,
    CustomDiffusionAttnProcessor,
    CustomDiffusionXFormersAttnProcessor,
    CustomDiffusionAttnProcessor2_5,
    FluxAttnProcessor2_0,
    FusedFluxAttnProcessor2_0,
    # deprecated
    LoRAAttnProcessor,
    LoRAAttnProcessor2_5,
    LoRAXFormersAttnProcessor,
    LoRAAttnAddedKVProcessor,
]
