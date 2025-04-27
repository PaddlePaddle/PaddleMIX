# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import functools
import inspect
import os
from functools import partial
from typing import Callable, Optional, Tuple, Union

import paddle
import paddle.nn as nn
from paddle.distributed.fleet.utils import recompute
from paddlenlp.transformers.configuration_utils import (
    PretrainedConfig as PPNLPPretrainedConfig,
)
from paddlenlp.transformers.model_utils import PretrainedModel as PPNLPPretrainedModel
from paddlenlp.utils.log import logger as ppnlp_logger

from ppdiffusers.utils import (
    is_safetensors_available,
    is_torch_available,
    recompute_use_reentrant,
)

from ..utils import logging
from .peft_utils import PeftAdapterMixin

logger = logging.get_logger(__name__)

if is_safetensors_available():
    from safetensors.numpy import save_file as np_safe_save_file

    if is_torch_available():
        import torch
        from safetensors.torch import save_file as torch_safe_save_file


ALL_LAYERNORM_LAYERS = [nn.LayerNorm]


class ModuleUtilsMixin:
    """
    A few utilities for `nn.Layer`, to be used as a mixin.
    """

    @property
    def device(self):
        """
        `paddle.place`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        try:
            return next(self.named_parameters())[1].place
        except StopIteration:
            try:
                return next(self.named_buffers())[1].place
            except StopIteration:
                return paddle.get_device()

    @property
    def dtype(self) -> paddle.dtype:
        """
        `paddle.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        try:
            return next(self.named_parameters())[1].dtype
        except StopIteration:
            try:
                return next(self.named_buffers())[1].dtype
            except StopIteration:
                return self._dtype

    def invert_attention_mask(self, encoder_attention_mask: paddle.Tensor) -> paddle.Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`paddle.Tensor`): An attention mask.

        Returns:
            `paddle.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.ndim == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.ndim == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.cast(dtype=self.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * paddle.finfo(self.dtype).min

        return encoder_extended_attention_mask

    @staticmethod
    def create_extended_attention_mask_for_decoder(input_shape, attention_mask):
        batch_size, seq_length = input_shape
        seq_ids = paddle.arange(seq_length)
        causal_mask = seq_ids[None, None, :].tile([batch_size, seq_length, 1]) <= seq_ids[None, :, None]
        # in case past_key_values are used we need to add a prefix ones mask to the causal mask
        # causal and attention masks must have same type with pytorch version < 1.3
        causal_mask = causal_mask.cast(dtype=attention_mask.dtype)

        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = paddle.concat(
                [
                    paddle.ones((batch_size, seq_length, prefix_seq_len), dtype=causal_mask.dtype),
                    causal_mask,
                ],
                axis=-1,
            )

        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        return extended_attention_mask

    def get_extended_attention_mask(
        self, attention_mask: paddle.Tensor, input_shape: Tuple[int], dtype: paddle.dtype = None
    ) -> paddle.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`paddle.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `paddle.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        if dtype is None:
            dtype = self.dtype

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.ndim == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.cast(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * paddle.finfo(dtype).min
        return extended_attention_mask

    def get_head_mask(
        self, head_mask: Optional[paddle.Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> paddle.Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
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
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask


class PretrainedModel(PPNLPPretrainedModel, ModuleUtilsMixin, PeftAdapterMixin):
    supports_gradient_checkpointing = False

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func: Callable = recompute):
        is_gradient_checkpointing_set = False

        # Apply it on the top-level module in case the top-level modules supports it
        # for example, LongT5Stack inherits from `PretrainedModel`.
        if hasattr(self, "gradient_checkpointing"):
            self._gradient_checkpointing_func = gradient_checkpointing_func
            self.gradient_checkpointing = enable
            is_gradient_checkpointing_set = True

        for module in self.sublayers(include_self=True):
            if hasattr(module, "gradient_checkpointing"):
                module._gradient_checkpointing_func = gradient_checkpointing_func
                module.gradient_checkpointing = enable
                is_gradient_checkpointing_set = True

        if not is_gradient_checkpointing_set:
            raise ValueError(
                f"{self.__class__.__name__} is not compatible with gradient checkpointing. Make sure all the architecture support it by setting a boolean attribute"
                " `gradient_checkpointing` to modules of the model that uses checkpointing."
            )

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        state_dict,
        loaded_keys,
        resolved_archive_file,
        pretrained_model_name_or_path,
        config=None,
        ignore_mismatched_sizes=False,
        low_cpu_mem_usage=False,
        dtype=None,
        keep_in_fp32_modules=None,
        quantization_linear_list=None,
        **kwargs,
    ):
        # load from deprecated state dict
        loaded_keys = cls._update_deprecated_state_dict(state_dict, loaded_keys, model)
        return super()._load_pretrained_model(
            model,
            state_dict,
            loaded_keys,
            resolved_archive_file,
            pretrained_model_name_or_path,
            config=config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            low_cpu_mem_usage=low_cpu_mem_usage,
            dtype=dtype,
            keep_in_fp32_modules=keep_in_fp32_modules,
            quantization_linear_list=quantization_linear_list,
            **kwargs,
        )

    @classmethod
    def _update_deprecated_state_dict(cls, state_dict=None, loaded_keys=None, model=None):
        if state_dict is None:
            return loaded_keys
        _deprecated_dict = getattr(cls, "_deprecated_dict", None)
        from_deprecated_state_dict = _deprecated_dict is not None and any(
            cls._deprecated_dict.get("key", "NONE") in all_key for all_key in state_dict.keys()
        )
        if from_deprecated_state_dict:
            ppnlp_logger.warning(
                "Loading from deprecated state_dict, please load new state_dict via setting `use_safetensors=True`."
            )
            for name in list(state_dict.keys()):
                deprecated_name = name
                for old_name, new_name in cls._deprecated_dict.get("name_mapping", {}).items():
                    name = name.replace(old_name, new_name)
                state_dict[name] = state_dict.pop(deprecated_name)
            loaded_keys = list(state_dict.keys())
        return loaded_keys

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        return any(
            hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing
            for m in self.sublayers(include_self=True)
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".

        We pass the `__call__` method of the modules instead of `forward` because `__call__` attaches all the hooks of
        the module. https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2

        Args:
            gradient_checkpointing_kwargs (dict, *optional*):
                Additional keyword arguments passed along to the `paddle.distributed.fleet.utils.recompute` function.
        """
        if not self.supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}

        if recompute_use_reentrant():
            # do nothint, default use_reentrant is True, we will use pylayer recompute
            pass
        else:
            gradient_checkpointing_kwargs["use_reentrant"] = False

        gradient_checkpointing_func = functools.partial(recompute, **gradient_checkpointing_kwargs)

        # For old GC format (transformers < 4.35.0) for models that live on the Hub
        # we will fall back to the overwritten `_set_gradient_checkpointing` method
        _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters

        if not _is_using_old_format:
            self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
        else:
            self.apply(partial(self._set_gradient_checkpointing, value=True))
            ppnlp_logger.warn(
                "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
            )

    def gradient_checkpointing_disable(self):
        """
        Deactivates gradient checkpointing for the current model.

        Note that in other frameworks this feature can be referred to as "activation checkpointing" or "checkpoint
        activations".
        """
        if self.supports_gradient_checkpointing:
            # For old GC format (transformers < 4.35.0) for models that live on the Hub
            # we will fall back to the overwritten `_set_gradient_checkpointing` method
            _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters
            if not _is_using_old_format:
                self._set_gradient_checkpointing(enable=False)
            else:
                ppnlp_logger.warn(
                    "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
                    "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
                )
                self.apply(partial(self._set_gradient_checkpointing, value=False))

    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()

    def _backward_compatibility_gradient_checkpointing(self):
        if self.supports_gradient_checkpointing and getattr(self.config, "gradient_checkpointing", False):
            self.gradient_checkpointing_enable()
            # Remove the attribute now that is has been consumed, so it's no saved in the config.
            delattr(self.config, "gradient_checkpointing")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        from_diffusers = kwargs.pop("from_diffusers", None)
        revision = kwargs.pop("revision", None)  # noqa: F841
        if from_diffusers is not None:
            kwargs["convert_from_torch"] = from_diffusers
        # pop `paddle_dtype`
        dtype = kwargs.pop("dtype", kwargs.pop("paddle_dtype", None))
        if isinstance(dtype, paddle.dtype):
            dtype = str(dtype).replace("paddle.", "")
        if dtype is not None:
            kwargs["dtype"] = dtype
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *args,
            **kwargs,
        )
        model.eval()
        return model

    def save_pretrained(
        self,
        save_dir: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = paddle.save,
        max_shard_size: Union[int, str] = "10GB",
        safe_serialization: bool = False,
        variant: Optional[str] = None,
        to_diffusers: bool = False,
        *args,
        **kwargs,
    ):
        if to_diffusers:
            # TODO, currently, we do not support to save the model in shared format.
            max_shard_size = "99999GB"
            state_dict = self.state_dict()
            from ppdiffusers.models.modeling_pytorch_paddle_utils import (
                convert_paddle_state_dict_to_pytorch,
            )

            if not is_torch_available():
                safe_serialization = True
            if safe_serialization:

                def replace_name(name):
                    name = name.replace("model_state", "model")
                    name = name.replace(".pdparams", ".safetensors")
                    return name

                if is_torch_available():

                    def save_function(state_dict, path):
                        return torch_safe_save_file(state_dict, replace_name(path), metadata={"format": "pt"})

                else:

                    def save_function(state_dict, path):
                        return np_safe_save_file(state_dict, replace_name(path), metadata={"format": "pt"})

            else:

                def replace_name(name):
                    name = name.replace("model_state", "pytorch_model")
                    name = name.replace(".pdparams", ".bin")
                    return name

                def save_function(state_dict, path):
                    return torch.save(state_dict, replace_name(path))

            convert_paddle_state_dict_to_pytorch(self, state_dict)
            safe_serialization = False

        return super().save_pretrained(
            save_dir=save_dir,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
        )

    @staticmethod
    def prepare_attention_mask_for_generation(input_ids, pad_token_id, eos_token_id):
        # NOTE: we use 2D attention mask!
        is_pad_token_in_inputs_ids = (pad_token_id is not None) and paddle.any(input_ids == pad_token_id).item()
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (
            (eos_token_id is not None) and (pad_token_id != eos_token_id)
        )
        if is_pad_token_in_inputs_ids and is_pad_token_not_equal_to_eos_token_id:
            attention_mask = (input_ids != pad_token_id).cast("int64")
        else:
            attention_mask = paddle.ones_like(input_ids, dtype="int64")
        return attention_mask


class PretrainedConfig(PPNLPPretrainedConfig):
    pass
