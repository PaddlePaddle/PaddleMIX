# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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
import copy
import inspect
import warnings
from typing import Optional

import paddle

from ppdiffusers.accelerate.hooks import add_hook_to_module, remove_hook_from_module

from .constants import (
    COMMON_LAYERS_PATTERN,
    CONFIG_NAME,
    EMBEDDING_LAYER_NAMES,
    SAFETENSORS_WEIGHTS_NAME,
    TORCH_WEIGHTS_NAME,
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    WEIGHTS_NAME,
)

__all__ = [
    "COMMON_LAYERS_PATTERN",
    "CONFIG_NAME",
    "EMBEDDING_LAYER_NAMES",
    "SAFETENSORS_WEIGHTS_NAME",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "WEIGHTS_NAME",
    "TORCH_WEIGHTS_NAME",
]


# Get current device name based on available devices
def infer_device():
    if paddle.device.is_compiled_with_cuda():
        device = "gpu"
        return device
    return "cpu"


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):
    return model


# For backward compatibility
def prepare_model_for_int8_training(*args, **kwargs):
    warnings.warn(
        "prepare_model_for_int8_training is deprecated and will be removed in a future version. Use prepare_model_for_kbit_training instead.",
        FutureWarning,
    )
    return prepare_model_for_kbit_training(*args, **kwargs)


# copied from paddlenlp.transformers.models.bart.modeling_bart
def shift_tokens_right(input_ids: paddle.Tensor, pad_token_id: int, decoder_start_token_id: int):
    return


class ModulesToSaveWrapper(paddle.nn.Layer):
    def __init__(self, module_to_save, adapter_name):
        super().__init__()
        self.original_module = module_to_save
        self.modules_to_save = paddle.nn.LayerDict({})
        self._active_adapter = adapter_name
        self._disable_adapters = False
        self.update(adapter_name)

    @property
    def disable_adapters(self) -> bool:
        # use a property to ensure that disable_adapters is not set directly, instead use the enable_adapters method
        return self._disable_adapters

    @property
    def active_adapter(self) -> str:
        # use a property to ensure that active_adapter is not set directly, instead use the set_adapter method
        return self._active_adapter

    def update(self, adapter_name):
        self.modules_to_save.update(paddle.nn.LayerDict({adapter_name: copy.deepcopy(self.original_module)}))

        if hasattr(self.modules_to_save[adapter_name], "_pp_hook"):
            old_hook = self.modules_to_save[adapter_name]._pp_hook
            new_hook = self._create_new_hook(old_hook)
            remove_hook_from_module(self.modules_to_save[adapter_name])
            add_hook_to_module(self.modules_to_save[adapter_name], new_hook)

        self.original_module.requires_grad_(False)
        if adapter_name == self.active_adapter:
            self.modules_to_save[adapter_name].requires_grad_(True)

    def _create_new_hook(self, old_hook):
        r"""
        Creates a new hook based on the old hook. Use it only if you know what you are doing !
        """
        import ppdiffusers.accelerate.hooks

        old_hook_cls = getattr(ppdiffusers.accelerate.hooks, old_hook.__class__.__name__)
        old_hook_attr = old_hook.__dict__
        filtered_old_hook_attr = {}
        old_hook_init_signature = inspect.signature(old_hook_cls.__init__)
        for k in old_hook_attr.keys():
            if k in old_hook_init_signature.parameters:
                filtered_old_hook_attr[k] = old_hook_attr[k]
        new_hook = old_hook_cls(**filtered_old_hook_attr)
        return new_hook

    def forward(self, *args, **kwargs):
        if self.disable_adapters or (self.active_adapter not in self.modules_to_save):
            return self.original_module(*args, **kwargs)
        return self.modules_to_save[self.active_adapter](*args, **kwargs)

    def enable_adapters(self, enabled: bool):
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if self._disable_adapters is not enabled:
            # already in the desired state, do nothing
            return

        if enabled:
            self.original_module.requires_grad_(False)
            self.modules_to_save[self.active_adapter].requires_grad_(True)
            self._disable_adapters = False
        else:
            self.original_module.requires_grad_(True)
            self.modules_to_save.requires_grad_(False)
            self._disable_adapters = True

    def set_adapter(self, adapter_name: str):
        """Set the active adapter

        Args:
            adapter_name (str): The name of the adapter to set as active
        """
        if adapter_name not in self.modules_to_save:
            raise ValueError(f"Adapter {adapter_name} not found in {self.modules_to_save.keys()}")

        self.modules_to_save[self.active_adapter].requires_grad_(False)
        self.modules_to_save[adapter_name].requires_grad_(True)
        self._active_adapter = adapter_name


def _get_submodules(model, key):
    parent = model.get_sublayer(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_sublayer(key)
    return parent, target, target_name


def _freeze_adapter(model, adapter_name):
    for n, p in model.named_parameters():
        if adapter_name in n:
            p.stop_gradient = True


def _set_trainable(model, adapter_name):
    key_list = [key for key, _ in model.named_sublayers(include_self=True)]
    for key in key_list:
        target_module_found = any(key.endswith(target_key) for target_key in model.modules_to_save)
        if target_module_found:
            parent, target, target_name = _get_submodules(model, key)
            if isinstance(target, ModulesToSaveWrapper):
                target.update(adapter_name)
                target.set_adapter(target.active_adapter)
            else:
                new_module = ModulesToSaveWrapper(target, adapter_name)
                new_module.set_adapter(adapter_name)
                setattr(parent, target_name, new_module)


def _set_adapter(model, adapter_name):
    def check_adapter_name(adapter_name):
        if isinstance(adapter_name, str):
            return adapter_name

        # adapter_name is a list of str
        if len(adapter_name) > 1:
            raise ValueError("Only one adapter can be set at a time for modules_to_save")
        elif len(adapter_name) == 0:
            raise ValueError("Please specify at least one adapter to set")
        adapter_name = adapter_name[0]
        return adapter_name

    for module in model.sublayers(include_self=True):
        if isinstance(module, ModulesToSaveWrapper):
            # only check the adapter_name if we actually encounter a ModulesToSaveWrapper, otherwise we don't care
            adapter_name = check_adapter_name(adapter_name)
            module.set_adapter(adapter_name)


def _prepare_prompt_learning_config(peft_config, model_config):
    if peft_config.num_layers is None:
        if "num_hidden_layers" in model_config:
            num_layers = model_config["num_hidden_layers"]
        elif "num_layers" in model_config:
            num_layers = model_config["num_layers"]
        elif "n_layer" in model_config:
            num_layers = model_config["n_layer"]
        else:
            raise ValueError("Please specify `num_layers` in `peft_config`")
        peft_config.num_layers = num_layers

    if peft_config.token_dim is None:
        if "hidden_size" in model_config:
            token_dim = model_config["hidden_size"]
        elif "n_embd" in model_config:
            token_dim = model_config["n_embd"]
        elif "d_model" in model_config:
            token_dim = model_config["d_model"]
        else:
            raise ValueError("Please specify `token_dim` in `peft_config`")
        peft_config.token_dim = token_dim

    if peft_config.num_attention_heads is None:
        if "num_attention_heads" in model_config:
            num_attention_heads = model_config["num_attention_heads"]
        elif "n_head" in model_config:
            num_attention_heads = model_config["n_head"]
        elif "num_heads" in model_config:
            num_attention_heads = model_config["num_heads"]
        elif "encoder_attention_heads" in model_config:
            num_attention_heads = model_config["encoder_attention_heads"]
        else:
            raise ValueError("Please specify `num_attention_heads` in `peft_config`")
        peft_config.num_attention_heads = num_attention_heads

    if getattr(peft_config, "encoder_hidden_size", None) is None:
        setattr(peft_config, "encoder_hidden_size", peft_config.token_dim)

    return peft_config


def fsdp_auto_wrap_policy(model):
    pass
    # import functools
    # import os

    # from accelerate import FullyShardedDataParallelPlugin
    # from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

    # from ..tuners import PrefixEncoder, PromptEmbedding, PromptEncoder

    # def lambda_policy_fn(module):
    #     if (
    #         len(list(module.named_children())) == 0
    #         and getattr(module, "weight", None) is not None
    #         and module.weight.requires_grad
    #     ):
    #         return True
    #     return False

    # lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    # transformer_wrap_policy = functools.partial(
    #     transformer_auto_wrap_policy,
    #     transformer_layer_cls=(
    #         PrefixEncoder,
    #         PromptEncoder,
    #         PromptEmbedding,
    #         FullyShardedDataParallelPlugin.get_module_class_from_name(
    #             model, os.environ.get("FSDP_TRANSFORMER_CLS_TO_WRAP", "")
    #         ),
    #     ),
    # )

    # auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    return None


def transpose(weight, fan_in_fan_out):
    if not fan_in_fan_out:
        return weight

    if isinstance(weight, paddle.base.framework.Parameter):
        return paddle.nn.Parameter(weight.T)
    return weight.T


def _is_valid_match(key: str, target_key: str):
    """
    Helper function to match module names target_key and key. Makes sure that either the key is exactly the target_key
    or the target_key is a submodule of key
    """
    if key.endswith(target_key):
        if len(key) > len(target_key):
            return key.endswith("." + target_key)  # must be a sub module
        return True
    return False


def _get_batch_size(input_ids: Optional[paddle.Tensor], inputs_embeds: Optional[paddle.Tensor]) -> int:
    """Get the batch size based on either input_ids or input_embeds

    Raises an ValueError if both are None.

    """
    if (input_ids is None) and (inputs_embeds is None):
        raise ValueError("You have to provide either input_ids or inputs_embeds")

    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]
    return batch_size


def get_quantization_config(model: paddle.nn.Layer, method: str):
    """
    Get the quantization config of the related quantization method
    """
    if (
        hasattr(model, "config")
        and hasattr(model.config, "quantization_config")
        and (getattr(model, "quantization_method", None) == method)
    ):
        return model.config.quantization_config
    return None


def get_auto_gptq_quant_linear(gptq_quantization_config):
    pass
    return None
