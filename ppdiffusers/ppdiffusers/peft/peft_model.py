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

from __future__ import annotations

import inspect
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import paddle
from paddlenlp.transformers import PretrainedModel
from safetensors.numpy import save_file as safe_save_file

from . import __version__  # noqa: 401
from .config import PeftConfig
from .tuners import (
    AdaLoraModel,
    IA3Model,
    LoHaModel,
    LoKrModel,
    LoraModel,
    MultitaskPromptEmbedding,
    OFTModel,
    PrefixEncoder,
    PromptEmbedding,
    PromptEncoder,
)
from .utils import (
    SAFETENSORS_WEIGHTS_NAME,
    TORCH_WEIGHTS_NAME,
    WEIGHTS_NAME,
    PeftType,
    _prepare_prompt_learning_config,
    _set_adapter,
    _set_trainable,
    get_peft_model_state_dict,
    infer_device,
    load_peft_weights,
    set_peft_model_state_dict,
)

PEFT_TYPE_TO_MODEL_MAPPING = {
    PeftType.LORA: LoraModel,
    PeftType.LOHA: LoHaModel,
    PeftType.LOKR: LoKrModel,
    PeftType.PROMPT_TUNING: PromptEmbedding,
    PeftType.P_TUNING: PromptEncoder,
    PeftType.PREFIX_TUNING: PrefixEncoder,
    PeftType.ADALORA: AdaLoraModel,
    PeftType.IA3: IA3Model,
    PeftType.OFT: OFTModel,
}
from ppdiffusers.utils import PushToHubMixin, SaveToAistudioMixin
from ppdiffusers.utils.downloader import bos_aistudio_hf_download


class PeftModel(PushToHubMixin, SaveToAistudioMixin, paddle.nn.Layer):
    """
    Base model encompassing various Peft methods.

    Args:
        model ([`~transformers.PretrainedModel`]): The base transformer model used for Peft.
        peft_config ([`PeftConfig`]): The configuration of the Peft model.
        adapter_name (`str`,  *optional*): The name of the adapter, defaults to `"default"`.

    **Attributes**:
        - **base_model** ([`paddle.nn.Layer`]) -- The base transformer model used for Peft.
        - **peft_config** ([`PeftConfig`]) -- The configuration of the Peft model.
        - **modules_to_save** (`list` of `str`) -- The list of sub-module names to save when
            saving the model.
        - **prompt_encoder** ([`PromptEncoder`]) -- The prompt encoder used for Peft if
            using [`PromptLearningConfig`].
        - **prompt_tokens** (`paddle.Tensor`) -- The virtual prompt tokens used for Peft if
            using [`PromptLearningConfig`].
        - **transformer_backbone_name** (`str`) -- The name of the transformer
            backbone in the base model if using [`PromptLearningConfig`].
        - **word_embeddings** (`paddle.nn.Embedding`) -- The word embeddings of the transformer backbone
            in the base model if using [`PromptLearningConfig`].
    """

    def __init__(self, model: PretrainedModel, peft_config: PeftConfig, adapter_name: str = "default") -> None:
        super().__init__()
        self.modules_to_save = None
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type

        self._is_prompt_learning = peft_config.is_prompt_learning
        if self._is_prompt_learning:
            self._peft_config = {adapter_name: peft_config}
            self.base_model = model
            self.add_adapter(adapter_name, peft_config)
        else:
            self._peft_config = None
            cls = PEFT_TYPE_TO_MODEL_MAPPING[peft_config.peft_type]
            self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)
            self.set_additional_trainable_modules(peft_config, adapter_name)

        if getattr(model, "is_gradient_checkpointing", True):
            model = self._prepare_model_for_gradient_checkpointing(model)

        # the `pretraining_tp` is set for some models to simulate Tensor Parallelism during inference to avoid
        # numerical differences, https://github.com/pytorch/pytorch/issues/76232 - to avoid any unexpected
        # behavior we disable that in this line.
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
            self.base_model.config.pretraining_tp = 1

    @property
    def peft_config(self) -> Dict[str, PeftConfig]:
        if self._is_prompt_learning:
            return self._peft_config
        return self.base_model.peft_config

    @property
    def active_adapters(self) -> list[str]:
        try:
            adapters = self.base_model.active_adapters
        except AttributeError:
            adapters = self.active_adapter
            if isinstance(adapters, str):
                adapters = [adapters]
        return adapters

    @peft_config.setter
    def peft_config(self, value: Dict[str, PeftConfig]):
        if self._is_prompt_learning:
            self._peft_config = value
        else:
            self.base_model.peft_config = value

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        selected_adapters: Optional[List[str]] = None,
        save_embedding_layers: Union[str, bool] = "auto",
        is_main_process: bool = True,
        to_diffusers: bool = False,
        **kwargs: Any,
    ) -> None:
        r"""
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`PeftModel.from_pretrained`] class method, and also used by the [`PeftModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            safe_serialization (`bool`, *optional*):
                Whether to save the adapter files in safetensors format, defaults to `True`.
            selected_adapters (`List[str]`,  *optional*):
                A list of adapters to be saved. If `None`, will default to all adapters.
            save_embedding_layers (`Union[bool, str]`, *optional*, defaults to `"auto"`):
                If `True`, save the embedding layers in addition to adapter weights. If `auto`, checks the common
                embedding layers `peft.utils.other.EMBEDDING_LAYER_NAMES` in config's `target_modules` when available.
                and automatically sets the boolean flag. This only works for 🤗 transformers models.
            is_main_process (`bool`, *optional*):
                Whether the process calling this is the main process or not. Will default to `True`. Will not save the
                checkpoint if not on the main process, which is important for multi device setups (e.g. DDP).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())
        else:
            if any(
                selected_adapter_name not in list(self.peft_config.keys())
                for selected_adapter_name in selected_adapters
            ):
                raise ValueError(
                    f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                    f" {list(self.peft_config.keys())} - got {selected_adapters}."
                )

        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            self.create_or_update_model_card(save_directory)

        for adapter_name in selected_adapters:
            peft_config = self.peft_config[adapter_name]
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                self,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name,
                save_embedding_layers=save_embedding_layers,
                to_diffusers=to_diffusers,
            )
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)

            if is_main_process and safe_serialization:
                safe_save_file(
                    {k: v.cpu().numpy() for k, v in output_state_dict.items()},
                    os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                    metadata={"format": "pt" if to_diffusers else "np"},
                )
            elif is_main_process:
                if to_diffusers:
                    import torch

                    torch.save(
                        {k: torch.from_numpy(v.cpu().numpy()) for k, v in output_state_dict.items()},
                        os.path.join(output_dir, TORCH_WEIGHTS_NAME),
                    )
                else:
                    paddle.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None)
                    if peft_config.is_prompt_learning
                    else self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True

            if peft_config.task_type is None:
                # deal with auto mapping
                base_model_class = self._get_base_model_class(
                    is_prompt_tuning=peft_config.is_prompt_learning,
                )
                parent_library = base_model_class.__module__

                auto_mapping_dict = {
                    "base_model_class": base_model_class.__name__,
                    "parent_library": parent_library,
                }
            else:
                auto_mapping_dict = None

            if is_main_process:
                peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
            peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(
        cls,
        model: paddle.nn.Layer,
        model_id: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        from_diffusers: bool = False,
        **kwargs: Any,
    ) -> "PeftModel":
        r"""
        Instantiate a PEFT model from a pretrained model and loaded PEFT weights.

        Note that the passed `model` may be modified inplace.

        Args:
            model ([`paddle.nn.Layer`]):
                The model to be adapted. For 🤗 Transformers models, the model should be initialized with the
                [`~transformers.PretrainedModel.from_pretrained`].
            model_id (`str` or `os.PathLike`):
                The name of the PEFT configuration to use. Can be either:
                    - A string, the `model id` of a PEFT configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a PEFT configuration file saved using the `save_pretrained`
                      method (`./my_peft_config_directory/`).
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to be loaded. This is useful for loading multiple adapters.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
                used for inference.
            config ([`~peft.PeftConfig`], *optional*):
                The configuration object to use instead of an automatically loaded configuration. This configuration
                object is mutually exclusive with `model_id` and `kwargs`. This is useful when configuration is already
                loaded before calling `from_pretrained`.
            kwargs: (`optional`):
                Additional keyword arguments passed along to the specific PEFT configuration class.
        """
        from .mapping import (
            MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
            PEFT_TYPE_TO_CONFIG_MAPPING,
        )

        # load the config
        if config is None:
            config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    subfolder=kwargs.get("subfolder", None),
                    revision=kwargs.get("revision", None),
                    cache_dir=kwargs.get("cache_dir", None),
                    token=kwargs.get("token", None) or kwargs.get("use_auth_token", None),
                    from_bos=kwargs.get("from_bos", True),
                    from_aistudio=kwargs.get("from_aistudio", False),
                    from_hf_hub=kwargs.get("from_hf_hub", False),
                )
            ].from_pretrained(model_id, **kwargs)
        elif isinstance(config, PeftConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")

        if config.is_prompt_learning and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable

        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            model = cls(model, config, adapter_name)
        else:
            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config, adapter_name)
        model.load_adapter(model_id, adapter_name, is_trainable=is_trainable, from_diffusers=from_diffusers, **kwargs)
        return model

    def _setup_prompt_encoder(self, adapter_name: str):
        raise NotImplementedError()

    def _prepare_model_for_gradient_checkpointing(self, model: PretrainedModel):
        r"""
        Prepares the model for gradient checkpointing if necessary
        """
        if not (
            getattr(model, "is_loaded_in_8bit", False)
            or getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_quantized", False)
        ):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            elif hasattr(model, "get_input_embeddings"):

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_pre_hook(make_inputs_require_grad)
        return model

    def get_prompt_embedding_to_save(self, adapter_name: str) -> paddle.Tensor:
        """
        Returns the prompt embedding to save when saving the model. Only applicable when using a prompt learning
        method.
        """
        prompt_encoder = self.prompt_encoder[adapter_name]
        prompt_tokens = self.prompt_tokens[adapter_name].unsqueeze(0).expand([1, -1])
        if self.peft_config[adapter_name].peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : self.peft_config[adapter_name].num_virtual_tokens]

        if self.peft_config[adapter_name].peft_type == PeftType.MULTITASK_PROMPT_TUNING:
            prompt_embeddings = super(MultitaskPromptEmbedding, prompt_encoder).forward(prompt_tokens)
        else:
            prompt_embeddings = prompt_encoder(prompt_tokens)

        return prompt_embeddings[0].detach().cpu()

    def get_prompt(self, batch_size: int, task_ids: Optional[paddle.Tensor] = None) -> paddle.Tensor:
        """
        Returns the virtual prompts to use for Peft. Only applicable when using a prompt learning method.
        """
        peft_config = self.active_peft_config
        prompt_encoder = self.prompt_encoder[self.active_adapter]
        prompt_tokens = self.prompt_tokens[self.active_adapter].unsqueeze(0).expand([batch_size, -1])
        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : peft_config.num_virtual_tokens]
            if peft_config.inference_mode:
                past_key_values = prompt_encoder.embedding.weight.tile([batch_size, 1, 1])
            else:
                past_key_values = prompt_encoder(prompt_tokens)
            if self.base_model_torch_dtype is not None:
                past_key_values = past_key_values.cast(self.base_model_torch_dtype)
            past_key_values = past_key_values.reshape(
                [
                    batch_size,
                    peft_config.num_virtual_tokens,
                    peft_config.num_layers * 2,
                    peft_config.num_attention_heads,
                    peft_config.token_dim // peft_config.num_attention_heads,
                ]
            )
            if peft_config.num_transformer_submodules == 2:
                past_key_values = paddle.concat([past_key_values, past_key_values], axis=2)
            past_key_values = past_key_values.transpose([2, 0, 3, 1, 4]).split(
                peft_config.num_transformer_submodules * 2
            )
            return past_key_values
        else:
            if peft_config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
                prompts = prompt_encoder(prompt_tokens, task_ids)
            else:
                if peft_config.inference_mode:
                    prompts = prompt_encoder.embedding.weight.tile([batch_size, 1, 1])
                else:
                    prompts = prompt_encoder(prompt_tokens)
            return prompts

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if not param.stop_gradient:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self) -> None:
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Layer's logic
        except AttributeError:
            return getattr(self.base_model, name)

    def forward(self, *args: Any, **kwargs: Any):
        """
        Forward pass of the model.
        """
        return self.get_base_model()(*args, **kwargs)

    def _get_base_model_class(self, is_prompt_tuning=False):
        """
        Returns the base model class.
        """
        if not is_prompt_tuning:
            return self.base_model.model.__class__
        return self.base_model.__class__

    @contextmanager
    def disable_adapter(self):
        """
        Context manager that disables the adapter module. Use this to run inference on the base model.

        Example:

        ```py
        >>> with model.disable_adapter():
        ...     model(inputs)
        ```
        """
        try:
            if self.peft_config[self.active_adapter].is_prompt_learning:
                # TODO: consider replacing this patching of methods with a more robust mechanism: setting a flag and
                # letting the underyling methods deal with it, same as how LoRA does it.
                old_forward = self.forward
                self.forward = self.base_model.forward
                old_prepare_inputs_for_generation = self.prepare_inputs_for_generation
                self.prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
            else:
                self.base_model.disable_adapter_layers()
            yield
        finally:
            if self.peft_config[self.active_adapter].is_prompt_learning:
                self.forward = old_forward
                self.old_prepare_inputs_for_generation = old_prepare_inputs_for_generation
            else:
                self.base_model.enable_adapter_layers()

    def get_base_model(self) -> paddle.nn.Layer:
        """
        Returns the base model.
        """
        return self.base_model if self.active_peft_config.is_prompt_learning else self.base_model.model

    def add_adapter(self, adapter_name: str, peft_config: PeftConfig) -> None:
        """
        Add an adapter to the model based on the passed configuration.

        The name for the new adapter should be unique.

        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
        adapter.

        Args:
            adapter_name (`str`):
                The name of the adapter to be added.
            peft_config ([`PeftConfig`]):
                The configuration of the adapter to be added.
        """
        if peft_config.peft_type != self.peft_type:
            raise ValueError(
                f"Cannot combine adapters with different peft types. "
                f"Found {self.peft_type} and {peft_config.peft_type}."
            )

        try:
            if peft_config.is_prompt_learning:
                self.peft_config[adapter_name] = peft_config
                if hasattr(self.config, "to_dict"):
                    dict_config = self.config.to_dict()
                else:
                    dict_config = self.config

                peft_config = _prepare_prompt_learning_config(peft_config, dict_config)
                self._setup_prompt_encoder(adapter_name)
            elif peft_config.is_adaption_prompt:
                self.base_model.add_adapter(adapter_name, peft_config)
            else:
                self.peft_config[adapter_name] = peft_config
                self.base_model.inject_adapter(self.base_model.model, adapter_name)
        except Exception:  # something went wrong, roll back
            if adapter_name in self.peft_config:
                del self.peft_config[adapter_name]
            raise

        self.set_additional_trainable_modules(peft_config, adapter_name)

    def set_additional_trainable_modules(self, peft_config, adapter_name):
        if getattr(peft_config, "modules_to_save", None) is not None:
            if self.modules_to_save is None:
                self.modules_to_save = set(peft_config.modules_to_save)
            else:
                self.modules_to_save.update(peft_config.modules_to_save)
            _set_trainable(self, adapter_name)

    @classmethod
    def _split_kwargs(cls, kwargs: Dict[str, Any]):
        _kwargs_not_in_hf_hub_download_signature = ("use_auth_token",)
        bos_aistudio_hf_hub_download_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if (
                key in inspect.signature(bos_aistudio_hf_download).parameters
                or key in _kwargs_not_in_hf_hub_download_signature
            ):
                bos_aistudio_hf_hub_download_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return bos_aistudio_hf_hub_download_kwargs, other_kwargs

    def load_adapter(
        self, model_id: str, adapter_name: str, is_trainable: bool = False, from_diffusers: bool = False, **kwargs: Any
    ):
        """
        Load a trained adapter into the model.

        The name for the new adapter should be unique.

        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
        adapter.

        Args:
            adapter_name (`str`):
                The name of the adapter to be added.
            peft_config ([`PeftConfig`]):
                The configuration of the adapter to be added.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
                used for inference.
            kwargs: (`optional`):
                Additional arguments to modify the way the adapter is loaded, e.g. the token for Hugging Face Hub.
        """
        from .mapping import PEFT_TYPE_TO_CONFIG_MAPPING

        bos_aistudio_hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)
        paddle_device = infer_device()

        if adapter_name not in self.peft_config:
            # load the config
            peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    **bos_aistudio_hf_hub_download_kwargs,
                )
            ].from_pretrained(
                model_id,
                **bos_aistudio_hf_hub_download_kwargs,
            )
            if peft_config.is_prompt_learning and is_trainable:
                raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
            else:
                peft_config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, peft_config)

        adapters_weights, from_diffusers = load_peft_weights(
            model_id, device=paddle_device, from_diffusers=from_diffusers, **bos_aistudio_hf_hub_download_kwargs
        )

        # load the weights into the model
        load_result = set_peft_model_state_dict(
            self, adapters_weights, adapter_name=adapter_name, from_diffusers=from_diffusers
        )

        # Set model in evaluation mode to deactivate Dropout modules by default
        if not is_trainable:
            self.eval()
        return load_result

    def set_adapter(self, adapter_name: str) -> None:
        """
        Sets the active adapter.

        Only one adapter can be active at a time.

        Args:
            adapter_name (`str`):
                The name of the adapter to be set as active. The adapter must be loaded first.
        """
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} not found.")
        self.active_adapter = adapter_name
        if not self.peft_config[adapter_name].is_prompt_learning:
            self.base_model.set_adapter(adapter_name)
        _set_adapter(self, adapter_name)

    @property
    def base_model_torch_dtype(self):
        return getattr(self.base_model, "dtype", None)

    @property
    def active_peft_config(self):
        return self.peft_config[self.active_adapter]

    def create_or_update_model_card(self, output_dir: str):
        return
