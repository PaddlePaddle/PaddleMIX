# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import gc
import json
import os
from collections import OrderedDict
from contextlib import ExitStack
from functools import partial
from typing import Any, Callable, List, Optional, Union

import numpy as np
from aistudio_sdk.hub import create_repo as aistudio_create_repo
from huggingface_hub import create_repo
from paddle import nn
from tqdm import tqdm

from ..utils import (
    CONFIG_NAME,
    DIFFUSERS_CACHE,
    FROM_AISTUDIO,
    FROM_DIFFUSERS,
    FROM_HF_HUB,
    HF_HUB_OFFLINE,
    LOW_CPU_MEM_USAGE_DEFAULT,
    MIN_PEFT_VERSION,
    PADDLE_SAFETENSORS_WEIGHTS_NAME,
    PADDLE_SAFETENSORS_WEIGHTS_NAME_INDEX_NAME,
    PADDLE_WEIGHTS_NAME,
    PADDLE_WEIGHTS_NAME_INDEX_NAME,
    PPDIFFUSERS_CACHE,
    TO_DIFFUSERS,
    TORCH_SAFETENSORS_WEIGHTS_NAME,
    TORCH_SAFETENSORS_WEIGHTS_NAME_INDEX_NAME,
    TORCH_WEIGHTS_NAME,
    TORCH_WEIGHTS_NAME_INDEX_NAME,
    _add_variant,
    _get_model_file,
    check_peft_version,
    deprecate,
    get_checkpoint_shard_files,
    is_paddle_available,
    is_paddle_version,
    is_paddlenlp_available,
    is_safetensors_available,
    is_torch_available,
    logging,
    smart_load,
)
from ..version import VERSION as __version__
from .modeling_pytorch_paddle_utils import (
    convert_paddle_state_dict_to_pytorch,
    convert_pytorch_state_dict_to_paddle,
)

logger = logging.get_logger(__name__)

if is_torch_available():
    import torch

if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.numpy import save_file as np_safe_save_file

    if is_torch_available():
        from safetensors.torch import save_file as torch_safe_save_file

if is_paddle_available():
    import paddle

if is_paddlenlp_available():
    try:
        from paddlenlp.transformers.model_utils import no_init_weights
    except ImportError:
        from ..utils.paddle_utils import no_init_weights
    from paddlenlp.transformers.model_utils import shard_checkpoint


def faster_set_state_dict(model, state_dict):
    # the state_dict will be destroyed.
    with paddle.no_grad():
        for k, v in model.state_dict(use_hook=False).items():
            if k in state_dict:
                v_new = state_dict.pop(k)
                # with device_guard(): do not do device guard
                if isinstance(v_new, np.ndarray):
                    v_new = paddle.Tensor(v_new, zero_copy=True)
                if v.dtype != v_new.dtype:
                    v_new = v_new.cast(v.dtype)
                v.copy_(v_new, False)
            else:
                if (hasattr(v, "_is_initialized") and not v._is_initialized()) or "undefined" in str(v.place):
                    v.initialize()
                    # logger.warning(f"key {k} is not in state_dict. And it is lazy tensor. We will initialize it.")


class ContextManagers:
    """
    Wrapper for `contextlib.ExitStack` which enters a collection of context managers. Adaptation of `ContextManagers`
    in the `fastcore` library.
    """

    def __init__(self, context_managers):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)


def get_parameter_device(parameter: nn.Layer):
    try:
        # TODO https://github.com/huggingface/diffusers/compare/v0.15.0...v0.16.0#diff-6a3b9a08c1d37dbc341131632415fea800af242a84fb31f1bcd40d725e2eeeebR64
        return next(parameter.named_parameters())[1].place
    except StopIteration:
        try:
            return next(parameter.named_buffers())[1].place
        except StopIteration:
            return paddle.get_device()


def get_parameter_dtype(parameter: nn.Layer) -> paddle.dtype:
    try:
        # TODO https://github.com/huggingface/diffusers/compare/v0.15.0...v0.16.0#diff-6a3b9a08c1d37dbc341131632415fea800af242a84fb31f1bcd40d725e2eeeebR80
        return next(parameter.named_parameters())[1].dtype
    except StopIteration:
        try:
            return next(parameter.named_buffers())[1].dtype
        except StopIteration:
            return parameter._dtype


def load_state_dict(
    checkpoint_file: Union[str, os.PathLike], state_dict, tensor_parallel_split_mapping=None, ignore_keys=None, map_location=None
):
    """
    Reads a PaddlePaddle checkpoint file, returning properly formatted errors if they arise.
    """
    if tensor_parallel_split_mapping is None:
        tensor_parallel_split_mapping = {}
    data_format = "pd"
    if checkpoint_file.endswith(".safetensors") and is_safetensors_available():
        # Check format of the archive
        with safe_open(checkpoint_file, framework="np") as f:
            metadata = f.metadata()
        if metadata is None:
            metadata = {}
        if metadata.get("format", "pt") not in ["pt", "pd", "np"]:
            raise OSError(
                f"The safetensors archive passed at {checkpoint_file} does not contain the valid metadata. Make sure "
                "you save your model with the `save_pretrained` method."
            )
        data_format = metadata.get("format", "pt")
        with safe_open(checkpoint_file, framework="np") as f:
            for key in f.keys():
                need_continue = False
                if ignore_keys is not None:
                    for ik in ignore_keys:
                        if key.startswith(ik):
                            logger.info("Deleting key {} from state_dict.".format(key))
                            need_continue = True
                            break
                if need_continue:
                    continue
                if key in tensor_parallel_split_mapping:
                    py_safe_slice_ = f.get_slice(key)
                    weight = tensor_parallel_split_mapping[key](py_safe_slice_)
                else:
                    weight = f.get_tensor(key)

                if map_location=="cpu":   
                    state_dict[key] = paddle.Tensor(weight, zero_copy=True,place=paddle.CPUPlace())
                else:
                    state_dict[key] = paddle.Tensor(weight, zero_copy=True)

    else:
        if any(checkpoint_file.endswith(suffix) for suffix in [".pt", ".pth", ".bin", ".ckpt"]):
            data_format = "pt"

        tmp_state_dict = smart_load(checkpoint_file, return_numpy=True)
        for key in list(tmp_state_dict.keys()):
            need_continue = False
            if ignore_keys is not None:
                for ik in ignore_keys:
                    if key.startswith(ik):
                        logger.info("Deleting key {} from state_dict.".format(key))
                        need_continue = True
                        break
            if need_continue:
                continue
            # with device_guard():
            t = tmp_state_dict.pop(key)
            if key in tensor_parallel_split_mapping:
                t = tensor_parallel_split_mapping[key](t)
            if isinstance(t, dict):
                if len(t) == 0:
                    state_dict[key] = {}
            else:
                state_dict[key] = paddle.Tensor(t, zero_copy=True)

    return data_format


class ModelMixin(nn.Layer):
    r"""
    Base class for all models.

    [`ModelMixin`] takes care of storing the model configuration and provides methods for loading, downloading and
    saving models.

        - **config_name** ([`str`]) -- Filename to save a model to when calling [`~models.ModelMixin.save_pretrained`].
    """

    config_name = CONFIG_NAME
    _automatically_saved_args = ["_ppdiffusers_version", "_class_name", "_name_or_path"]
    _supports_gradient_checkpointing = False
    _keys_to_ignore_on_load_unexpected = None
    _pp_peft_config_loaded = False

    def __init__(self):
        super().__init__()

    def __getattr__(self, name: str) -> Any:
        """The only reason we overwrite `getattr` here is to gracefully deprecate accessing
        config attributes directly. See https://github.com/huggingface/diffusers/pull/3129 We need to overwrite
        __getattr__ here in addition so that we don't trigger `nn.Layer`'s __getattr__':
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
        """

        is_in_config = "_internal_dict" in self.__dict__ and hasattr(self.__dict__["_internal_dict"], name)
        is_attribute = name in self.__dict__

        if is_in_config and not is_attribute:
            deprecation_message = f"Accessing config attribute `{name}` directly via '{type(self).__name__}' object attribute is deprecated. Please access '{name}' over '{type(self).__name__}'s config object instead, e.g. 'unet.config.{name}'."
            deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False, stacklevel=3)
            return self._internal_dict[name]

        # call PyTorch's https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
        return super().__getattr__(name)

    @property
    def is_gradient_checkpointing(self) -> bool:
        """
        Whether gradient checkpointing is activated for this model or not.
        """
        return any(
            hasattr(m, "gradient_checkpointing") and m.gradient_checkpointing
            for m in self.sublayers(include_self=True)
        )

    def enable_gradient_checkpointing(self) -> None:
        """
        Activates gradient checkpointing for the current model (may be referred to as *activation checkpointing* or
        *checkpoint activations* in other frameworks).
        """
        if not self._supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    def disable_gradient_checkpointing(self) -> None:
        """
        Deactivates gradient checkpointing for the current model (may be referred to as *activation checkpointing* or
        *checkpoint activations* in other frameworks).
        """
        if self._supports_gradient_checkpointing:
            self.apply(partial(self._set_gradient_checkpointing, value=False))

    def set_use_memory_efficient_attention_xformers(self, valid: bool, attention_op: Optional[str] = None) -> None:
        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: nn.Layer):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        for module in self.children():
            if isinstance(module, nn.Layer):
                fn_recursive_set_mem_eff(module)

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[str] = None) -> None:
        r"""
        Enable memory efficient attention from [xFormers](https://facebookresearch.github.io/xformers/).

        When this option is enabled, you should observe lower GPU memory usage and a potential speed up during
        inference. Speed up during training is not guaranteed.

        <Tip warning={true}>

        ⚠️ When memory efficient attention and sliced attention are both enabled, memory efficient attention takes
        precedent.

        </Tip>

        Parameters:
            attention_op (`str`, *optional*):
                Override the default `None`

        Examples:

        ```py
        >>> import paddle
        >>> from ppdiffusers import UNet2DConditionModel

        >>> model = UNet2DConditionModel.from_pretrained(
        ...     "stabilityai/stable-diffusion-2-1", subfolder="unet", paddle_dtype=paddle.float16
        ... )
        >>> model.enable_xformers_memory_efficient_attention(attention_op="auto")
        ```
        """
        self.set_use_memory_efficient_attention_xformers(True, attention_op)

    def disable_xformers_memory_efficient_attention(self) -> None:
        r"""
        Disable memory efficient attention as implemented in xformers.
        """
        self.set_use_memory_efficient_attention_xformers(False)

    def add_adapter(self, adapter_config, adapter_name: str = "default") -> None:
        r"""
        Adds a new adapter to the current model for training. If no adapter name is passed, a default name is assigned
        to the adapter to follow the convention of the PEFT library.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them in the PEFT
        [documentation](https://huggingface.co/docs/peft).

        Args:
            adapter_config (`[~peft.PeftConfig]`):
                The configuration of the adapter to add; supported adapters are non-prefix tuning and adaption prompt
                methods.
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to add. If no name is passed, a default name is assigned to the adapter.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        from ppdiffusers.peft import PeftConfig, inject_adapter_in_model

        if not self._pp_peft_config_loaded:
            self._pp_peft_config_loaded = True
        elif adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        if not isinstance(adapter_config, PeftConfig):
            raise ValueError(
                f"adapter_config should be an instance of PeftConfig. Got {type(adapter_config)} instead."
            )

        # Unlike transformers, here we don't need to retrieve the name_or_path of the unet as the loading logic is
        # handled by the `load_lora_layers` or `LoraLoaderMixin`. Therefore we set it to `None` here.
        adapter_config.base_model_name_or_path = None
        inject_adapter_in_model(adapter_config, self, adapter_name)
        self.set_adapter(adapter_name)

    def set_adapter(self, adapter_name: Union[str, List[str]]) -> None:
        """
        Sets a specific adapter by forcing the model to only use that adapter and disables the other adapters.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Args:
            adapter_name (Union[str, List[str]])):
                The list of adapters to set or the adapter name in case of single adapter.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._pp_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        if isinstance(adapter_name, str):
            adapter_name = [adapter_name]

        missing = set(adapter_name) - set(self.peft_config)
        if len(missing) > 0:
            raise ValueError(
                f"Following adapter(s) could not be found: {', '.join(missing)}. Make sure you are passing the correct adapter name(s)."
                f" current loaded adapters are: {list(self.peft_config.keys())}"
            )

        from ppdiffusers.peft.tuners.tuners_utils import BaseTunerLayer

        _adapters_has_been_set = False

        for _, module in self.named_sublayers(include_self=True):
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                # Previous versions of PEFT does not support multi-adapter inference
                elif not hasattr(module, "set_adapter") and len(adapter_name) != 1:
                    raise ValueError(
                        "You are trying to set multiple adapters and you have a PEFT version that does not support multi-adapter inference. Please upgrade to the latest version of PEFT."
                        " `pip install -U peft` or `pip install -U git+https://github.com/huggingface/peft.git`"
                    )
                else:
                    module.active_adapter = adapter_name
                _adapters_has_been_set = True

        if not _adapters_has_been_set:
            raise ValueError(
                "Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters."
            )

    def disable_adapters(self) -> None:
        r"""
        Disable all adapters attached to the model and fallback to inference with the base model only.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._pp_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from ppdiffusers.peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_sublayers(include_self=True):
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=False)
                else:
                    # support for older PEFT versions
                    module.disable_adapters = True

    def enable_adapters(self) -> None:
        """
        Enable adapters that are attached to the model. The model will use `self.active_adapters()` to retrieve the
        list of adapters to enable.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._pp_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from ppdiffusers.peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_sublayers(include_self=True):
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=True)
                else:
                    # support for older PEFT versions
                    module.disable_adapters = False

    def active_adapters(self) -> List[str]:
        """
        Gets the current list of active adapters of the model.

        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._pp_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from ppdiffusers.peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_sublayers(include_self=True):
            if isinstance(module, BaseTunerLayer):
                return module.active_adapter

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Optional[Callable] = None,
        max_shard_size: Union[int, str] = "10GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        push_to_hub: bool = False,
        save_to_aistudio: bool = False,
        to_diffusers: Optional[bool] = None,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory so that it can be reloaded using the
        [`~models.ModelMixin.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save a model and its configuration file to. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face Hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        # distributed kwargs
        merge_tensor_parallel = kwargs.get("merge_tensor_parallel", False)
        tensor_parallel_degree = kwargs.pop("tensor_parallel_degree", 1)

        if to_diffusers is None:
            to_diffusers = TO_DIFFUSERS

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        # create repo
        commit_message = kwargs.pop("commit_message", None)
        private = kwargs.pop("private", False)
        create_pr = kwargs.pop("create_pr", False)
        token = kwargs.pop("token", None)
        token_kwargs = {}
        if token is not None:
            token_kwargs["token"] = token
        repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
        license = kwargs.pop("license", "creativeml-openrail-m")
        exist_ok = kwargs.pop("exist_ok", True)

        if push_to_hub:
            repo_id = create_repo(repo_id, exist_ok=True, private=private, **token_kwargs).repo_id

        if save_to_aistudio:
            assert "/" in repo_id, "Please specify the repo id in format of `user_id/repo_name`"
            res = aistudio_create_repo(repo_id=repo_id, private=private, license=license, **token_kwargs)
            if "error_code" in res:
                if res["error_code"] == 10003 and exist_ok:
                    logger.info(
                        f"Repo {repo_id} already exists, it will override files with the same name. To avoid this, please set exist_ok=False"
                    )
                else:
                    logger.error(
                        f"Failed to create repo {repo_id}, error_code: {res['error_code']}, error_msg: {res['error_msg']}"
                    )
            else:
                logger.info(f"Successfully created repo {repo_id}")

        # Only save the model itself if we are using distributed training
        model_to_save = self

        # Attach architecture to the config
        # Save the config
        if is_main_process:
            model_to_save.save_config(save_directory, to_diffusers=to_diffusers)

        # Save the model
        state_dict = model_to_save.state_dict()
        if tensor_parallel_degree > 1:
            if merge_tensor_parallel:
                config_to_save = model_to_save._internal_dict
                state_dict = model_to_save.merge_tensor_parallel(state_dict, config_to_save)
                tensor_parallel_degree = 1
                if paddle.distributed.fleet.get_hybrid_communicate_group().get_model_parallel_rank() != 0:
                    logger.info("Saving with merge_tensor_parallel, tensor_parallel_rank > 0 don't need save")
                    return

        if to_diffusers:
            if not is_torch_available() and not safe_serialization:
                safe_serialization = True
                logger.warning(
                    "PyTorch is not installed, and `safe_serialization` is currently set to `False`. "
                    "To ensure proper model saving, we will automatically set `safe_serialization=True`. "
                    "If you want to keep `safe_serialization=False`, please make sure PyTorch is installed."
                )
            if safe_serialization:
                save_index_file = TORCH_SAFETENSORS_WEIGHTS_NAME_INDEX_NAME
                weights_name = TORCH_SAFETENSORS_WEIGHTS_NAME
                if is_torch_available():
                    save_function = partial(torch_safe_save_file, metadata={"format": "pt"})
                else:
                    save_function = partial(np_safe_save_file, metadata={"format": "pt"})
            else:
                save_index_file = TORCH_WEIGHTS_NAME_INDEX_NAME
                weights_name = TORCH_WEIGHTS_NAME
                save_function = torch.save
        else:
            if safe_serialization:
                save_index_file = PADDLE_SAFETENSORS_WEIGHTS_NAME_INDEX_NAME
                weights_name = PADDLE_SAFETENSORS_WEIGHTS_NAME
                save_function = partial(np_safe_save_file, metadata={"format": "pd"})
            else:
                save_index_file = PADDLE_WEIGHTS_NAME_INDEX_NAME
                weights_name = PADDLE_WEIGHTS_NAME
                save_function = paddle.save

        weights_name = _add_variant(weights_name, variant)

        # Save model
        shards, index = shard_checkpoint(
            state_dict,
            max_shard_size=max_shard_size,
            weights_name=weights_name,
        )
        # Save the model
        for shard_file, shard in shards.items():
            for k in list(shard.keys()):
                if isinstance(shard[k], paddle.Tensor):
                    shard[k] = np.ascontiguousarray(shard.pop(k).cpu().numpy())
            if to_diffusers:
                convert_paddle_state_dict_to_pytorch(self, shard)
            save_function(shard, os.path.join(save_directory, shard_file))

        # Save the model
        if index is None:
            logger.info(f"Model weights saved in {os.path.join(save_directory, weights_name)}")

        else:
            save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )
        # upload to aistudio or huggingface hub
        if save_to_aistudio:
            self._upload_folder_aistudio(
                save_directory,
                repo_id,
                commit_message=commit_message,
                **token_kwargs,
            )
        if push_to_hub:
            self._upload_folder(
                save_directory,
                repo_id,
                commit_message=commit_message,
                create_pr=create_pr,
                **token_kwargs,
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a pretrained PyTorch model from a pretrained model configuration.

        The model is set in evaluation mode - `model.eval()` - by default, and dropout modules are deactivated. To
        train the model, set it back in training mode with `model.train()`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`~ModelMixin.save_pretrained`].

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            paddle_dtype (`str` or `paddle.dtype`, *optional*):
                Override the default `paddle.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info (`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            from_flax (`bool`, *optional*, defaults to `False`):
                Load the model weights from a Flax checkpoint save file.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if `device_map` contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            variant (`str`, *optional*):
                Load weights from a specified `variant` filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the `safetensors` weights are downloaded if they're available **and** if the
                `safetensors` library is installed. If set to `True`, the model is forcibly loaded from `safetensors`
                weights. If set to `False`, `safetensors` weights are not loaded.

        <Tip>

        To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with
        `huggingface-cli login`. You can also activate the special
        ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use this method in a
        firewalled environment.

        </Tip>

        Example:

        ```py
        from ppdiffusers import UNet2DConditionModel

        unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
        ```

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```bash
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```
        """
        from_hf_hub = kwargs.pop("from_hf_hub", FROM_HF_HUB)
        from_aistudio = kwargs.pop("from_aistudio", FROM_AISTUDIO)
        cache_dir = kwargs.pop("cache_dir", None)
        if cache_dir is None:
            if from_aistudio:
                cache_dir = None  # TODO, check aistudio cache
            elif from_hf_hub:
                cache_dir = DIFFUSERS_CACHE
            else:
                cache_dir = PPDIFFUSERS_CACHE
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_diffusers = kwargs.pop("from_diffusers", FROM_DIFFUSERS)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        paddle_dtype = kwargs.pop("paddle_dtype", None)
        subfolder = kwargs.pop("subfolder", "")
        if subfolder is None:
            subfolder = ""
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        map_location = kwargs.pop("map_location", None)

        # distributed kwargs
        tensor_parallel_degree = kwargs.pop("tensor_parallel_degree", 1)

        if use_safetensors is None:
            use_safetensors = True

        if low_cpu_mem_usage and (not is_paddle_version(">=", "2.5.0") and not is_paddle_version("==", "0.0.0")):
            raise NotImplementedError(
                "Low memory initialization requires paddlepaddle-gpu >= 2.5.0. Please either update your PaddlePaddle version or set"
                " `low_cpu_mem_usage=False`."
            )

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        user_agent = {
            "ppdiffusers": __version__,
            "file_type": "model",
            "framework": "pytorch" if from_diffusers else "paddle",
        }

        # load config
        config, unused_kwargs, commit_hash, config_file = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            return_config_file=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            revision=revision,
            subfolder=subfolder,
            device_map=device_map,
            max_memory=max_memory,
            offload_folder=offload_folder,
            offload_state_dict=offload_state_dict,
            user_agent=user_agent,
            from_hf_hub=from_hf_hub,
            from_aistudio=from_aistudio,
            **kwargs,
        )
        index_file = None

        variant_list = [variant]
        if None not in variant_list:
            variant_list.append(None)
        if "fp16" not in variant_list:
            variant_list.append("fp16")
        if "fp32" not in variant_list:
            variant_list.append("fp32")
        for v_index, variant in enumerate(variant_list):
            try:
                if use_safetensors:
                    try:
                        # is sharded model
                        index_file = _get_model_file(
                            pretrained_model_name_or_path,
                            weights_name=_add_variant(TORCH_SAFETENSORS_WEIGHTS_NAME_INDEX_NAME, variant)
                            if from_diffusers
                            else _add_variant(PADDLE_SAFETENSORS_WEIGHTS_NAME_INDEX_NAME, variant),
                            cache_dir=cache_dir,
                            force_download=force_download,
                            resume_download=resume_download,
                            proxies=proxies,
                            local_files_only=local_files_only,
                            use_auth_token=use_auth_token,
                            revision=revision,
                            subfolder=subfolder,
                            user_agent=user_agent,
                            commit_hash=commit_hash,
                            from_hf_hub=from_hf_hub,
                            from_aistudio=from_aistudio,
                        )
                    except Exception:
                        index_file = None
                if index_file is None:
                    # is sharded model
                    try:
                        index_file = _get_model_file(
                            pretrained_model_name_or_path,
                            weights_name=_add_variant(TORCH_WEIGHTS_NAME_INDEX_NAME, variant)
                            if from_diffusers
                            else _add_variant(PADDLE_WEIGHTS_NAME_INDEX_NAME, variant),
                            cache_dir=cache_dir,
                            force_download=force_download,
                            resume_download=resume_download,
                            proxies=proxies,
                            local_files_only=local_files_only,
                            use_auth_token=use_auth_token,
                            revision=revision,
                            subfolder=subfolder,
                            user_agent=user_agent,
                            commit_hash=commit_hash,
                            from_hf_hub=from_hf_hub,
                            from_aistudio=from_aistudio,
                        )
                    except Exception:
                        index_file = None
                is_sharded = index_file is not None

                if is_sharded:
                    resolved_model_files, sharded_metadata = get_checkpoint_shard_files(
                        pretrained_model_name_or_path,
                        index_filename=index_file,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        commit_hash=commit_hash,
                        from_hf_hub=from_hf_hub,
                        from_aistudio=from_aistudio,
                    )
                    if not isinstance(resolved_model_files, list):
                        resolved_model_files = [resolved_model_files]
                else:
                    # load model
                    model_file = None
                    if use_safetensors:
                        try:
                            model_file = _get_model_file(
                                pretrained_model_name_or_path,
                                weights_name=_add_variant(TORCH_SAFETENSORS_WEIGHTS_NAME, variant)
                                if from_diffusers
                                else _add_variant(PADDLE_SAFETENSORS_WEIGHTS_NAME, variant),
                                cache_dir=cache_dir,
                                force_download=force_download,
                                resume_download=resume_download,
                                proxies=proxies,
                                local_files_only=local_files_only,
                                use_auth_token=use_auth_token,
                                revision=revision,
                                subfolder=subfolder,
                                user_agent=user_agent,
                                commit_hash=commit_hash,
                                from_hf_hub=from_hf_hub,
                                from_aistudio=from_aistudio,
                            )
                        except Exception:
                            model_file = None
                            pass
                    if model_file is None:
                        model_file = _get_model_file(
                            pretrained_model_name_or_path,
                            weights_name=_add_variant(TORCH_WEIGHTS_NAME, variant)
                            if from_diffusers
                            else _add_variant(PADDLE_WEIGHTS_NAME, variant),
                            cache_dir=cache_dir,
                            force_download=force_download,
                            resume_download=resume_download,
                            proxies=proxies,
                            local_files_only=local_files_only,
                            use_auth_token=use_auth_token,
                            revision=revision,
                            subfolder=subfolder,
                            user_agent=user_agent,
                            commit_hash=commit_hash,
                            from_hf_hub=from_hf_hub,
                            from_aistudio=from_aistudio,
                        )
                    resolved_model_files = [model_file]
            except Exception as e:  # NOQA
                logger.warning(
                    f"Unable to load the `variant={variant}` of the model from `{pretrained_model_name_or_path}`! "
                    "Please make sure the specified variant exists and is correct."
                )
                resolved_model_files = []
            if len(resolved_model_files) > 0:
                if v_index > 0:
                    name = (
                        ", ".join([config_file, index_file] + resolved_model_files)
                        if index_file is not None
                        else ", ".join(resolved_model_files)
                    )
                    logger.warning(
                        f"Proceeding to load the `variant={variant}` of the model with the resolved model files: {name}. "
                        "Please note that this might not be the desired variant."
                    )
                break
        variant_str = ", ".join(map(lambda x: "`" + str(x) + "`", variant_list))
        assert len(resolved_model_files) > 0, (
            f"We are attempting to load the variant in [{variant_str}]. "
            f"But unfortunately, no model files were found in the path {pretrained_model_name_or_path}. "
            "Please check if the provided path is correct and ensure that it contains the necessary model files. "
            "If the issue persists, consider redownloading the model files or contacting the model provider for assistance."
        )
        init_contexts = []

        dtype = paddle.float32 if paddle_dtype is None else paddle_dtype
        init_contexts.append(paddle.dtype_guard(dtype))

        if low_cpu_mem_usage:
            # Instantiate model.
            init_contexts.append(no_init_weights(_enable=True))
            if hasattr(paddle, "LazyGuard"):
                init_contexts.append(paddle.LazyGuard())

        with ContextManagers(init_contexts):
            model = cls.from_config(config, **unused_kwargs)

        # (westfish) 2024/04/01:
        #  Tensor parallel is only supported for models that inherit from `ConversionMixin`
        if tensor_parallel_degree > 1:
            from paddlenlp.transformers.conversion_utils import ConversionMixin

            if not issubclass(cls, ConversionMixin):
                raise NotImplementedError(
                    "Tensor parallel is only supported for models that inherit from `ConversionMixin`."
                )
            if len(resolved_model_files) > 1:
                raise NotImplementedError("Tensor parallel is not supported for multiple shards yet.")
            tmp_state_dict = smart_load(resolved_model_files[0], return_numpy=True)
            tensor_parallel_split_mapping = cls.get_tensor_parallel_convert_actions(config, tmp_state_dict.keys())
        else:
            tensor_parallel_split_mapping = None

        model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
            model,
            resolved_model_files,
            pretrained_model_name_or_path,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            ignore_keys=ignore_keys,
            from_diffusers=from_diffusers,
            tensor_parallel_split_mapping=tensor_parallel_split_mapping,
            tensor_parallel_degree=tensor_parallel_degree,
            map_location=map_location,
        )

        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_keys": mismatched_keys,
            "error_msgs": error_msgs,
        }

        if paddle_dtype is not None:
            model = model.to(dtype=paddle_dtype)

        model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()
        if output_loading_info:
            return model, loading_info

        return model

    @classmethod
    def custom_modify_weight(cls, model_to_load, state_dict):
        pass

    @classmethod
    def _load_pretrained_model(
        cls,
        model: "ModelMixin",
        resolved_model_files,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        ignore_mismatched_sizes: bool = False,
        ignore_keys=None,
        from_diffusers=False,
        tensor_parallel_split_mapping=None,
        tensor_parallel_degree=1,
        map_location=None,
    ):
        state_dict = OrderedDict()
        model_state_dict = model.state_dict()
        loaded_keys = []
        expected_keys = list(model_state_dict.keys())
        error_msgs = []
        mismatched_keys = []

        if len(resolved_model_files) > 1:
            resolved_model_files = tqdm(resolved_model_files, desc="Loading checkpoint shards")
            if tensor_parallel_degree > 1:
                raise NotImplementedError("Tensor parallel is not supported for multiple shards yet.")

        # load shard state dict
        for shard_file in resolved_model_files:
            data_format = load_state_dict(
                shard_file,
                state_dict,  # inplace update state_dict
                tensor_parallel_split_mapping=tensor_parallel_split_mapping,
                ignore_keys=ignore_keys,
                map_location=map_location,
            )
            # NOTE: new add support old state_dict
            model._update_deprecated_state_dict(state_dict)
            # NOTE: convert old model state dict!
            model._convert_deprecated_attention_blocks(state_dict)

            # NOTE: convert torch model state dict!
            if from_diffusers or data_format in ["pt"]:
                convert_pytorch_state_dict_to_paddle(model, state_dict)

            original_loaded_keys = list(state_dict.keys())
            loaded_keys.extend(original_loaded_keys)

            # Make sure we are able to load base models as well as derived models (with heads)
            model_to_load = model

            def _find_mismatched_keys(
                state_dict,
                model_state_dict,
                loaded_keys,
                ignore_mismatched_sizes,
            ):
                mismatched_keys = []
                for checkpoint_key in loaded_keys:
                    model_key = checkpoint_key

                    if model_key in model_state_dict and list(state_dict[checkpoint_key].shape) != list(
                        model_state_dict[model_key].shape
                    ):
                        mismatched_keys.append(
                            (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                        )
                        del state_dict[checkpoint_key]
                if ignore_mismatched_sizes:
                    mismatched_keys = []
                return mismatched_keys

            if state_dict is not None and len(state_dict) > 0:
                _mismatched_keys = _find_mismatched_keys(
                    state_dict,
                    model_state_dict,
                    original_loaded_keys,
                    ignore_mismatched_sizes,
                )
                mismatched_keys.extend(_mismatched_keys)
                for key_name, loaded_shape, model_shape in _mismatched_keys:
                    error_msgs.append(
                        f"Error size mismatch, {key_name} receives a shape {loaded_shape}, but the expected shape is {model_shape}."
                    )
                cls.custom_modify_weight(model_to_load, state_dict)
                faster_set_state_dict(model_to_load, state_dict)

        missing_keys = sorted(list(set(expected_keys) - set(loaded_keys)))
        unexpected_keys = sorted(list(set(loaded_keys) - set(expected_keys)))

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task"
                " or with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly"
                " identical (initializing a BertForSequenceClassification model from a"
                " BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the"
                f" checkpoint was trained on, you can already use {model.__class__.__name__} for predictions"
                " without further training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be"
                " able to use it for predictions and inference."
            )
        del state_dict
        gc.collect()
        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs

    @property
    def device(self):
        """
        `paddle.place`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> paddle.dtype:
        """
        `paddle.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """
        Get number of (trainable or non-embedding) parameters in the module.

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters.

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embedding parameters.

        Returns:
            `int`: The number of parameters.

        Example:
        ```py
        from ppdiffusers import UNet2DConditionModel
        model_id = "runwayml/stable-diffusion-v1-5"
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        unet.num_parameters(only_trainable=True)
        859520964
        ```
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight"
                for name, module_type in self.named_sublayers(include_self=True)
                if isinstance(module_type, nn.Embedding)
            ]
            non_embedding_parameters = [
                parameter for name, parameter in self.named_parameters() if name not in embedding_param_names
            ]
            return sum(p.numel() for p in non_embedding_parameters if not p.stop_gradient or not only_trainable)
        else:
            return sum(p.numel() for p in self.parameters() if not p.stop_gradient or not only_trainable)

    def _convert_deprecated_attention_blocks(self, state_dict: OrderedDict) -> None:
        deprecated_attention_block_paths = []

        def recursive_find_attn_block(name, module):
            if hasattr(module, "_from_deprecated_attn_block") and module._from_deprecated_attn_block:
                deprecated_attention_block_paths.append(name)

            for sub_name, sub_module in module.named_children():
                sub_name = sub_name if name == "" else f"{name}.{sub_name}"
                recursive_find_attn_block(sub_name, sub_module)

        recursive_find_attn_block("", self)

        # NOTE: we have to check if the deprecated parameters are in the state dict
        # because it is possible we are loading from a state dict that was already
        # converted

        for path in deprecated_attention_block_paths:
            # group_norm path stays the same

            # query -> to_q
            if f"{path}.query.weight" in state_dict:
                state_dict[f"{path}.to_q.weight"] = state_dict.pop(f"{path}.query.weight")
            if f"{path}.query.bias" in state_dict:
                state_dict[f"{path}.to_q.bias"] = state_dict.pop(f"{path}.query.bias")

            # key -> to_k
            if f"{path}.key.weight" in state_dict:
                state_dict[f"{path}.to_k.weight"] = state_dict.pop(f"{path}.key.weight")
            if f"{path}.key.bias" in state_dict:
                state_dict[f"{path}.to_k.bias"] = state_dict.pop(f"{path}.key.bias")

            # value -> to_v
            if f"{path}.value.weight" in state_dict:
                state_dict[f"{path}.to_v.weight"] = state_dict.pop(f"{path}.value.weight")
            if f"{path}.value.bias" in state_dict:
                state_dict[f"{path}.to_v.bias"] = state_dict.pop(f"{path}.value.bias")

            # proj_attn -> to_out.0
            if f"{path}.proj_attn.weight" in state_dict:
                state_dict[f"{path}.to_out.0.weight"] = state_dict.pop(f"{path}.proj_attn.weight")
            if f"{path}.proj_attn.bias" in state_dict:
                state_dict[f"{path}.to_out.0.bias"] = state_dict.pop(f"{path}.proj_attn.bias")

    def _temp_convert_self_to_deprecated_attention_blocks(self) -> None:
        deprecated_attention_block_modules = []

        def recursive_find_attn_block(module):
            if hasattr(module, "_from_deprecated_attn_block") and module._from_deprecated_attn_block:
                deprecated_attention_block_modules.append(module)

            for sub_module in module.children():
                recursive_find_attn_block(sub_module)

        recursive_find_attn_block(self)

        for module in deprecated_attention_block_modules:
            module.query = module.to_q
            module.key = module.to_k
            module.value = module.to_v
            module.proj_attn = module.to_out[0]

            # We don't _have_ to delete the old attributes, but it's helpful to ensure
            # that _all_ the weights are loaded into the new attributes and we're not
            # making an incorrect assumption that this model should be converted when
            # it really shouldn't be.
            del module.to_q
            del module.to_k
            del module.to_v
            del module.to_out

    def _undo_temp_convert_self_to_deprecated_attention_blocks(self) -> None:
        deprecated_attention_block_modules = []

        def recursive_find_attn_block(module) -> None:
            if hasattr(module, "_from_deprecated_attn_block") and module._from_deprecated_attn_block:
                deprecated_attention_block_modules.append(module)

            for sub_module in module.children():
                recursive_find_attn_block(sub_module)

        recursive_find_attn_block(self)

        for module in deprecated_attention_block_modules:
            module.to_q = module.query
            module.to_k = module.key
            module.to_v = module.value
            module.to_out = nn.LayerList([module.proj_attn, nn.Dropout(module.dropout)])

            del module.query
            del module.key
            del module.value
            del module.proj_attn

    @classmethod
    def _update_deprecated_state_dict(cls, state_dict=None, loaded_keys=None, model=None):
        if state_dict is None:
            return loaded_keys
        _deprecated_dict = getattr(cls, "_deprecated_dict", None)
        from_deprecated_state_dict = _deprecated_dict is not None and any(
            cls._deprecated_dict.get("key", "NONE") in all_key for all_key in state_dict.keys()
        )
        if from_deprecated_state_dict:
            logger.warning(
                "Loading from deprecated state_dict, please load new state_dict via setting `use_safetensors=True`."
            )
            for name in list(state_dict.keys()):
                deprecated_name = name
                for old_name, new_name in cls._deprecated_dict.get("name_mapping", {}).items():
                    name = name.replace(old_name, new_name)
                state_dict[name] = state_dict.pop(deprecated_name)
            loaded_keys = list(state_dict.keys())
        return loaded_keys
