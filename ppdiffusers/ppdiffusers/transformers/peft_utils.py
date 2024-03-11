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
import warnings
from typing import Any, Dict, List, Optional, Union

import paddle

from ppdiffusers.utils import (
    MIN_PEFT_VERSION,
    check_peft_version,
    is_peft_available,
    logging,
)
from ppdiffusers.utils.downloader import bos_aistudio_hf_download

logger = logging.get_logger(__name__)

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_TORCH_WEIGHTS_NAME = "adapter_model.bin"
ADAPTER_WEIGHTS_NAME = "adapter_model.pdparams"
ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"


def find_adapter_config_file(
    model_id: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
    _commit_hash: Optional[str] = None,
    from_bos: bool = True,
    from_aistudio: bool = False,
    from_hf_hub: bool = False,
) -> Optional[str]:
    r"""
    Simply checks if the model stored on the Hub or locally is an adapter model or not, return the path of the adapter
    config file if it is, None otherwise.

    Args:
        model_id (`str`):
            The identifier of the model to look for, can be either a local path or an id to the repository on the Hub.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.

            <Tip>

            To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

            </Tip>

        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.
    """
    adapter_cached_filename = None
    if model_id is None:
        return None
    elif os.path.isdir(model_id):
        list_remote_files = os.listdir(model_id)
        if ADAPTER_CONFIG_NAME in list_remote_files:
            adapter_cached_filename = os.path.join(model_id, ADAPTER_CONFIG_NAME)
    else:
        adapter_cached_filename = bos_aistudio_hf_download(
            model_id,
            ADAPTER_CONFIG_NAME,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            revision=revision,
            local_files_only=local_files_only,
            subfolder=subfolder,
            from_bos=from_bos,
            from_aistudio=from_aistudio,
            from_hf_hub=from_hf_hub,
        )

    return adapter_cached_filename


class PeftAdapterMixin:
    """
    A class containing all functions for loading and using adapters weights that are supported in PEFT library. For
    more details about adapters and injecting them on a transformer-based model, check out the documentation of PEFT
    library: https://huggingface.co/docs/peft/index

    Currently supported PEFT methods are all non-prefix tuning methods. Below is the list of supported PEFT methods
    that anyone can load, train and run with this mixin class:
    - Low Rank Adapters (LoRA): https://huggingface.co/docs/peft/conceptual_guides/lora
    - IA3: https://huggingface.co/docs/peft/conceptual_guides/ia3
    - AdaLora: https://arxiv.org/abs/2303.10512

    Other PEFT models such as prompt tuning, prompt learning are out of scope as these adapters are not "injectable"
    into a torch module. For using these methods, please refer to the usage guide of PEFT library.

    With this mixin, if the correct PEFT version is installed, it is possible to:

    - Load an adapter stored on a local path or in a remote Hub repository, and inject it in the model
    - Attach new adapters in the model and train them with Trainer or by your own.
    - Attach multiple adapters and iteratively activate / deactivate them
    - Activate / deactivate all adapters from the model.
    - Get the `state_dict` of the active adapter.
    """

    _pp_peft_config_loaded = False

    def load_adapter(
        self,
        peft_model_id: Optional[str] = None,
        adapter_name: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        peft_config: Dict[str, Any] = None,
        adapter_state_dict: Optional[Dict[str, "paddle.Tensor"]] = None,
        adapter_kwargs: Optional[Dict[str, Any]] = None,
        from_diffusers: bool = False,
        from_bos: bool = True,
        from_aistudio: bool = False,
        from_hf_hub: bool = False,
    ) -> None:
        """
        Load adapter weights from file or remote Hub folder. If you are not familiar with adapters and PEFT methods, we
        invite you to read more about them on PEFT official documentation: https://huggingface.co/docs/peft

        Requires peft as a backend to load the adapter weights.

        Args:
            peft_model_id (`str`, *optional*):
                The identifier of the model to look for on the Hub, or a local path to the saved adapter config file
                and adapter weights.
            adapter_name (`str`, *optional*):
                The adapter name to use. If not set, will use the default adapter.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

                </Tip>

            token (`str`, `optional`):
                Whether to use authentication token to load the remote folder. Userful to load private repositories
                that are on HuggingFace Hub. You might need to call `huggingface-cli login` and paste your tokens to
                cache it.
            peft_config (`Dict[str, Any]`, *optional*):
                The configuration of the adapter to add, supported adapters are non-prefix tuning and adaption prompts
                methods. This argument is used in case users directly pass PEFT state dicts
            adapter_state_dict (`Dict[str, torch.Tensor]`, *optional*):
                The state dict of the adapter to load. This argument is used in case users directly pass PEFT state
                dicts
            adapter_kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the `from_pretrained` method of the adapter config and
                `find_adapter_config_file` method.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        adapter_name = adapter_name if adapter_name is not None else "default"
        if adapter_kwargs is None:
            adapter_kwargs = {}

        adapter_kwargs["from_bos"] = from_bos
        adapter_kwargs["from_aistudio"] = from_aistudio
        adapter_kwargs["from_hf_hub"] = from_hf_hub

        from ppdiffusers.peft import (
            PeftConfig,
            inject_adapter_in_model,
            load_peft_weights,
        )
        from ppdiffusers.peft.utils import set_peft_model_state_dict

        if self._pp_peft_config_loaded and adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        if peft_model_id is None and (adapter_state_dict is None and peft_config is None):
            raise ValueError(
                "You should either pass a `peft_model_id` or a `peft_config` and `adapter_state_dict` to load an adapter."
            )

        # We keep `revision` in the signature for backward compatibility
        if revision is not None and "revision" not in adapter_kwargs:
            adapter_kwargs["revision"] = revision
        elif revision is not None and "revision" in adapter_kwargs and revision != adapter_kwargs["revision"]:
            logger.error(
                "You passed a `revision` argument both in `adapter_kwargs` and as a standalone argument. "
                "The one in `adapter_kwargs` will be used."
            )

        # Override token with adapter_kwargs' token
        if "token" in adapter_kwargs:
            token = adapter_kwargs.pop("token")

        if peft_config is None:
            adapter_config_file = find_adapter_config_file(
                peft_model_id,
                token=token,
                **adapter_kwargs,
            )

            if adapter_config_file is None:
                raise ValueError(
                    f"adapter model file not found in {peft_model_id}. Make sure you are passing the correct path to the "
                    "adapter model."
                )

            peft_config = PeftConfig.from_pretrained(
                peft_model_id,
                token=token,
                **adapter_kwargs,
            )

        # Create and add fresh new adapters into the model.
        inject_adapter_in_model(peft_config, self, adapter_name)

        if not self._pp_peft_config_loaded:
            self._pp_peft_config_loaded = True

        if peft_model_id is not None:
            adapter_state_dict, from_diffusers = load_peft_weights(
                peft_model_id, token=token, from_diffusers=from_diffusers, **adapter_kwargs
            )

        # We need to pre-process the state dict to remove unneeded prefixes - for backward compatibility
        processed_adapter_state_dict = {}
        prefix = "base_model.model."
        for key, value in adapter_state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
            else:
                new_key = key
            processed_adapter_state_dict[new_key] = value

        # Load state dict
        incompatible_keys = set_peft_model_state_dict(
            self, processed_adapter_state_dict, adapter_name, from_diffusers=from_diffusers
        )

        if incompatible_keys is not None:
            # check only for unexpected keys
            if hasattr(incompatible_keys, "unexpected_keys") and len(incompatible_keys.unexpected_keys) > 0:
                logger.warning(
                    f"Loading adapter weights from {peft_model_id} led to unexpected keys not found in the model: "
                    f" {incompatible_keys.unexpected_keys}. "
                )

    def add_adapter(self, adapter_config, adapter_name: Optional[str] = None) -> None:
        r"""
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Adds a fresh new adapter to the current model for training purpose. If no adapter name is passed, a default
        name is assigned to the adapter to follow the convention of PEFT library (in PEFT we use "default" as the
        default adapter name).

        Args:
            adapter_config (`~peft.PeftConfig`):
                The configuration of the adapter to add, supported adapters are non-prefix tuning and adaption prompts
                methods
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to add. If no name is passed, a default name is assigned to the adapter.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        from peft import PeftConfig, inject_adapter_in_model

        adapter_name = adapter_name or "default"

        if not self._pp_peft_config_loaded:
            self._pp_peft_config_loaded = True
        elif adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        if not isinstance(adapter_config, PeftConfig):
            raise ValueError(
                f"adapter_config should be an instance of PeftConfig. Got {type(adapter_config)} instead."
            )

        # Retrieve the name or path of the model, one could also use self.config._name_or_path
        # but to be consistent with what we do in PEFT: https://github.com/huggingface/peft/blob/6e783780ca9df3a623992cc4d1d665001232eae0/src/peft/mapping.py#L100
        adapter_config.base_model_name_or_path = self.__dict__.get("name_or_path", None)
        inject_adapter_in_model(adapter_config, self, adapter_name)

        self.set_adapter(adapter_name)

    def set_adapter(self, adapter_name: Union[List[str], str]) -> None:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Sets a specific adapter by forcing the model to use a that adapter and disable the other adapters.

        Args:
            adapter_name (`Union[List[str], str]`):
                The name of the adapter to set. Can be also a list of strings to set multiple adapters.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)
        if not self._pp_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
        elif isinstance(adapter_name, list):
            missing = set(adapter_name) - set(self.peft_config)
            if len(missing) > 0:
                raise ValueError(
                    f"Following adapter(s) could not be found: {', '.join(missing)}. Make sure you are passing the correct adapter name(s)."
                    f" current loaded adapters are: {list(self.peft_config.keys())}"
                )
        elif adapter_name not in self.peft_config:
            raise ValueError(
                f"Adapter with name {adapter_name} not found. Please pass the correct adapter name among {list(self.peft_config.keys())}"
            )

        from ppdiffusers.peft.tuners.tuners_utils import BaseTunerLayer
        from ppdiffusers.peft.utils import ModulesToSaveWrapper

        _adapters_has_been_set = False

        for _, module in self.named_sublayers(include_self=True):
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                # For backward compatbility with previous PEFT versions
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                else:
                    module.active_adapter = adapter_name
                _adapters_has_been_set = True

        if not _adapters_has_been_set:
            raise ValueError(
                "Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters."
            )

    def disable_adapters(self) -> None:
        r"""
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Disable all adapters that are attached to the model. This leads to inferring with the base model only.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._pp_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from ppdiffusers.peft.tuners.tuners_utils import BaseTunerLayer
        from ppdiffusers.peft.utils import ModulesToSaveWrapper

        for _, module in self.named_sublayers(include_self=True):
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                # The recent version of PEFT need to call `enable_adapters` instead
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=False)
                else:
                    module.disable_adapters = True

    def enable_adapters(self) -> None:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Enable adapters that are attached to the model. The model will use `self.active_adapter()`
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._pp_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from ppdiffusers.peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_sublayers(include_self=True):
            if isinstance(module, BaseTunerLayer):
                # The recent version of PEFT need to call `enable_adapters` instead
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=True)
                else:
                    module.disable_adapters = False

    def active_adapters(self) -> List[str]:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Gets the current active adapters of the model. In case of multi-adapter inference (combining multiple adapters
        for inference) returns the list of all active adapters so that users can deal with them accordingly.

        For previous PEFT versions (that does not support multi-adapter inference), `module.active_adapter` will return
        a single string.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not is_peft_available():
            raise ImportError("PEFT is not available. Please install PEFT to use this function: `pip install peft`.")

        if not self._pp_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_sublayers(include_self=True):
            if isinstance(module, BaseTunerLayer):
                active_adapters = module.active_adapter
                break

        # For previous PEFT versions
        if isinstance(active_adapters, str):
            active_adapters = [active_adapters]

        return active_adapters

    def active_adapter(self) -> str:
        warnings.warn(
            "The `active_adapter` method is deprecated and will be removed in a future version.", FutureWarning
        )

        return self.active_adapters()[0]

    def get_adapter_state_dict(self, adapter_name: Optional[str] = None) -> dict:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Gets the adapter state dict that should only contain the weights tensors of the specified adapter_name adapter.
        If no adapter_name is passed, the active adapter is used.

        Args:
            adapter_name (`str`, *optional*):
                The name of the adapter to get the state dict from. If no name is passed, the active adapter is used.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._pp_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft import get_peft_model_state_dict

        if adapter_name is None:
            adapter_name = self.active_adapter()

        adapter_state_dict = get_peft_model_state_dict(self, adapter_name=adapter_name)
        return adapter_state_dict
