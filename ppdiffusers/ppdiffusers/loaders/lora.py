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
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import paddle
from huggingface_hub import model_info
from packaging import version
from paddle import nn

from ..models.modeling_pytorch_paddle_utils import (
    convert_paddle_state_dict_to_pytorch,
    convert_pytorch_state_dict_to_paddle,
)
from ..models.modeling_utils import faster_set_state_dict
from ..utils import (
    DIFFUSERS_CACHE,
    FROM_AISTUDIO,
    FROM_DIFFUSERS,
    FROM_HF_HUB,
    HF_HUB_OFFLINE,
    LOW_CPU_MEM_USAGE_DEFAULT,
    PPDIFFUSERS_CACHE,
    TO_DIFFUSERS,
    USE_PEFT_BACKEND,
    _get_model_file,
    convert_state_dict_to_peft,
    convert_state_dict_to_ppdiffusers,
    convert_unet_state_dict_to_peft,
    delete_adapter_layers,
    deprecate,
    get_adapter_name,
    get_peft_kwargs,
    is_paddlenlp_available,
    is_safetensors_available,
    is_torch_available,
    logging,
    recurse_remove_peft_layers,
    scale_lora_layers,
    set_adapter_layers,
    set_weights_and_activate_adapters,
)
from ..version import VERSION as __version__
from .lora_conversion_utils import (
    _convert_kohya_lora_to_diffusers,
    _maybe_map_sgm_blocks_to_diffusers,
)

if is_safetensors_available():
    from safetensors.numpy import save_file as np_safe_save_file

    if is_torch_available():
        from safetensors.torch import save_file as torch_safe_save_file

if is_torch_available():
    import torch

if is_paddlenlp_available():
    from paddlenlp.transformers import PretrainedModel

    from ..models.lora import (
        PatchedLoraProjection,
        text_encoder_attn_modules,
        text_encoder_mlp_modules,
    )


logger = logging.get_logger(__name__)

TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"
TRANSFORMER_NAME = "transformer"

TORCH_LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
TORCH_LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"

PADDLE_LORA_WEIGHT_NAME = "paddle_lora_weights.pdparams"
PADDLE_LORA_WEIGHT_NAME_SAFE = "paddle_lora_weights.safetensors"


LORA_DEPRECATION_MESSAGE = "You are using an old version of LoRA backend. This will be deprecated in the next releases in favor of PEFT make sure to install the latest PEFT and transformers packages in the future."


class LoraLoaderMixin:
    r"""
    Load LoRA layers into [`UNet2DConditionModel`] and
    [`CLIPTextModel`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel).
    """

    text_encoder_name = TEXT_ENCODER_NAME
    unet_name = UNET_NAME
    num_fused_loras = 0

    def load_lora_weights(
        self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, paddle.Tensor]], adapter_name=None, **kwargs
    ):
        """
        Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.unet` and
        `self.text_encoder`.

        All kwargs are forwarded to `self.lora_state_dict`.

        See [`~loaders.LoraLoaderMixin.lora_state_dict`] for more details on how the state dict is loaded.

        See [`~loaders.LoraLoaderMixin.load_lora_into_unet`] for more details on how the state dict is loaded into
        `self.unet`.

        See [`~loaders.LoraLoaderMixin.load_lora_into_text_encoder`] for more details on how the state dict is loaded
        into `self.text_encoder`.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                See [`~loaders.LoraLoaderMixin.lora_state_dict`].
            kwargs (`dict`, *optional*):
                See [`~loaders.LoraLoaderMixin.lora_state_dict`].
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
        """
        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        state_dict, network_alphas, from_diffusers = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict, **kwargs
        )

        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", LOW_CPU_MEM_USAGE_DEFAULT)

        self.load_lora_into_unet(
            state_dict,
            network_alphas=network_alphas,
            unet=getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet,
            low_cpu_mem_usage=low_cpu_mem_usage,
            adapter_name=adapter_name,
            _pipeline=self,
            from_diffusers=from_diffusers,
        )
        self.load_lora_into_text_encoder(
            state_dict,
            network_alphas=network_alphas,
            text_encoder=getattr(self, self.text_encoder_name)
            if not hasattr(self, "text_encoder")
            else self.text_encoder,
            lora_scale=self.lora_scale,
            low_cpu_mem_usage=low_cpu_mem_usage,
            adapter_name=adapter_name,
            _pipeline=self,
            from_diffusers=from_diffusers,
        )

    @classmethod
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, paddle.Tensor]],
        **kwargs,
    ):
        r"""
        Return state dict for lora weights and the network alphas.

        <Tip warning={true}>

        We support loading A1111 formatted LoRA checkpoints in a limited capacity.

        This function is experimental and might change in the future.

        </Tip>

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.

        """
        # Load the main state dict first which has the LoRA layers for either of
        # UNet and text encoder or both.
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
        from_diffusers = kwargs.pop("from_diffusers", FROM_DIFFUSERS)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        unet_config = kwargs.pop("unet_config", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        if use_safetensors is None:
            use_safetensors = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch" if from_diffusers else "paddle",
        }

        model_file = None
        state_dict = {}
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            # Let's first try to load .safetensors weights
            if (use_safetensors and weight_name is None) or (
                weight_name is not None and weight_name.endswith(".safetensors")
            ):
                # Here we're relaxing the loading check to enable more Inference API
                # friendliness where sometimes, it's not at all possible to automatically
                # determine `weight_name`.
                # if weight_name is None:
                #     weight_name = cls._best_guess_weight_name(
                #         pretrained_model_name_or_path_or_dict, file_extension=".safetensors"
                #     )
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=(weight_name or TORCH_LORA_WEIGHT_NAME_SAFE)
                        if from_diffusers
                        else ((weight_name or PADDLE_LORA_WEIGHT_NAME_SAFE)),
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        from_aistudio=from_aistudio,
                        from_hf_hub=from_hf_hub,
                    )
                except Exception:
                    model_file = None

            if model_file is None:
                # if weight_name is None:
                #     weight_name = cls._best_guess_weight_name(
                #         pretrained_model_name_or_path_or_dict, file_extension=".bin"
                #     )
                if model_file is None:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=(weight_name or TORCH_LORA_WEIGHT_NAME)
                        if from_diffusers
                        else ((weight_name or PADDLE_LORA_WEIGHT_NAME)),
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        from_aistudio=from_aistudio,
                        from_hf_hub=from_hf_hub,
                    )

            assert model_file is not None, "Could not find the model file!"
            # TODO, check this
            from ppdiffusers.utils import smart_load

            state_dict = smart_load(model_file, return_is_torch_weight=True)
            is_torch_weight = state_dict.pop("is_torch_weight", False)
            if not from_diffusers and is_torch_weight:
                logger.warning(
                    "Detect the weight is in diffusers format, but currently, `from_diffusers` is set to `False`. To proceed, we will change the value of `from_diffusers` to `True`!"
                )
                from_diffusers = True
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        network_alphas = None
        # TODO: replace it with a method from `state_dict_utils`
        if all(
            (
                k.startswith("lora_te_")
                or k.startswith("lora_unet_")
                or k.startswith("lora_te1_")
                or k.startswith("lora_te2_")
            )
            for k in state_dict.keys()
        ):
            # Map SDXL blocks correctly.
            if unet_config is not None:
                # use unet config to remap block numbers
                state_dict = _maybe_map_sgm_blocks_to_diffusers(state_dict, unet_config)
            state_dict, network_alphas = _convert_kohya_lora_to_diffusers(state_dict)

        return state_dict, network_alphas, from_diffusers

    @classmethod
    def _best_guess_weight_name(cls, pretrained_model_name_or_path_or_dict, file_extension=".safetensors"):
        targeted_files = []

        if os.path.isfile(pretrained_model_name_or_path_or_dict):
            return
        elif os.path.isdir(pretrained_model_name_or_path_or_dict):
            targeted_files = [
                f for f in os.listdir(pretrained_model_name_or_path_or_dict) if f.endswith(file_extension)
            ]
        else:
            files_in_repo = model_info(pretrained_model_name_or_path_or_dict).siblings
            targeted_files = [f.rfilename for f in files_in_repo if f.rfilename.endswith(file_extension)]
        if len(targeted_files) == 0:
            return

        # "scheduler" does not correspond to a LoRA checkpoint.
        # "optimizer" does not correspond to a LoRA checkpoint
        # only top-level checkpoints are considered and not the other ones, hence "checkpoint".
        unallowed_substrings = {"scheduler", "optimizer", "checkpoint"}
        targeted_files = list(
            filter(lambda x: all(substring not in x for substring in unallowed_substrings), targeted_files)
        )

        if any(f.endswith(TORCH_LORA_WEIGHT_NAME) for f in targeted_files):
            targeted_files = list(filter(lambda x: x.endswith(TORCH_LORA_WEIGHT_NAME), targeted_files))
        elif any(f.endswith(TORCH_LORA_WEIGHT_NAME_SAFE) for f in targeted_files):
            targeted_files = list(filter(lambda x: x.endswith(TORCH_LORA_WEIGHT_NAME_SAFE), targeted_files))

        if len(targeted_files) > 1:
            raise ValueError(
                f"Provided path contains more than one weights file in the {file_extension} format. Either specify `weight_name` in `load_lora_weights` or make sure there's only one  `.safetensors` or `.bin` file in  {pretrained_model_name_or_path_or_dict}."
            )
        weight_name = targeted_files[0]
        return weight_name

    @classmethod
    def _optionally_disable_offloading(cls, _pipeline):
        """
        Optionally removes offloading in case the pipeline has been already sequentially offloaded to CPU.

        Args:
            _pipeline (`DiffusionPipeline`):
                The pipeline to disable offloading for.

        Returns:
            tuple:
                A tuple indicating if `is_model_cpu_offload` or `is_sequential_cpu_offload` is True.
        """
        pass

    @classmethod
    def load_lora_into_unet(
        cls,
        state_dict,
        network_alphas,
        unet,
        low_cpu_mem_usage=None,
        adapter_name=None,
        _pipeline=None,
        from_diffusers=None,
    ):
        """
        This will load the LoRA layers specified in `state_dict` into `unet`.

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The keys can either be indexed directly
                into the unet or prefixed with an additional `unet` which can be used to distinguish between text
                encoder lora layers.
            network_alphas (`Dict[str, float]`):
                See `LoRALinearLayer` for more details.
            unet (`UNet2DConditionModel`):
                The UNet model to load the LoRA layers into.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
        """
        if from_diffusers is None:
            from_diffusers = FROM_DIFFUSERS
        low_cpu_mem_usage = low_cpu_mem_usage if low_cpu_mem_usage is not None else LOW_CPU_MEM_USAGE_DEFAULT
        # If the serialization format is new (introduced in https://github.com/huggingface/diffusers/pull/2918),
        # then the `state_dict` keys should have `cls.unet_name` and/or `cls.text_encoder_name` as
        # their prefixes.
        keys = list(state_dict.keys())

        if all(key.startswith(cls.unet_name) or key.startswith(cls.text_encoder_name) for key in keys):
            # Load the layers corresponding to UNet.
            logger.info(f"Loading {cls.unet_name}.")

            unet_keys = [k for k in keys if k.startswith(cls.unet_name)]
            state_dict = {k.replace(f"{cls.unet_name}.", ""): v for k, v in state_dict.items() if k in unet_keys}

            if network_alphas is not None:
                alpha_keys = [k for k in network_alphas.keys() if k.startswith(cls.unet_name)]
                network_alphas = {
                    k.replace(f"{cls.unet_name}.", ""): v for k, v in network_alphas.items() if k in alpha_keys
                }

        else:
            # Otherwise, we're dealing with the old format. This means the `state_dict` should only
            # contain the module names of the `unet` as its keys WITHOUT any prefix.
            warn_message = "You have saved the LoRA weights using the old format. To convert the old LoRA weights to the new format, you can first load them in a dictionary and then create a new dictionary like the following: `new_state_dict = {f'unet.{module_name}': params for module_name, params in old_state_dict.items()}`."
            logger.warn(warn_message)

        if USE_PEFT_BACKEND and len(state_dict.keys()) > 0:
            from ppdiffusers.peft import (
                LoraConfig,
                inject_adapter_in_model,
                set_peft_model_state_dict,
            )

            if adapter_name in getattr(unet, "peft_config", {}):
                raise ValueError(
                    f"Adapter name {adapter_name} already in use in the Unet - please select a new adapter name."
                )

            state_dict = convert_unet_state_dict_to_peft(state_dict)

            if network_alphas is not None:
                # The alphas state dict have the same structure as Unet, thus we convert it to peft format using
                # `convert_unet_state_dict_to_peft` method.
                network_alphas = convert_unet_state_dict_to_peft(network_alphas)

            rank = {}
            index = 1 if from_diffusers else 0
            for key, val in state_dict.items():
                if "lora_B" in key:
                    rank[key] = val.shape[index]

            lora_config_kwargs = get_peft_kwargs(rank, network_alphas, state_dict, is_unet=True)
            lora_config = LoraConfig(**lora_config_kwargs)

            # adapter_name
            if adapter_name is None:
                adapter_name = get_adapter_name(unet)

            # In case the pipeline has been already offloaded to CPU - temporarily remove the hooks
            # otherwise loading LoRA weights will lead to an error

            inject_adapter_in_model(lora_config, unet, adapter_name=adapter_name)
            incompatible_keys = set_peft_model_state_dict(
                unet, state_dict, adapter_name, from_diffusers=from_diffusers
            )  # NOTE: new add from_diffusers

            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

        unet.load_attn_procs(
            state_dict,
            network_alphas=network_alphas,
            low_cpu_mem_usage=low_cpu_mem_usage,
            _pipeline=_pipeline,
            from_diffusers=from_diffusers,
        )

    @classmethod
    def load_lora_into_text_encoder(
        cls,
        state_dict,
        network_alphas,
        text_encoder,
        prefix=None,
        lora_scale=1.0,
        low_cpu_mem_usage=None,
        adapter_name=None,
        _pipeline=None,
        from_diffusers=None,
    ):
        """
        This will load the LoRA layers specified in `state_dict` into `text_encoder`

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The key should be prefixed with an
                additional `text_encoder` to distinguish between unet lora layers.
            network_alphas (`Dict[str, float]`):
                See `LoRALinearLayer` for more details.
            text_encoder (`CLIPTextModel`):
                The text encoder model to load the LoRA layers into.
            prefix (`str`):
                Expected prefix of the `text_encoder` in the `state_dict`.
            lora_scale (`float`):
                How much to scale the output of the lora linear layer before it is added with the output of the regular
                lora layer.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
        """
        if from_diffusers is None:
            from_diffusers = FROM_DIFFUSERS
        low_cpu_mem_usage = low_cpu_mem_usage if low_cpu_mem_usage is not None else LOW_CPU_MEM_USAGE_DEFAULT

        # If the serialization format is new (introduced in https://github.com/huggingface/diffusers/pull/2918),
        # then the `state_dict` keys should have `self.unet_name` and/or `self.text_encoder_name` as
        # their prefixes.
        keys = list(state_dict.keys())
        prefix = cls.text_encoder_name if prefix is None else prefix

        # Safe prefix to check with.
        if any(cls.text_encoder_name in key for key in keys):
            # Load the layers corresponding to text encoder and make necessary adjustments.
            text_encoder_keys = [k for k in keys if k.startswith(prefix) and k.split(".")[0] == prefix]
            text_encoder_lora_state_dict = {
                k.replace(f"{prefix}.", ""): v for k, v in state_dict.items() if k in text_encoder_keys
            }

            if len(text_encoder_lora_state_dict) > 0:
                logger.info(f"Loading {prefix}.")
                rank = {}
                text_encoder_lora_state_dict = convert_state_dict_to_ppdiffusers(text_encoder_lora_state_dict)
                index = 1 if from_diffusers else 0
                if USE_PEFT_BACKEND:
                    # convert state dict
                    text_encoder_lora_state_dict = convert_state_dict_to_peft(text_encoder_lora_state_dict)

                    for name, _ in text_encoder_attn_modules(text_encoder):
                        rank_key = f"{name}.out_proj.lora_B.weight"
                        rank[rank_key] = text_encoder_lora_state_dict[rank_key].shape[index]

                    patch_mlp = any(".mlp." in key for key in text_encoder_lora_state_dict.keys())
                    if patch_mlp:
                        for name, _ in text_encoder_mlp_modules(text_encoder):
                            rank_key_fc1 = f"{name}.fc1.lora_B.weight"
                            rank_key_fc2 = f"{name}.fc2.lora_B.weight"

                            rank[rank_key_fc1] = text_encoder_lora_state_dict[rank_key_fc1].shape[index]
                            rank[rank_key_fc2] = text_encoder_lora_state_dict[rank_key_fc2].shape[index]
                else:
                    for name, _ in text_encoder_attn_modules(text_encoder):
                        rank_key = f"{name}.out_proj.lora_linear_layer.up.weight"
                        rank.update({rank_key: text_encoder_lora_state_dict[rank_key].shape[index]})

                    patch_mlp = any(".mlp." in key for key in text_encoder_lora_state_dict.keys())
                    if patch_mlp:
                        for name, _ in text_encoder_mlp_modules(text_encoder):
                            rank_key_fc1 = f"{name}.fc1.lora_linear_layer.up.weight"
                            rank_key_fc2 = f"{name}.fc2.lora_linear_layer.up.weight"
                            rank[rank_key_fc1] = text_encoder_lora_state_dict[rank_key_fc1].shape[index]
                            rank[rank_key_fc2] = text_encoder_lora_state_dict[rank_key_fc2].shape[index]
                if network_alphas is not None:
                    alpha_keys = [
                        k for k in network_alphas.keys() if k.startswith(prefix) and k.split(".")[0] == prefix
                    ]
                    network_alphas = {
                        k.replace(f"{prefix}.", ""): v for k, v in network_alphas.items() if k in alpha_keys
                    }

                if USE_PEFT_BACKEND:
                    from ppdiffusers.peft import LoraConfig

                    lora_config_kwargs = get_peft_kwargs(
                        rank, network_alphas, text_encoder_lora_state_dict, is_unet=False
                    )

                    lora_config = LoraConfig(**lora_config_kwargs)

                    # adapter_name
                    if adapter_name is None:
                        adapter_name = get_adapter_name(text_encoder)

                    # inject LoRA layers and load the state dict
                    # in transformers we automatically check whether the adapter name is already in use or not
                    text_encoder.load_adapter(
                        adapter_name=adapter_name,
                        adapter_state_dict=text_encoder_lora_state_dict,
                        peft_config=lora_config,
                        from_diffusers=from_diffusers,  # new add from diffusers
                    )

                    # scale LoRA layers with `lora_scale`
                    scale_lora_layers(text_encoder, weight=lora_scale)
                else:
                    cls._modify_text_encoder(
                        text_encoder,
                        lora_scale,
                        network_alphas,
                        rank=rank,
                        patch_mlp=patch_mlp,
                        low_cpu_mem_usage=low_cpu_mem_usage,
                    )

                    if from_diffusers:
                        convert_pytorch_state_dict_to_paddle(text_encoder, text_encoder_lora_state_dict)
                    faster_set_state_dict(text_encoder, text_encoder_lora_state_dict)

                text_encoder.to(dtype=text_encoder.dtype)

    @property
    def lora_scale(self) -> float:
        # property function that returns the lora scale which can be set at run time by the pipeline.
        # if _lora_scale has not been set, return 1
        return self._lora_scale if hasattr(self, "_lora_scale") else 1.0

    def _remove_text_encoder_monkey_patch(self):
        if USE_PEFT_BACKEND:
            remove_method = recurse_remove_peft_layers
        else:
            remove_method = self._remove_text_encoder_monkey_patch_classmethod

        if hasattr(self, "text_encoder"):
            remove_method(self.text_encoder)

            # In case text encoder have no Lora attached
            if USE_PEFT_BACKEND and getattr(self.text_encoder, "peft_config", None) is not None:
                del self.text_encoder.peft_config
                self.text_encoder._pp_peft_config_loaded = None
        if hasattr(self, "text_encoder_2"):
            remove_method(self.text_encoder_2)
            if USE_PEFT_BACKEND:
                del self.text_encoder_2.peft_config
                self.text_encoder_2._pp_peft_config_loaded = None

    @classmethod
    def _remove_text_encoder_monkey_patch_classmethod(cls, text_encoder):
        if version.parse(__version__) > version.parse("0.23"):
            deprecate("_remove_text_encoder_monkey_patch_classmethod", "1.0", LORA_DEPRECATION_MESSAGE)

        for _, attn_module in text_encoder_attn_modules(text_encoder):
            if isinstance(attn_module.q_proj, PatchedLoraProjection):
                attn_module.q_proj.lora_linear_layer = None
                attn_module.k_proj.lora_linear_layer = None
                attn_module.v_proj.lora_linear_layer = None
                attn_module.out_proj.lora_linear_layer = None

        for _, mlp_module in text_encoder_mlp_modules(text_encoder):
            if isinstance(mlp_module.fc1, PatchedLoraProjection):
                mlp_module.fc1.lora_linear_layer = None
                mlp_module.fc2.lora_linear_layer = None

    @classmethod
    def _modify_text_encoder(
        cls,
        text_encoder,
        lora_scale=1,
        network_alphas=None,
        rank: Union[Dict[str, int], int] = 4,
        dtype=None,
        patch_mlp=False,
        low_cpu_mem_usage=False,
    ):
        r"""
        Monkey-patches the forward passes of attention modules of the text encoder.
        """
        if version.parse(__version__) > version.parse("0.23"):
            deprecate("_modify_text_encoder", "1.0", LORA_DEPRECATION_MESSAGE)

        def create_patched_linear_lora(model, network_alpha, rank, dtype, lora_parameters):
            linear_layer = model.regular_linear_layer if isinstance(model, PatchedLoraProjection) else model
            ctx = paddle.LazyGuard if low_cpu_mem_usage else nullcontext
            with ctx():
                model = PatchedLoraProjection(linear_layer, lora_scale, network_alpha, rank, dtype=dtype)

            lora_parameters.extend(model.lora_linear_layer.parameters())
            return model

        # First, remove any monkey-patch that might have been applied before
        cls._remove_text_encoder_monkey_patch_classmethod(text_encoder)

        lora_parameters = []
        network_alphas = {} if network_alphas is None else network_alphas
        is_network_alphas_populated = len(network_alphas) > 0

        for name, attn_module in text_encoder_attn_modules(text_encoder):
            query_alpha = network_alphas.pop(name + ".to_q_lora.down.weight.alpha", None)
            key_alpha = network_alphas.pop(name + ".to_k_lora.down.weight.alpha", None)
            value_alpha = network_alphas.pop(name + ".to_v_lora.down.weight.alpha", None)
            out_alpha = network_alphas.pop(name + ".to_out_lora.down.weight.alpha", None)

            if isinstance(rank, dict):
                current_rank = rank.pop(f"{name}.out_proj.lora_linear_layer.up.weight")
            else:
                current_rank = rank

            attn_module.q_proj = create_patched_linear_lora(
                attn_module.q_proj, query_alpha, current_rank, dtype, lora_parameters
            )
            attn_module.k_proj = create_patched_linear_lora(
                attn_module.k_proj, key_alpha, current_rank, dtype, lora_parameters
            )
            attn_module.v_proj = create_patched_linear_lora(
                attn_module.v_proj, value_alpha, current_rank, dtype, lora_parameters
            )
            attn_module.out_proj = create_patched_linear_lora(
                attn_module.out_proj, out_alpha, current_rank, dtype, lora_parameters
            )

        if patch_mlp:
            for name, mlp_module in text_encoder_mlp_modules(text_encoder):
                fc1_alpha = network_alphas.pop(name + ".fc1.lora_linear_layer.down.weight.alpha", None)
                fc2_alpha = network_alphas.pop(name + ".fc2.lora_linear_layer.down.weight.alpha", None)

                current_rank_fc1 = rank.pop(f"{name}.fc1.lora_linear_layer.up.weight")
                current_rank_fc2 = rank.pop(f"{name}.fc2.lora_linear_layer.up.weight")

                mlp_module.fc1 = create_patched_linear_lora(
                    mlp_module.fc1, fc1_alpha, current_rank_fc1, dtype, lora_parameters
                )
                mlp_module.fc2 = create_patched_linear_lora(
                    mlp_module.fc2, fc2_alpha, current_rank_fc2, dtype, lora_parameters
                )

        if is_network_alphas_populated and len(network_alphas) > 0:
            raise ValueError(
                f"The `network_alphas` has to be empty at this point but has the following keys \n\n {', '.join(network_alphas.keys())}"
            )

        return lora_parameters

    @classmethod
    def save_lora_weights(
        cls,
        save_directory: Union[str, os.PathLike],
        unet_lora_layers: Dict[str, Union[nn.Layer, paddle.Tensor]] = None,
        text_encoder_lora_layers: Dict[str, nn.Layer] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        to_diffusers=None,
    ):
        r"""
        Save the LoRA parameters corresponding to the UNet and text encoder.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save LoRA parameters to. Will be created if it doesn't exist.
            unet_lora_layers (`Dict[str, nn.Layer]` or `Dict[str, paddle.Tensor]`):
                State dict of the LoRA layers corresponding to the `unet`.
            text_encoder_lora_layers (`Dict[str, nn.Layer]` or `Dict[str, paddle.Tensor]`):
                State dict of the LoRA layers corresponding to the `text_encoder`. Must explicitly pass the text
                encoder LoRA state dict because it comes from 🤗 Transformers.
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
        """
        if to_diffusers is None:
            to_diffusers = TO_DIFFUSERS
        # Create a flat dictionary.
        state_dict = {}

        # Populate the dictionary.
        if unet_lora_layers is not None:
            weights = unet_lora_layers.state_dict() if isinstance(unet_lora_layers, nn.Layer) else unet_lora_layers
            if to_diffusers and isinstance(unet_lora_layers, nn.Layer):
                convert_paddle_state_dict_to_pytorch(unet_lora_layers, weights)

            unet_lora_state_dict = {f"{cls.unet_name}.{module_name}": param for module_name, param in weights.items()}

            state_dict.update(unet_lora_state_dict)

        if text_encoder_lora_layers is not None:
            weights = (
                text_encoder_lora_layers.state_dict()
                if isinstance(text_encoder_lora_layers, nn.Layer)
                else text_encoder_lora_layers
            )
            if to_diffusers and isinstance(text_encoder_lora_layers, nn.Layer):
                convert_paddle_state_dict_to_pytorch(text_encoder_lora_layers, weights)

            text_encoder_lora_state_dict = {
                f"{cls.text_encoder_name}.{module_name}": param for module_name, param in weights.items()
            }
            state_dict.update(text_encoder_lora_state_dict)

        # Save the model
        cls.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
            to_diffusers=to_diffusers,  # only change save weights name and save function
        )

    @staticmethod
    def write_lora_layers(
        state_dict: Dict[str, paddle.Tensor],
        save_directory: str,
        is_main_process: bool,
        weight_name: str,
        save_function: Callable,
        safe_serialization: bool,
        to_diffusers=None,
    ):
        if to_diffusers is None:
            to_diffusers = TO_DIFFUSERS
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        if weight_name is None:
            if to_diffusers:
                if safe_serialization:
                    weight_name = TORCH_LORA_WEIGHT_NAME_SAFE
                else:
                    weight_name = TORCH_LORA_WEIGHT_NAME
            else:
                if safe_serialization:
                    weight_name = PADDLE_LORA_WEIGHT_NAME_SAFE
                else:
                    weight_name = PADDLE_LORA_WEIGHT_NAME
        else:
            if "paddle" in weight_name.lower() or "pdparams" in weight_name.lower():
                to_diffusers = False
            elif "torch" in weight_name.lower() or "bin" in weight_name.lower():
                to_diffusers = True

        # choose save_function
        if save_function is None:
            if to_diffusers:
                if not is_torch_available() and not safe_serialization:
                    safe_serialization = True
                    logger.warning(
                        "PyTorch is not installed, and `safe_serialization` is currently set to `False`. "
                        "To ensure proper model saving, we will automatically set `safe_serialization=True`. "
                        "If you want to keep `safe_serialization=False`, please make sure PyTorch is installed."
                    )
                if safe_serialization:
                    if is_torch_available():
                        save_function = partial(torch_safe_save_file, metadata={"format": "pt"})
                    else:
                        save_function = partial(np_safe_save_file, metadata={"format": "pt"})
                else:
                    save_function = torch.save
            else:
                if safe_serialization:
                    state_dict = {k: np.ascontiguousarray(v) for k, v in state_dict.items()}
                    save_function = partial(np_safe_save_file, metadata={"format": "pd"})
                else:
                    save_function = paddle.save

        # we have transpose state_dict!

        save_function(state_dict, os.path.join(save_directory, weight_name))
        logger.info(f"Model weights saved in {os.path.join(save_directory, weight_name)}")

    def unload_lora_weights(self):
        """
        Unloads the LoRA parameters.

        Examples:

        ```python
        >>> # Assuming `pipeline` is already loaded with the LoRA parameters.
        >>> pipeline.unload_lora_weights()
        >>> ...
        ```
        """
        if not USE_PEFT_BACKEND:
            if version.parse(__version__) > version.parse("0.23"):
                logger.warn(
                    "You are using `unload_lora_weights` to disable and unload lora weights. If you want to iteratively enable and disable adapter weights,"
                    "you can use `pipe.enable_lora()` or `pipe.disable_lora()`. After installing the latest version of PEFT."
                )

            for _, module in self.unet.named_sublayers(include_self=True):
                if hasattr(module, "set_lora_layer"):
                    module.set_lora_layer(None)
        else:
            recurse_remove_peft_layers(self.unet)
            if hasattr(self.unet, "peft_config"):
                del self.unet.peft_config

        # Safe to call the following regardless of LoRA.
        self._remove_text_encoder_monkey_patch()

    def fuse_lora(
        self,
        fuse_unet: bool = True,
        fuse_text_encoder: bool = True,
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
    ):
        r"""
        Fuses the LoRA parameters into the original parameters of the corresponding blocks.

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            fuse_unet (`bool`, defaults to `True`): Whether to fuse the UNet LoRA parameters.
            fuse_text_encoder (`bool`, defaults to `True`):
                Whether to fuse the text encoder LoRA parameters. If the text encoder wasn't monkey-patched with the
                LoRA parameters then it won't have any effect.
            lora_scale (`float`, defaults to 1.0):
                Controls how much to influence the outputs with the LoRA parameters.
            safe_fusing (`bool`, defaults to `False`):
                Whether to check fused weights for NaN values before fusing and if values are NaN not fusing them.
        """
        if fuse_unet or fuse_text_encoder:
            self.num_fused_loras += 1
            if self.num_fused_loras > 1:
                logger.warn(
                    "The current API is supported for operating with a single LoRA file. You are trying to load and fuse more than one LoRA which is not well-supported.",
                )

        if fuse_unet:
            self.unet.fuse_lora(lora_scale, safe_fusing=safe_fusing)

        if USE_PEFT_BACKEND:
            from ppdiffusers.peft.tuners.tuners_utils import BaseTunerLayer

            def fuse_text_encoder_lora(text_encoder, lora_scale=1.0, safe_fusing=False):
                # TODO(Patrick, Younes): enable "safe" fusing
                for module in text_encoder.sublayers(include_self=True):
                    if isinstance(module, BaseTunerLayer):
                        if lora_scale != 1.0:
                            module.scale_layer(lora_scale)

                        module.merge()

        else:
            if version.parse(__version__) > version.parse("0.23"):
                deprecate("fuse_text_encoder_lora", "1.0", LORA_DEPRECATION_MESSAGE)

            def fuse_text_encoder_lora(text_encoder, lora_scale=1.0, safe_fusing=False):
                for _, attn_module in text_encoder_attn_modules(text_encoder):
                    if isinstance(attn_module.q_proj, PatchedLoraProjection):
                        attn_module.q_proj._fuse_lora(lora_scale, safe_fusing)
                        attn_module.k_proj._fuse_lora(lora_scale, safe_fusing)
                        attn_module.v_proj._fuse_lora(lora_scale, safe_fusing)
                        attn_module.out_proj._fuse_lora(lora_scale, safe_fusing)

                for _, mlp_module in text_encoder_mlp_modules(text_encoder):
                    if isinstance(mlp_module.fc1, PatchedLoraProjection):
                        mlp_module.fc1._fuse_lora(lora_scale, safe_fusing)
                        mlp_module.fc2._fuse_lora(lora_scale, safe_fusing)

        if fuse_text_encoder:
            if hasattr(self, "text_encoder"):
                fuse_text_encoder_lora(self.text_encoder, lora_scale, safe_fusing)
            if hasattr(self, "text_encoder_2"):
                fuse_text_encoder_lora(self.text_encoder_2, lora_scale, safe_fusing)

    def unfuse_lora(self, unfuse_unet: bool = True, unfuse_text_encoder: bool = True):
        r"""
        Reverses the effect of
        [`pipe.fuse_lora()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.fuse_lora).

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            unfuse_unet (`bool`, defaults to `True`): Whether to unfuse the UNet LoRA parameters.
            unfuse_text_encoder (`bool`, defaults to `True`):
                Whether to unfuse the text encoder LoRA parameters. If the text encoder wasn't monkey-patched with the
                LoRA parameters then it won't have any effect.
        """
        if unfuse_unet:
            if not USE_PEFT_BACKEND:
                self.unet.unfuse_lora()
            else:
                from ppdiffusers.peft.tuners.tuners_utils import BaseTunerLayer

                for module in self.unet.sublayers(include_self=True):
                    if isinstance(module, BaseTunerLayer):
                        module.unmerge()

        if USE_PEFT_BACKEND:
            from ppdiffusers.peft.tuners.tuners_utils import BaseTunerLayer

            def unfuse_text_encoder_lora(text_encoder):
                for module in text_encoder.sublayers(include_self=True):
                    if isinstance(module, BaseTunerLayer):
                        module.unmerge()

        else:
            if version.parse(__version__) > version.parse("0.23"):
                deprecate("unfuse_text_encoder_lora", "1.0", LORA_DEPRECATION_MESSAGE)

            def unfuse_text_encoder_lora(text_encoder):
                for _, attn_module in text_encoder_attn_modules(text_encoder):
                    if isinstance(attn_module.q_proj, PatchedLoraProjection):
                        attn_module.q_proj._unfuse_lora()
                        attn_module.k_proj._unfuse_lora()
                        attn_module.v_proj._unfuse_lora()
                        attn_module.out_proj._unfuse_lora()

                for _, mlp_module in text_encoder_mlp_modules(text_encoder):
                    if isinstance(mlp_module.fc1, PatchedLoraProjection):
                        mlp_module.fc1._unfuse_lora()
                        mlp_module.fc2._unfuse_lora()

        if unfuse_text_encoder:
            if hasattr(self, "text_encoder"):
                unfuse_text_encoder_lora(self.text_encoder)
            if hasattr(self, "text_encoder_2"):
                unfuse_text_encoder_lora(self.text_encoder_2)

        self.num_fused_loras -= 1

    def set_adapters_for_text_encoder(
        self,
        adapter_names: Union[List[str], str],
        text_encoder: Optional["PretrainedModel"] = None,  # noqa: F821
        text_encoder_weights: List[float] = None,
    ):
        """
        Sets the adapter layers for the text encoder.

        Args:
            adapter_names (`List[str]` or `str`):
                The names of the adapters to use.
            text_encoder (`nn.Layer`, *optional*):
                The text encoder module to set the adapter layers for. If `None`, it will try to get the `text_encoder`
                attribute.
            text_encoder_weights (`List[float]`, *optional*):
                The weights to use for the text encoder. If `None`, the weights are set to `1.0` for all the adapters.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        def process_weights(adapter_names, weights):
            if weights is None:
                weights = [1.0] * len(adapter_names)
            elif isinstance(weights, float):
                weights = [weights]

            if len(adapter_names) != len(weights):
                raise ValueError(
                    f"Length of adapter names {len(adapter_names)} is not equal to the length of the weights {len(weights)}"
                )
            return weights

        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names
        text_encoder_weights = process_weights(adapter_names, text_encoder_weights)
        text_encoder = text_encoder or getattr(self, "text_encoder", None)
        if text_encoder is None:
            raise ValueError(
                "The pipeline does not have a default `pipe.text_encoder` class. Please make sure to pass a `text_encoder` instead."
            )
        set_weights_and_activate_adapters(text_encoder, adapter_names, text_encoder_weights)

    def disable_lora_for_text_encoder(self, text_encoder: Optional["PretrainedModel"] = None):
        """
        Disables the LoRA layers for the text encoder.

        Args:
            text_encoder (`nn.Layer`, *optional*):
                The text encoder module to disable the LoRA layers for. If `None`, it will try to get the
                `text_encoder` attribute.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        text_encoder = text_encoder or getattr(self, "text_encoder", None)
        if text_encoder is None:
            raise ValueError("Text Encoder not found.")
        set_adapter_layers(text_encoder, enabled=False)

    def enable_lora_for_text_encoder(self, text_encoder: Optional["PretrainedModel"] = None):
        """
        Enables the LoRA layers for the text encoder.

        Args:
            text_encoder (`nn.Layer`, *optional*):
                The text encoder module to enable the LoRA layers for. If `None`, it will try to get the `text_encoder`
                attribute.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        text_encoder = text_encoder or getattr(self, "text_encoder", None)
        if text_encoder is None:
            raise ValueError("Text Encoder not found.")
        set_adapter_layers(self.text_encoder, enabled=True)

    def set_adapters(
        self,
        adapter_names: Union[List[str], str],
        adapter_weights: Optional[List[float]] = None,
    ):
        # Handle the UNET
        self.unet.set_adapters(adapter_names, adapter_weights)

        # Handle the Text Encoder
        if hasattr(self, "text_encoder"):
            self.set_adapters_for_text_encoder(adapter_names, self.text_encoder, adapter_weights)
        if hasattr(self, "text_encoder_2"):
            self.set_adapters_for_text_encoder(adapter_names, self.text_encoder_2, adapter_weights)

    def disable_lora(self):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        # Disable unet adapters
        self.unet.disable_lora()

        # Disable text encoder adapters
        if hasattr(self, "text_encoder"):
            self.disable_lora_for_text_encoder(self.text_encoder)
        if hasattr(self, "text_encoder_2"):
            self.disable_lora_for_text_encoder(self.text_encoder_2)

    def enable_lora(self):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        # Enable unet adapters
        self.unet.enable_lora()

        # Enable text encoder adapters
        if hasattr(self, "text_encoder"):
            self.enable_lora_for_text_encoder(self.text_encoder)
        if hasattr(self, "text_encoder_2"):
            self.enable_lora_for_text_encoder(self.text_encoder_2)

    def delete_adapters(self, adapter_names: Union[List[str], str]):
        """
        Args:
        Deletes the LoRA layers of `adapter_name` for the unet and text-encoder(s).
            adapter_names (`Union[List[str], str]`):
                The names of the adapter to delete. Can be a single string or a list of strings
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Delete unet adapters
        self.unet.delete_adapters(adapter_names)

        for adapter_name in adapter_names:
            # Delete text encoder adapters
            if hasattr(self, "text_encoder"):
                delete_adapter_layers(self.text_encoder, adapter_name)
            if hasattr(self, "text_encoder_2"):
                delete_adapter_layers(self.text_encoder_2, adapter_name)

    def get_active_adapters(self) -> List[str]:
        """
        Gets the list of the current active adapters.

        Example:

        ```python
        from ppdiffusers import DiffusionPipeline

        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
        )
        pipeline.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy")
        pipeline.get_active_adapters()
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError(
                "PEFT backend is required for this method. Please install the latest version of PEFT `pip install -U peft`"
            )

        from ppdiffusers.peft.tuners.tuners_utils import BaseTunerLayer

        active_adapters = []

        for module in self.unet.sublayers(include_self=True):
            if isinstance(module, BaseTunerLayer):
                active_adapters = module.active_adapters
                break

        return active_adapters

    def get_list_adapters(self) -> Dict[str, List[str]]:
        """
        Gets the current list of all available adapters in the pipeline.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError(
                "PEFT backend is required for this method. Please install the latest version of PEFT `pip install -U peft`"
            )

        set_adapters = {}

        if hasattr(self, "text_encoder") and hasattr(self.text_encoder, "peft_config"):
            set_adapters["text_encoder"] = list(self.text_encoder.peft_config.keys())

        if hasattr(self, "text_encoder_2") and hasattr(self.text_encoder_2, "peft_config"):
            set_adapters["text_encoder_2"] = list(self.text_encoder_2.peft_config.keys())

        if hasattr(self, "unet") and hasattr(self.unet, "peft_config"):
            set_adapters["unet"] = list(self.unet.peft_config.keys())

        return set_adapters

    def set_lora_device(self, adapter_names: List[str], device: Union[str, int]) -> None:
        """
        Moves the LoRAs listed in `adapter_names` to a target device. Useful for offloading the LoRA to the CPU in case
        you want to load multiple adapters and free some GPU memory.

        Args:
            adapter_names (`List[str]`):
                List of adapters to send device to.
            device (`Union[str, int]`):
                Device to send the adapters to. Can be either a torch device, a str or an integer.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        # from ppdiffusers.peft.tuners.tuners_utils import BaseTunerLayer

        # Handle the UNET
        # for unet_module in self.unet.sublayers(include_self=True):
        #     if isinstance(unet_module, BaseTunerLayer):
        #         for adapter_name in adapter_names:
        #             unet_module.lora_A[adapter_name].to(device)
        #             unet_module.lora_B[adapter_name].to(device)

        # Handle the text encoder
        # modules_to_process = []
        # if hasattr(self, "text_encoder"):
        #     modules_to_process.append(self.text_encoder)

        # if hasattr(self, "text_encoder_2"):
        #     modules_to_process.append(self.text_encoder_2)

        # for text_encoder in modules_to_process:
        #     # loop over submodules
        #     for text_encoder_module in text_encoder.sublayers(include_self=True):
        #         if isinstance(text_encoder_module, BaseTunerLayer):
        #             for adapter_name in adapter_names:
        #                 text_encoder_module.lora_A[adapter_name].to(device)
        #                 text_encoder_module.lora_B[adapter_name].to(device)


class StableDiffusionXLLoraLoaderMixin(LoraLoaderMixin):
    """This class overrides `LoraLoaderMixin` with LoRA loading/saving code that's specific to SDXL"""

    # Override to properly handle the loading and unloading of the additional text encoder.
    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, paddle.Tensor]],
        adapter_name: Optional[str] = None,
        **kwargs,
    ):
        """
        Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.unet` and
        `self.text_encoder`.

        All kwargs are forwarded to `self.lora_state_dict`.

        See [`~loaders.LoraLoaderMixin.lora_state_dict`] for more details on how the state dict is loaded.

        See [`~loaders.LoraLoaderMixin.load_lora_into_unet`] for more details on how the state dict is loaded into
        `self.unet`.

        See [`~loaders.LoraLoaderMixin.load_lora_into_text_encoder`] for more details on how the state dict is loaded
        into `self.text_encoder`.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                See [`~loaders.LoraLoaderMixin.lora_state_dict`].
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            kwargs (`dict`, *optional*):
                See [`~loaders.LoraLoaderMixin.lora_state_dict`].
        """
        # We could have accessed the unet config from `lora_state_dict()` too. We pass
        # it here explicitly to be able to tell that it's coming from an SDXL
        # pipeline.

        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        state_dict, network_alphas, from_diffusers = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict,
            unet_config=self.unet.config,
            **kwargs,
        )
        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        self.load_lora_into_unet(
            state_dict,
            network_alphas=network_alphas,
            unet=self.unet,
            adapter_name=adapter_name,
            _pipeline=self,
            from_diffusers=from_diffusers,
        )
        text_encoder_state_dict = {k: v for k, v in state_dict.items() if "text_encoder." in k}
        if len(text_encoder_state_dict) > 0:
            self.load_lora_into_text_encoder(
                text_encoder_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder,
                prefix="text_encoder",
                lora_scale=self.lora_scale,
                adapter_name=adapter_name,
                _pipeline=self,
                from_diffusers=from_diffusers,
            )

        text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if "text_encoder_2." in k}
        if len(text_encoder_2_state_dict) > 0:
            self.load_lora_into_text_encoder(
                text_encoder_2_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder_2,
                prefix="text_encoder_2",
                lora_scale=self.lora_scale,
                adapter_name=adapter_name,
                _pipeline=self,
                from_diffusers=from_diffusers,
            )

    @classmethod
    def save_lora_weights(
        cls,
        save_directory: Union[str, os.PathLike],
        unet_lora_layers: Dict[str, Union[nn.Layer, paddle.Tensor]] = None,
        text_encoder_lora_layers: Dict[str, Union[nn.Layer, paddle.Tensor]] = None,
        text_encoder_2_lora_layers: Dict[str, Union[nn.Layer, paddle.Tensor]] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        to_diffusers: bool = None,
    ):
        r"""
        Save the LoRA parameters corresponding to the UNet and text encoder.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save LoRA parameters to. Will be created if it doesn't exist.
            unet_lora_layers (`Dict[str, nn.Layer]` or `Dict[str, paddle.Tensor]`):
                State dict of the LoRA layers corresponding to the `unet`.
            text_encoder_lora_layers (`Dict[str, nn.Layer]` or `Dict[str, paddle.Tensor]`):
                State dict of the LoRA layers corresponding to the `text_encoder`. Must explicitly pass the text
                encoder LoRA state dict because it comes from 🤗 Transformers.
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
        """
        if to_diffusers is None:
            to_diffusers = TO_DIFFUSERS

        state_dict = {}

        def pack_weights(layers, prefix):
            layers_weights = layers.state_dict() if isinstance(layers, nn.Layer) else layers
            if to_diffusers and isinstance(layers, nn.Layer):
                convert_paddle_state_dict_to_pytorch(layers, layers_weights)
            layers_state_dict = {f"{prefix}.{module_name}": param for module_name, param in layers_weights.items()}
            return layers_state_dict

        if not (unet_lora_layers or text_encoder_lora_layers or text_encoder_2_lora_layers):
            raise ValueError(
                "You must pass at least one of `unet_lora_layers`, `text_encoder_lora_layers` or `text_encoder_2_lora_layers`."
            )

        if unet_lora_layers:
            state_dict.update(pack_weights(unet_lora_layers, "unet"))

        if text_encoder_lora_layers and text_encoder_2_lora_layers:
            state_dict.update(pack_weights(text_encoder_lora_layers, "text_encoder"))
            state_dict.update(pack_weights(text_encoder_2_lora_layers, "text_encoder_2"))

        cls.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
            to_diffusers=to_diffusers,
        )

    def _remove_text_encoder_monkey_patch(self):
        if USE_PEFT_BACKEND:
            recurse_remove_peft_layers(self.text_encoder)
            # TODO: @younesbelkada handle this in transformers side
            if getattr(self.text_encoder, "peft_config", None) is not None:
                del self.text_encoder.peft_config
                self.text_encoder._pp_peft_config_loaded = None

            recurse_remove_peft_layers(self.text_encoder_2)
            if getattr(self.text_encoder_2, "peft_config", None) is not None:
                del self.text_encoder_2.peft_config
                self.text_encoder_2._pp_peft_config_loaded = None
        else:
            self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder)
            self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder_2)


class SD3LoraLoaderMixin:
    r"""
    Load LoRA layers into [`SD3Transformer2DModel`].
    """

    transformer_name = TRANSFORMER_NAME
    num_fused_loras = 0

    def load_lora_weights(
        self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, paddle.Tensor]], adapter_name=None, **kwargs
    ):
        """
        Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.unet` and
        `self.text_encoder`.
        All kwargs are forwarded to `self.lora_state_dict`.
        See [`~loaders.LoraLoaderMixin.lora_state_dict`] for more details on how the state dict is loaded.
        See [`~loaders.LoraLoaderMixin.load_lora_into_transformer`] for more details on how the state dict is loaded
        into `self.transformer`.
        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                See [`~loaders.LoraLoaderMixin.lora_state_dict`].
            kwargs (`dict`, *optional*):
                See [`~loaders.LoraLoaderMixin.lora_state_dict`].
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        # if a dict is passed, copy it instead of modifying it inplace
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        state_dict, from_diffusers = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)

        is_correct_format = all("lora" in key or "dora_scale" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        self.load_lora_into_transformer(
            state_dict,
            transformer=getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer,
            adapter_name=adapter_name,
            _pipeline=self,
            from_diffusers=from_diffusers,
        )

    @classmethod
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, paddle.Tensor]],
        **kwargs,
    ):
        r"""
        Return state dict for lora weights and the network alphas.
        <Tip warning={true}>
        We support loading A1111 formatted LoRA checkpoints in a limited capacity.
        This function is experimental and might change in the future.
        </Tip>
        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:
                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
        """
        # Load the main state dict first which has the LoRA layers for either of
        # UNet and text encoder or both.
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
        from_diffusers = kwargs.pop("from_diffusers", FROM_DIFFUSERS)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        if use_safetensors is None:
            use_safetensors = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch" if from_diffusers else "paddle",
        }

        model_file = None
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            # Let's first try to load .safetensors weights
            if (use_safetensors and weight_name is None) or (
                weight_name is not None and weight_name.endswith(".safetensors")
            ):
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=(weight_name or TORCH_LORA_WEIGHT_NAME_SAFE)
                        if from_diffusers
                        else ((weight_name or PADDLE_LORA_WEIGHT_NAME_SAFE)),
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        from_aistudio=from_aistudio,
                        from_hf_hub=from_hf_hub,
                    )
                except Exception:
                    model_file = None

            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=(weight_name or TORCH_LORA_WEIGHT_NAME)
                    if from_diffusers
                    else ((weight_name or PADDLE_LORA_WEIGHT_NAME)),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    from_aistudio=from_aistudio,
                    from_hf_hub=from_hf_hub,
                )

            assert model_file is not None, "Could not find the model file!"
            # TODO, check this
            from ppdiffusers.utils import smart_load

            state_dict = smart_load(model_file, return_is_torch_weight=True)
            is_torch_weight = state_dict.pop("is_torch_weight", False)
            if not from_diffusers and is_torch_weight:
                logger.warning(
                    "Detect the weight is in diffusers format, but currently, `from_diffusers` is set to `False`. To proceed, we will change the value of `from_diffusers` to `True`!"
                )
                from_diffusers = True
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        return state_dict, from_diffusers

    @classmethod
    def load_lora_into_transformer(
        cls,
        state_dict,
        transformer,
        adapter_name=None,
        _pipeline=None,
        from_diffusers=None,
    ):
        """
        This will load the LoRA layers specified in `state_dict` into `transformer`.
        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The keys can either be indexed directly
                into the unet or prefixed with an additional `unet` which can be used to distinguish between text
                encoder lora layers.
            transformer (`SD3Transformer2DModel`):
                The Transformer model to load the LoRA layers into.
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
        """
        if from_diffusers is None:
            from_diffusers = FROM_DIFFUSERS

        from ppdiffusers.peft import (
            LoraConfig,
            inject_adapter_in_model,
            set_peft_model_state_dict,
        )

        keys = list(state_dict.keys())

        transformer_keys = [k for k in keys if k.startswith(cls.transformer_name)]
        state_dict = {
            k.replace(f"{cls.transformer_name}.", ""): v for k, v in state_dict.items() if k in transformer_keys
        }

        if len(state_dict.keys()) > 0:
            if adapter_name in getattr(transformer, "peft_config", {}):
                raise ValueError(
                    f"Adapter name {adapter_name} already in use in the transformer - please select a new adapter name."
                )

            rank = {}
            index = 1 if from_diffusers else 0
            for key, val in state_dict.items():
                if "lora_B" in key:
                    # linear
                    if len(val.shape) == 2:
                        rank[key] = val.shape[index]
                    # conv2d
                    elif len(val.shape) == 4:
                        rank[key] = val.shape[1]
                    else:
                        raise ValueError(
                            f"LoRA B matrix shape should be 2D or 4D, got {len(val.shape)}D instead."
                        )

            lora_config_kwargs = get_peft_kwargs(rank, network_alpha_dict=None, peft_state_dict=state_dict)
            if "use_dora" in lora_config_kwargs:
                raise ValueError("ppdiffusers.peft does not support dora yet")
            lora_config = LoraConfig(**lora_config_kwargs)

            # adapter_name
            if adapter_name is None:
                adapter_name = get_adapter_name(transformer)

            inject_adapter_in_model(lora_config, transformer, adapter_name=adapter_name)
            incompatible_keys = set_peft_model_state_dict(transformer, state_dict, adapter_name)

            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

            # Unsafe code />

    @classmethod
    def save_lora_weights(
        cls,
        save_directory: Union[str, os.PathLike],
        transformer_lora_layers: Dict[str, paddle.nn.Layer] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        to_diffusers=None,
    ):
        r"""
        Save the LoRA parameters corresponding to the UNet and text encoder.
        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save LoRA parameters to. Will be created if it doesn't exist.
            transformer_lora_layers (`Dict[str, paddle.nn.Layer]` or `Dict[str, paddletorch.Tensor]`):
                State dict of the LoRA layers corresponding to the `transformer`.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` or `paddle.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional way.
        """
        if to_diffusers is None:
            to_diffusers = TO_DIFFUSERS

        state_dict = {}

        def pack_weights(layers, prefix):
            layers_weights = layers.state_dict() if isinstance(layers, nn.Layer) else layers
            layers_state_dict = {f"{prefix}.{module_name}": param for module_name, param in layers_weights.items()}
            return layers_state_dict

        if not transformer_lora_layers:
            raise ValueError("You must pass `transformer_lora_layers`.")

        if transformer_lora_layers:
            state_dict.update(pack_weights(transformer_lora_layers, cls.transformer_name))

        # Save the model
        cls.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
            to_diffusers=to_diffusers,  # only change save weights name and save function
        )

    @staticmethod
    def write_lora_layers(
        state_dict: Dict[str, paddle.Tensor],
        save_directory: str,
        is_main_process: bool,
        weight_name: str,
        save_function: Callable,
        safe_serialization: bool,
        to_diffusers=None,
    ):
        if to_diffusers is None:
            to_diffusers = TO_DIFFUSERS

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            if to_diffusers:
                if not is_torch_available() and not safe_serialization:
                    safe_serialization = True
                    logger.warning(
                        "PyTorch is not installed, and `safe_serialization` is currently set to `False`. "
                        "To ensure proper model saving, we will automatically set `safe_serialization=True`. "
                        "If you want to keep `safe_serialization=False`, please make sure PyTorch is installed."
                    )
                if safe_serialization:
                    if is_torch_available():
                        save_function = partial(torch_safe_save_file, metadata={"format": "pt"})
                    else:
                        save_function = partial(np_safe_save_file, metadata={"format": "pt"})
                else:
                    save_function = torch.save
            else:
                if safe_serialization:
                    state_dict = {k: np.ascontiguousarray(v) for k, v in state_dict.items()}
                    save_function = partial(np_safe_save_file, metadata={"format": "pd"})
                else:
                    save_function = paddle.save

        os.makedirs(save_directory, exist_ok=True)

        if weight_name is None:
            if to_diffusers:
                if safe_serialization:
                    weight_name = TORCH_LORA_WEIGHT_NAME_SAFE
                else:
                    weight_name = TORCH_LORA_WEIGHT_NAME
            else:
                if safe_serialization:
                    weight_name = PADDLE_LORA_WEIGHT_NAME_SAFE
                else:
                    weight_name = PADDLE_LORA_WEIGHT_NAME
        else:
            if "paddle" in weight_name.lower() or "pdparams" in weight_name.lower():
                to_diffusers = False
            elif "torch" in weight_name.lower() or "bin" in weight_name.lower():
                to_diffusers = True

        save_path = Path(save_directory, weight_name).as_posix()
        save_function(state_dict, save_path)
        logger.info(f"Model weights saved in {save_path}")

    def unload_lora_weights(self):
        """
        Unloads the LoRA parameters.
        Examples:
        ```python
        >>> # Assuming `pipeline` is already loaded with the LoRA parameters.
        >>> pipeline.unload_lora_weights()
        >>> ...
        ```
        """
        transformer = getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer
        recurse_remove_peft_layers(transformer)
        if hasattr(transformer, "peft_config"):
            del transformer.peft_config

    @classmethod
    def _optionally_disable_offloading(cls, _pipeline):
        """
        Optionally removes offloading in case the pipeline has been already sequentially offloaded to CPU.
        Args:
            _pipeline (`DiffusionPipeline`):
                The pipeline to disable offloading for.
        Returns:
            tuple:
                A tuple indicating if `is_model_cpu_offload` or `is_sequential_cpu_offload` is True.
        """
        pass
