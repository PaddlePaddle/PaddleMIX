# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from typing import Callable, Dict, List, Optional, Union

import paddle

from ppdiffusers.loaders import LoraLoaderMixin

from ..models.modeling_pytorch_paddle_utils import convert_paddle_state_dict_to_pytorch
from ..utils import USE_PEFT_BACKEND, logging

logger = logging.get_logger(__name__)

TRANSFORMER_NAME = "transformer"


def _maybe_adjust_config(config):
    """
    We may run into some ambiguous configuration values when a model has module names, sharing a common prefix
    (`proj_out.weight` and `blocks.transformer.proj_out.weight`, for example) and they have different LoRA ranks. This
    method removes the ambiguity by following what is described here:
    https://github.com/huggingface/diffusers/pull/9985#issuecomment-2493840028.
    """
    rank_pattern = config["rank_pattern"].copy()
    target_modules = config["target_modules"]
    original_r = config["r"]

    for key in list(rank_pattern.keys()):
        key_rank = rank_pattern[key]

        # try to detect ambiguity
        # `target_modules` can also be a str, in which case this loop would loop
        # over the chars of the str. The technically correct way to match LoRA keys
        # in PEFT is to use LoraModel._check_target_module_exists (lora_config, key).
        # But this cuts it for now.
        exact_matches = [mod for mod in target_modules if mod == key]
        substring_matches = [mod for mod in target_modules if key in mod and mod != key]
        ambiguous_key = key

        if exact_matches and substring_matches:
            # if ambiguous we update the rank associated with the ambiguous key (`proj_out`, for example)
            config["r"] = key_rank
            # remove the ambiguous key from `rank_pattern` and update its rank to `r`, instead
            del config["rank_pattern"][key]
            for mod in substring_matches:
                # avoid overwriting if the module already has a specific rank
                if mod not in config["rank_pattern"]:
                    config["rank_pattern"][mod] = original_r

            # update the rest of the keys with the `original_r`
            for mod in target_modules:
                if mod != ambiguous_key and mod not in config["rank_pattern"]:
                    config["rank_pattern"][mod] = original_r

    # handle alphas to deal with cases like
    has_different_ranks = len(config["rank_pattern"]) > 1 and list(config["rank_pattern"])[0] != config["r"]
    if has_different_ranks:
        config["lora_alpha"] = config["r"]
        alpha_pattern = {}
        for module_name, rank in config["rank_pattern"].items():
            alpha_pattern[module_name] = rank
        config["alpha_pattern"] = alpha_pattern

    return config


class CogVideoXLoraLoaderMixin(LoraLoaderMixin):
    """
    Load LoRA layers into [`CogVideoXTransformer3DModel`]. Specific to [`CogVideoXPipeline`].
    """

    _lora_loadable_modules = ["transformer"]
    transformer_name = TRANSFORMER_NAME

    @classmethod
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, paddle.Tensor]],
        **kwargs,
    ):
        """
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

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.

        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True

        user_agent = {"file_type": "attn_procs_weights", "framework": "paddle"}
        state_dict = super().lora_state_dict(
            pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
            weight_name=weight_name,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            allow_pickle=allow_pickle,
        )
        is_dora_scale_present = any("dora_scale" in k for k in state_dict)
        if is_dora_scale_present:
            warn_msg = "It seems like you are using a DoRA checkpoint that is not compatible in Diffusers at the moment. So, we are going to filter out the keys associated to 'dora_scale` from the state dict. If you think this is a mistake please open an issue https://github.com/huggingface/diffusers/issues/new."
            logger.warning(warn_msg)
            state_dict = {k: v for k, v in state_dict.items() if "dora_scale" not in k}
        return state_dict

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, paddle.Tensor]],
        adapter_name=None,
        **kwargs,
    ):
        """
        Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.transformer` and
        `self.text_encoder`. All kwargs are forwarded to `self.lora_state_dict`. See
        [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`] for more details on how the state dict is loaded.
        See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_into_transformer`] for more details on how the state
        dict is loaded into `self.transformer`.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`].
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            low_cpu_mem_usage (`bool`, *optional*):
                Speed up model loading by only loading the pretrained LoRA weights and not initializing the random
                weights.
            kwargs (`dict`, *optional*):
                See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`].
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()
        state_dict = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)
        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")
        self.load_lora_into_transformer(
            state_dict,
            transformer=getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer,
            adapter_name=adapter_name,
            _pipeline=self,
        )

    @classmethod
    def load_lora_into_transformer(
        cls,
        state_dict,
        transformer,
        adapter_name=None,
        _pipeline=None,
        low_cpu_mem_usage=False,
    ):
        """
        This will load the LoRA layers specified in `state_dict` into `transformer`.

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The keys can either be indexed directly
                into the unet or prefixed with an additional `unet` which can be used to distinguish between text
                encoder lora layers.
            transformer (`CogVideoXTransformer3DModel`):
                The Transformer model to load the LoRA layers into.
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            low_cpu_mem_usage (`bool`, *optional*):
                Speed up model loading by only loading the pretrained LoRA weights and not initializing the random
                weights.
        """

        logger.info(f"Loading {cls.transformer_name}.")
        transformer.load_lora_adapter(
            state_dict,
            network_alphas=None,
            adapter_name=adapter_name,
            _pipeline=_pipeline,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

    @classmethod
    def save_lora_weights(
        cls,
        save_directory: Union[str, os.PathLike],
        transformer_lora_layers: Dict[str, Union[paddle.nn.Layer, paddle.Tensor]] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        to_diffusers: bool = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
    ):
        """
        Save the LoRA parameters corresponding to the UNet and text encoder.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save LoRA parameters to. Will be created if it doesn't exist.
            transformer_lora_layers (`Dict[str, paddle.nn.Layer]` or `Dict[str, paddle.Tensor]`):
                State dict of the LoRA layers corresponding to the `transformer`.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `paddle.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional Paddle way with `pickle`.
        """
        state_dict = {}
        if not transformer_lora_layers:
            raise ValueError("You must pass `transformer_lora_layers`.")

        def pack_weights(layers, prefix):
            layers_weights = layers.state_dict() if isinstance(layers, paddle.nn.Layer) else layers
            if to_diffusers and isinstance(layers, paddle.nn.Layer):
                convert_paddle_state_dict_to_pytorch(layers, layers_weights)
            layers_state_dict = {f"{prefix}.{module_name}": param for module_name, param in layers_weights.items()}
            return layers_state_dict

        if transformer_lora_layers:
            state_dict.update(pack_weights(transformer_lora_layers, cls.transformer_name))
        cls.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
        )

    def fuse_lora(
        self,
        components: List[str] = ["transformer"],
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
        adapter_names: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Fuses the LoRA parameters into the original parameters of the corresponding blocks.

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            components: (`List[str]`): List of LoRA-injectable components to fuse the LoRAs into.
            lora_scale (`float`, defaults to 1.0):
                Controls how much to influence the outputs with the LoRA parameters.
            safe_fusing (`bool`, defaults to `False`):
                Whether to check fused weights for NaN values before fusing and if values are NaN not fusing them.
            adapter_names (`List[str]`, *optional*):
                Adapter names to be used for fusing. If nothing is passed, all active adapters will be fused.

        """
        super().fuse_lora(
            components=components,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
            adapter_names=adapter_names,
        )

    def unfuse_lora(self, components: List[str] = ["transformer"], **kwargs):
        """
        Reverses the effect of
        [`pipe.fuse_lora()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraBaseMixin.fuse_lora).

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            components (`List[str]`): List of LoRA-injectable components to unfuse LoRA from.
            unfuse_transformer (`bool`, defaults to `True`): Whether to unfuse the UNet LoRA parameters.
        """
        super().unfuse_lora(components=components)
