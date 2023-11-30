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
import copy
import os
import re
import warnings
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import requests
from huggingface_hub import hf_hub_download
from huggingface_hub.file_download import _request_wrapper, hf_raise_for_status

from .models.modeling_utils import convert_state_dict
from .utils import (
    DIFFUSERS_CACHE,
    FROM_DIFFUSERS,
    FROM_HF_HUB,
    HF_HUB_OFFLINE,
    PPDIFFUSERS_CACHE,
    TO_DIFFUSERS,
    _get_model_file,
    deprecate,
    is_omegaconf_available,
    is_paddlenlp_available,
    is_safetensors_available,
    is_torch_available,
    is_torch_file,
    logging,
    ppdiffusers_url_download,
    safetensors_load,
    smart_load,
    torch_load,
)
from .utils.import_utils import BACKENDS_MAPPING

logger = logging.get_logger(__name__)

if is_torch_available():
    import torch
if is_safetensors_available():
    import safetensors

if is_paddlenlp_available():
    from paddlenlp.transformers import (
        CLIPTextModel,
        CLIPTextModelWithProjection,
        PretrainedModel,
        PretrainedTokenizer,
    )

TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"

TORCH_LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
TORCH_LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"
PADDLE_LORA_WEIGHT_NAME = "paddle_lora_weights.pdparams"

TORCH_TEXT_INVERSION_NAME = "learned_embeds.bin"
TORCH_TEXT_INVERSION_NAME_SAFE = "learned_embeds.safetensors"
PADDLE_TEXT_INVERSION_NAME = "learned_embeds.pdparams"

TORCH_CUSTOM_DIFFUSION_WEIGHT_NAME = "pytorch_custom_diffusion_weights.bin"
TORCH_CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE = "pytorch_custom_diffusion_weights.safetensors"
PADDLE_CUSTOM_DIFFUSION_WEIGHT_NAME = "paddle_custom_diffusion_weights.pdparams"


def transpose_state_dict(state_dict, name_mapping=None):
    new_state_dict = {}
    for k, v in state_dict.items():
        if name_mapping is not None:
            for old_name, new_name in name_mapping.items():
                k = k.replace(old_name, new_name)
        if v.ndim == 2:
            new_state_dict[k] = v.T.contiguous() if hasattr(v, "contiguous") else v.T
        else:
            new_state_dict[k] = v.contiguous() if hasattr(v, "contiguous") else v
    return new_state_dict


class PatchedLoraProjection(nn.Layer):
    def __init__(self, regular_linear_layer, lora_scale=1, network_alpha=None, rank=4, dtype=None):
        super().__init__()
        from .models.lora import LoRALinearLayer

        self.regular_linear_layer = regular_linear_layer

        if dtype is None:
            dtype = self.regular_linear_layer.weight.dtype

        in_features, out_features = self.regular_linear_layer.weight.shape

        self.lora_linear_layer = LoRALinearLayer(
            in_features,
            out_features,
            network_alpha=network_alpha,
            dtype=dtype,
            rank=rank,
        )

        self.lora_scale = lora_scale

    def forward(self, input):
        return self.regular_linear_layer(input) + self.lora_scale * self.lora_linear_layer(input)


def text_encoder_attn_modules(text_encoder):
    attn_modules = []

    if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
        for i, layer in enumerate(text_encoder.text_model.transformer.layers):
            name = f"text_model.transformer.layers.{i}.self_attn"
            mod = layer.self_attn
            attn_modules.append((name, mod))
    else:
        raise ValueError(f"do not know how to get attention modules for: {text_encoder.__class__.__name__}")

    return attn_modules


def text_encoder_mlp_modules(text_encoder):
    mlp_modules = []

    if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
        for i, layer in enumerate(text_encoder.text_model.transformer.layers):
            mlp_mod = layer
            name = f"text_model.transformer.layers.{i}"
            mlp_modules.append((name, mlp_mod))
    else:
        raise ValueError(f"do not know how to get mlp modules for: {text_encoder.__class__.__name__}")

    return mlp_modules


def text_encoder_lora_state_dict(text_encoder):
    state_dict = {}

    for name, module in text_encoder_attn_modules(text_encoder):
        for k, v in module.q_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.q_proj.lora_linear_layer.{k}"] = v

        for k, v in module.k_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.k_proj.lora_linear_layer.{k}"] = v

        for k, v in module.v_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.v_proj.lora_linear_layer.{k}"] = v

        for k, v in module.out_proj.lora_linear_layer.state_dict().items():
            state_dict[f"{name}.out_proj.lora_linear_layer.{k}"] = v

    return state_dict


class AttnProcsLayers(nn.Layer):
    def __init__(self, state_dict: Dict[str, paddle.Tensor]):
        super().__init__()
        self.layers = nn.LayerList(state_dict.values())
        self.mapping = dict(enumerate(state_dict.keys()))
        self.rev_mapping = {v: k for k, v in enumerate(state_dict.keys())}

        # .processor for unet, .self_attn for text encoder
        self.split_keys = [".processor", ".self_attn"]

        # we add a hook to state_dict() and load_state_dict() so that the
        # naming fits with `unet.attn_processors`
        def map_to(state_dict, *args, **kwargs):
            new_state_dict = {}
            for key, value in state_dict.items():
                num = int(key.split(".")[1])  # 0 is always "layers"
                new_key = key.replace(f"layers.{num}", self.mapping[num])
                new_state_dict[new_key] = value

            return new_state_dict

        def remap_key(key, state_dict):
            for k in self.split_keys:
                if k in key:
                    return key.split(k)[0] + k

            raise ValueError(
                f"There seems to be a problem with the state_dict: {set(state_dict.keys())}. {key} has to have one of {self.split_keys}."
            )

        def map_from(module, state_dict, *args, **kwargs):
            all_keys = list(state_dict.keys())
            for key in all_keys:
                replace_key = remap_key(key, state_dict)
                new_key = key.replace(replace_key, f"layers.{module.rev_mapping[replace_key]}")
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        self.register_state_dict_hook(map_to)
        self.register_load_state_dict_pre_hook(map_from, with_module=True)


class UNet2DConditionLoadersMixin:
    text_encoder_name = TEXT_ENCODER_NAME
    unet_name = UNET_NAME

    def load_attn_procs(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, paddle.Tensor]], **kwargs):
        r"""
        Load pretrained attention processor layers into [`UNet2DConditionModel`]. Attention processor layers have to be
        defined in
        [`cross_attention.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py)
        and be a `paddle.nn.Layer` class.
        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:
                    - A string, the model id (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a directory (for example `./my_model_directory`) containing the model weights saved
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
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
        """
        from .models.attention_processor import (
            AttnAddedKVProcessor,
            AttnAddedKVProcessor2_5,
            CustomDiffusionAttnProcessor,
            LoRAAttnAddedKVProcessor,
            LoRAAttnProcessor,
            LoRAAttnProcessor2_5,
            LoRAXFormersAttnProcessor,
            SlicedAttnAddedKVProcessor,
            XFormersAttnProcessor,
        )
        from .models.lora import (
            LoRACompatibleConv,
            LoRACompatibleLinear,
            LoRAConv2dLayer,
            LoRALinearLayer,
        )

        from_hf_hub = kwargs.pop("from_hf_hub", FROM_HF_HUB)
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        from_diffusers = kwargs.pop("from_diffusers", FROM_DIFFUSERS)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        network_alphas = kwargs.pop("network_alphas", None)

        if from_diffusers and use_safetensors and not is_safetensors_available():
            raise ValueError(
                "`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetensors"
            )
        if use_safetensors is None:
            use_safetensors = is_safetensors_available()

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch" if from_diffusers else "paddle",
        }

        model_file = None

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            if from_diffusers:
                # Let's first try to load .safetensors weights
                if (use_safetensors and weight_name is None) or (
                    weight_name is not None and weight_name.endswith(".safetensors")
                ):
                    try:
                        model_file = _get_model_file(
                            pretrained_model_name_or_path_or_dict,
                            weights_name=weight_name or TORCH_LORA_WEIGHT_NAME_SAFE,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            resume_download=resume_download,
                            proxies=proxies,
                            local_files_only=local_files_only,
                            use_auth_token=use_auth_token,
                            revision=revision,
                            subfolder=subfolder,
                            user_agent=user_agent,
                            from_hf_hub=from_hf_hub,
                        )
                        state_dict = smart_load(model_file)
                    except Exception:
                        model_file = None
                        pass
                if model_file is None:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weight_name or TORCH_LORA_WEIGHT_NAME,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        from_hf_hub=from_hf_hub,
                    )
                    state_dict = smart_load(model_file)
            else:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or PADDLE_LORA_WEIGHT_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    from_hf_hub=from_hf_hub,
                )
                state_dict = smart_load(model_file)
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        # fill attn processors
        attn_processors = {}
        non_attn_lora_layers = []

        is_lora = all(("lora" in k or k.endswith(".alpha")) for k in state_dict.keys())
        is_custom_diffusion = any("custom_diffusion" in k for k in state_dict.keys())

        if from_diffusers or is_torch_file(model_file):
            state_dict = transpose_state_dict(state_dict)

        if is_lora:
            is_new_lora_format = all(
                key.startswith(self.unet_name) or key.startswith(self.text_encoder_name) for key in state_dict.keys()
            )
            if is_new_lora_format:
                # Strip the `"unet"` prefix.
                is_text_encoder_present = any(key.startswith(self.text_encoder_name) for key in state_dict.keys())
                if is_text_encoder_present:
                    warn_message = "The state_dict contains LoRA params corresponding to the text encoder which are not being used here. To use both UNet and text encoder related LoRA params, use [`pipe.load_lora_weights()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.load_lora_weights)."
                    warnings.warn(warn_message)
                unet_keys = [k for k in state_dict.keys() if k.startswith(self.unet_name)]
                state_dict = {k.replace(f"{self.unet_name}.", ""): v for k, v in state_dict.items() if k in unet_keys}

            lora_grouped_dict = defaultdict(dict)
            mapped_network_alphas = {}

            all_keys = list(state_dict.keys())
            for key in all_keys:
                value = state_dict.pop(key)
                attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                lora_grouped_dict[attn_processor_key][sub_key] = value.cast(
                    dtype="float32"
                )  # we must cast this to float32

                # Create another `mapped_network_alphas` dictionary so that we can properly map them.
                if network_alphas is not None:
                    for k in network_alphas:
                        if k.replace(".alpha", "") in key:
                            mapped_network_alphas.update({attn_processor_key: network_alphas[k]})

            if len(state_dict) > 0:
                raise ValueError(
                    f"The state_dict has to be empty at this point but has the following keys \n\n {', '.join(state_dict.keys())}"
                )

            for key, value_dict in lora_grouped_dict.items():
                attn_processor = self
                for sub_key in key.split("."):
                    attn_processor = getattr(attn_processor, sub_key)

                # Process non-attention layers, which don't have to_{k,v,q,out_proj}_lora layers
                # or add_{k,v,q,out_proj}_proj_lora layers.
                if "lora.down.weight" in value_dict:
                    if value_dict["lora.down.weight"].ndim == 2:
                        rank = value_dict["lora.down.weight"].shape[1]
                    else:
                        rank = value_dict["lora.down.weight"].shape[0]

                    if isinstance(attn_processor, LoRACompatibleConv):
                        in_features = attn_processor._in_channels
                        out_features = attn_processor._out_channels
                        kernel_size = attn_processor._kernel_size

                        lora = LoRAConv2dLayer(
                            in_features=in_features,
                            out_features=out_features,
                            rank=rank,
                            kernel_size=kernel_size,
                            stride=attn_processor._stride,
                            padding=attn_processor._padding,
                            network_alpha=mapped_network_alphas.get(key),
                        )
                    elif isinstance(attn_processor, LoRACompatibleLinear):
                        lora = LoRALinearLayer(
                            attn_processor.weight.shape[0],
                            attn_processor.weight.shape[1],
                            rank,
                            mapped_network_alphas.get(key),
                        )
                    else:
                        raise ValueError(f"Module {key} is not a LoRACompatibleConv or LoRACompatibleLinear module.")

                    value_dict = {k.replace("lora.", ""): v for k, v in value_dict.items()}
                    lora.load_dict(value_dict)
                    non_attn_lora_layers.append((attn_processor, lora))
                else:
                    # To handle SDXL.
                    rank_mapping = {}
                    hidden_size_mapping = {}
                    for projection_id in ["to_k", "to_q", "to_v", "to_out"]:
                        rank = value_dict[f"{projection_id}_lora.down.weight"].shape[1]
                        hidden_size = value_dict[f"{projection_id}_lora.up.weight"].shape[1]

                        rank_mapping.update({f"{projection_id}_lora.down.weight": rank})
                        hidden_size_mapping.update({f"{projection_id}_lora.up.weight": hidden_size})

                    if isinstance(
                        attn_processor, (AttnAddedKVProcessor, SlicedAttnAddedKVProcessor, AttnAddedKVProcessor2_5)
                    ):
                        cross_attention_dim = value_dict["add_k_proj_lora.down.weight"].shape[0]
                        attn_processor_class = LoRAAttnAddedKVProcessor
                    else:
                        cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[0]
                        if isinstance(attn_processor, (XFormersAttnProcessor, LoRAXFormersAttnProcessor)):
                            attn_processor_class = LoRAXFormersAttnProcessor
                        else:
                            attn_processor_class = (
                                LoRAAttnProcessor2_5
                                if hasattr(F, "scaled_dot_product_attention_")
                                else LoRAAttnProcessor
                            )

                    if attn_processor_class is not LoRAAttnAddedKVProcessor:
                        attn_processors[key] = attn_processor_class(
                            rank=rank_mapping.get("to_k_lora.down.weight"),
                            hidden_size=hidden_size_mapping.get("to_k_lora.up.weight"),
                            cross_attention_dim=cross_attention_dim,
                            network_alpha=mapped_network_alphas.get(key),
                            q_rank=rank_mapping.get("to_q_lora.down.weight"),
                            q_hidden_size=hidden_size_mapping.get("to_q_lora.up.weight"),
                            v_rank=rank_mapping.get("to_v_lora.down.weight"),
                            v_hidden_size=hidden_size_mapping.get("to_v_lora.up.weight"),
                            out_rank=rank_mapping.get("to_out_lora.down.weight"),
                            out_hidden_size=hidden_size_mapping.get("to_out_lora.up.weight"),
                            # rank=rank_mapping.get("to_k_lora.down.weight", None),
                            # hidden_size=hidden_size_mapping.get("to_k_lora.up.weight", None),
                            # q_rank=rank_mapping.get("to_q_lora.down.weight", None),
                            # q_hidden_size=hidden_size_mapping.get("to_q_lora.up.weight", None),
                            # v_rank=rank_mapping.get("to_v_lora.down.weight", None),
                            # v_hidden_size=hidden_size_mapping.get("to_v_lora.up.weight", None),
                            # out_rank=rank_mapping.get("to_out_lora.down.weight", None),
                            # out_hidden_size=hidden_size_mapping.get("to_out_lora.up.weight", None),
                        )
                    else:
                        attn_processors[key] = attn_processor_class(
                            rank=rank_mapping.get("to_k_lora.down.weight", None),
                            hidden_size=hidden_size_mapping.get("to_k_lora.up.weight", None),
                            cross_attention_dim=cross_attention_dim,
                            network_alpha=mapped_network_alphas.get(key),
                        )

                    attn_processors[key].load_dict(value_dict)

        elif is_custom_diffusion:
            custom_diffusion_grouped_dict = defaultdict(dict)
            for key, value in state_dict.items():
                if len(value) == 0:
                    custom_diffusion_grouped_dict[key] = {}
                else:
                    if "to_out" in key:
                        attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                    else:
                        attn_processor_key, sub_key = ".".join(key.split(".")[:-2]), ".".join(key.split(".")[-2:])
                    custom_diffusion_grouped_dict[attn_processor_key][sub_key] = value.cast(
                        dtype="float32"
                    )  # we must cast this to float32

            for key, value_dict in custom_diffusion_grouped_dict.items():
                if len(value_dict) == 0:
                    attn_processors[key] = CustomDiffusionAttnProcessor(
                        train_kv=False, train_q_out=False, hidden_size=None, cross_attention_dim=None
                    )
                else:
                    cross_attention_dim = value_dict["to_k_custom_diffusion.weight"].shape[
                        0
                    ]  # 1 -> 0, torch vs paddle nn.Linear
                    hidden_size = value_dict["to_k_custom_diffusion.weight"].shape[
                        1
                    ]  # 0 -> 1, torch vs paddle nn.Linear
                    train_q_out = True if "to_q_custom_diffusion.weight" in value_dict else False
                    attn_processors[key] = CustomDiffusionAttnProcessor(
                        train_kv=True,
                        train_q_out=train_q_out,
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                    )
                    attn_processors[key].load_dict(value_dict)
        else:
            raise ValueError(
                f"{model_file} does not seem to be in the correct format expected by LoRA or Custom Diffusion training."
            )
        # set correct dtype & device
        attn_processors = {k: v.to(dtype=self.dtype) for k, v in attn_processors.items()}
        non_attn_lora_layers = [(t, l.to(dtype=self.dtype)) for t, l in non_attn_lora_layers]
        # set layers
        self.set_attn_processor(attn_processors)

        # set ff layers
        for target_module, lora_layer in non_attn_lora_layers:
            target_module.set_lora_layer(lora_layer)
            # It should raise an error if we don't have a set lora here
            # if hasattr(target_module, "set_lora_layer"):
            #     target_module.set_lora_layer(lora_layer)

    def save_attn_procs(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = False,
        to_diffusers: Optional[bool] = None,
    ):
        r"""
        Save an attention processor to a directory so that it can be reloaded using the
        [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`] method.
        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save an attention processor to. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `paddle.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
        """
        from .models.attention_processor import (
            CustomDiffusionAttnProcessor,
            CustomDiffusionXFormersAttnProcessor,
        )

        if to_diffusers is None:
            to_diffusers = TO_DIFFUSERS
        if to_diffusers and safe_serialization and not is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        is_custom_diffusion = any(
            isinstance(x, (CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor))
            for (_, x) in self.attn_processors.items()
        )
        if is_custom_diffusion:
            model_to_save = AttnProcsLayers(
                {
                    y: x
                    for (y, x) in self.attn_processors.items()
                    if isinstance(x, (CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor))
                }
            )
            state_dict = model_to_save.state_dict()
            for name, attn in self.attn_processors.items():
                if len(attn.state_dict()) == 0:
                    state_dict[name] = {}
        else:
            model_to_save = AttnProcsLayers(self.attn_processors)
            state_dict = model_to_save.state_dict()

        if weight_name is None:
            if to_diffusers:
                if safe_serialization:
                    weight_name = (
                        TORCH_CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE if is_custom_diffusion else TORCH_LORA_WEIGHT_NAME_SAFE
                    )
                else:
                    weight_name = TORCH_CUSTOM_DIFFUSION_WEIGHT_NAME if is_custom_diffusion else TORCH_LORA_WEIGHT_NAME
            else:
                weight_name = PADDLE_CUSTOM_DIFFUSION_WEIGHT_NAME if is_custom_diffusion else PADDLE_LORA_WEIGHT_NAME

        # choose save_function
        if save_function is None:
            if to_diffusers:
                if safe_serialization:
                    if is_torch_available():
                        _save_function = safetensors.torch.save_file
                        state_dict = convert_state_dict(state_dict, framework="torch")
                    else:
                        _save_function = safetensors.numpy.save_file
                        state_dict = convert_state_dict(state_dict, framework="numpy")

                    def save_function(weights, filename):
                        return _save_function(weights, filename, metadata={"format": "pt"})

                else:
                    if not is_torch_available():
                        raise ImportError(
                            "`to_diffusers=True` with `safe_serialization=False` requires the `torch library: `pip install torch`."
                        )
                    save_function = torch.save
                    state_dict = convert_state_dict(state_dict, framework="torch")
                state_dict = transpose_state_dict(state_dict)
            else:
                save_function = paddle.save

        # Save the model
        save_function(state_dict, os.path.join(save_directory, weight_name))

        logger.info(f"Model weights saved in {os.path.join(save_directory, weight_name)}")


class TextualInversionLoaderMixin:
    r"""
    Load textual inversion tokens and embeddings to the tokenizer and text encoder.
    """

    def maybe_convert_prompt(self, prompt: Union[str, List[str]], tokenizer: "PretrainedTokenizer"):
        r"""
        Processes prompts that include a special token corresponding to a multi-vector textual inversion embedding to
        be replaced with multiple special tokens each corresponding to one of the vectors. If the prompt has no textual
        inversion token or if the textual inversion token is a single vector, the input prompt is returned.
        """
        if not isinstance(prompt, List):
            prompts = [prompt]
        else:
            prompts = prompt

        prompts = [self._maybe_convert_prompt(p, tokenizer) for p in prompts]

        if not isinstance(prompt, List):
            return prompts[0]

        return prompts

    def _maybe_convert_prompt(self, prompt: str, tokenizer: "PretrainedTokenizer"):
        r"""
        Maybe convert a prompt into a "multi vector"-compatible prompt. If the prompt includes a token that corresponds
        to a multi-vector textual inversion embedding, this function will process the prompt so that the special token
        is replaced with multiple special tokens each corresponding to one of the vectors. If the prompt has no textual
        inversion token or a textual inversion token that is a single vector, the input prompt is simply returned.
        Parameters:
            prompt (`str`):
                The prompt to guide the image generation.
            tokenizer (`PretrainedTokenizer`):
                The tokenizer responsible for encoding the prompt into input tokens.
        Returns:
            `str`: The converted prompt
        """
        tokens = tokenizer.tokenize(prompt)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in tokenizer.added_tokens_encoder:
                replacement = token
                i = 1
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    replacement += f" {token}_{i}"
                    i += 1

                prompt = prompt.replace(token, replacement)

        return prompt

    def load_textual_inversion(
        self,
        pretrained_model_name_or_path: Union[str, List[str], Dict[str, paddle.Tensor], List[Dict[str, paddle.Tensor]]],
        token: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        r"""
        Load textual inversion embeddings into the text encoder of [`StableDiffusionPipeline`] (both 🤗 Diffusers and
        Automatic1111 formats are supported).
        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike` or `List[str or os.PathLike]` or `Dict` or `List[Dict]`):
                Can be either one of the following or a list of them:
                    - A string, the *model id* (for example `sd-concepts-library/low-poly-hd-logos-icons`) of a
                      pretrained model hosted on the Hub.
                    - A path to a *directory* (for example `./my_text_inversion_directory/`) containing the textual
                      inversion weights.
                    - A path to a *file* (for example `./my_text_inversions.pt`) containing textual inversion weights.
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).
            token (`str` or `List[str]`, *optional*):
                Override the token to use for the textual inversion weights. If `pretrained_model_name_or_path` is a
                list, then `token` must also be a list of equal length.
            weight_name (`str`, *optional*):
                Name of a custom weight file. This should be used when:
                    - The saved textual inversion file is in 🤗 Diffusers format, but was saved under a specific weight
                      name such as `text_inv.bin`.
                    - The saved textual inversion file is in the Automatic1111 format.
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
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
        Example:
        To load a textual inversion embedding vector in `ppdiffusers` format:
        ```py
        from ppdiffusers import StableDiffusionPipeline
        import paddle
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, paddle_dtype=paddle.float16)
        pipe.load_textual_inversion("sd-concepts-library/cat-toy")
        prompt = "A <cat-toy> backpack"
        image = pipe(prompt, num_inference_steps=50).images[0]
        image.save("cat-backpack.png")
        ```
        To load a textual inversion embedding vector in Automatic1111 format, make sure to first download the vector,
        e.g. from [civitAI](https://civitai.com/models/3036?modelVersionId=9857) and then load the vector locally:
        ```py
        from diffusers import StableDiffusionPipeline
        import paddle
        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, paddle_dtype=paddle.float16)
        pipe.load_textual_inversion("./charturnerv2.pt", token="charturnerv2")
        prompt = "charturnerv2, multiple views of the same character in the same outfit, a character turnaround of a woman wearing a black jacket and red shirt, best quality, intricate details."
        image = pipe(prompt, num_inference_steps=50).images[0]
        image.save("character.png")
        ```
        """
        if not hasattr(self, "tokenizer") or not isinstance(self.tokenizer, PretrainedTokenizer):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.tokenizer` of type `PretrainedTokenizer` for calling"
                f" `{self.load_textual_inversion.__name__}`"
            )

        if not hasattr(self, "text_encoder") or not isinstance(self.text_encoder, PretrainedModel):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.text_encoder` of type `PretrainedModel` for calling"
                f" `{self.load_textual_inversion.__name__}`"
            )
        from_hf_hub = kwargs.pop("from_hf_hub", FROM_HF_HUB)
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
        from_diffusers = kwargs.pop("from_diffusers", FROM_DIFFUSERS)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        if from_diffusers and use_safetensors and not is_safetensors_available():
            raise ValueError(
                "`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetensors"
            )
        if use_safetensors is None:
            use_safetensors = is_safetensors_available()
        user_agent = {
            "file_type": "text_inversion",
            "framework": "pytorch" if from_diffusers else "paddle",
        }

        if not isinstance(pretrained_model_name_or_path, list):
            pretrained_model_name_or_paths = [pretrained_model_name_or_path]
        else:
            pretrained_model_name_or_paths = pretrained_model_name_or_path

        if isinstance(token, str):
            tokens = [token]
        elif token is None:
            tokens = [None] * len(pretrained_model_name_or_paths)
        else:
            tokens = token

        if len(pretrained_model_name_or_paths) != len(tokens):
            raise ValueError(
                f"You have passed a list of models of length {len(pretrained_model_name_or_paths)}, and list of tokens of length {len(tokens)}"
                f"Make sure both lists have the same length."
            )

        valid_tokens = [t for t in tokens if t is not None]
        if len(set(valid_tokens)) < len(valid_tokens):
            raise ValueError(f"You have passed a list of tokens that contains duplicates: {tokens}")

        token_ids_and_embeddings = []

        for pretrained_model_name_or_path, token in zip(pretrained_model_name_or_paths, tokens):
            if not isinstance(pretrained_model_name_or_path, dict):
                # 1. Load textual inversion file
                model_file = None
                # Let's first try to load .safetensors weights
                if from_diffusers:
                    if (use_safetensors and weight_name is None) or (
                        weight_name is not None and weight_name.endswith(".safetensors")
                    ):
                        try:
                            model_file = _get_model_file(
                                pretrained_model_name_or_path,
                                weights_name=weight_name or TORCH_TEXT_INVERSION_NAME_SAFE,
                                cache_dir=cache_dir,
                                force_download=force_download,
                                resume_download=resume_download,
                                proxies=proxies,
                                local_files_only=local_files_only,
                                use_auth_token=use_auth_token,
                                revision=revision,
                                subfolder=subfolder,
                                user_agent=user_agent,
                                from_hf_hub=from_hf_hub,
                            )
                            state_dict = safetensors_load(model_file)
                        except Exception:
                            model_file = None
                            pass
                    if model_file is None:
                        model_file = _get_model_file(
                            pretrained_model_name_or_path,
                            weights_name=weight_name or TORCH_TEXT_INVERSION_NAME,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            resume_download=resume_download,
                            proxies=proxies,
                            local_files_only=local_files_only,
                            use_auth_token=use_auth_token,
                            revision=revision,
                            subfolder=subfolder,
                            user_agent=user_agent,
                            from_hf_hub=from_hf_hub,
                        )
                        state_dict = torch_load(model_file)
                else:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=weight_name or PADDLE_TEXT_INVERSION_NAME,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        from_hf_hub=from_hf_hub,
                    )
                if is_torch_file(model_file):
                    try:
                        state_dict = safetensors_load(model_file)
                    except:
                        state_dict = torch_load(model_file)
                else:
                    state_dict = paddle.load(model_file)
            else:
                state_dict = pretrained_model_name_or_path

            # 2. Load token and embedding correcly from file
            loaded_token = None
            if isinstance(state_dict, paddle.Tensor):
                if token is None:
                    raise ValueError(
                        "You are trying to load a textual inversion embedding that has been saved as a PyTorch tensor. Make sure to pass the name of the corresponding token in this case: `token=...`."
                    )
                embedding = state_dict
            elif len(state_dict) == 1:
                # diffusers
                loaded_token, embedding = next(iter(state_dict.items()))
            elif "string_to_param" in state_dict:
                # A1111
                loaded_token = state_dict["name"]
                embedding = state_dict["string_to_param"]["*"]

            if token is not None and loaded_token != token:
                logger.info(f"The loaded token: {loaded_token} is overwritten by the passed token {token}.")
            else:
                token = loaded_token

            if not isinstance(state_dict, paddle.Tensor):
                if hasattr(embedding, "detach"):
                    embedding = embedding.detach()
                if hasattr(embedding, "cpu"):
                    embedding = embedding.cpu()
                if hasattr(embedding, "numpy"):
                    embedding = embedding.numpy()
                embedding = paddle.to_tensor(embedding)
            embedding = embedding.cast(dtype=self.text_encoder.dtype)

            # 3. Make sure we don't mess up the tokenizer or text encoder
            vocab = self.tokenizer.get_vocab()
            if token in vocab:
                raise ValueError(
                    f"Token {token} already in tokenizer vocabulary. Please choose a different token name or remove {token} and embedding from the tokenizer and text encoder."
                )
            elif f"{token}_1" in vocab:
                multi_vector_tokens = [token]
                i = 1
                while f"{token}_{i}" in self.tokenizer.added_tokens_encoder:
                    multi_vector_tokens.append(f"{token}_{i}")
                    i += 1

                raise ValueError(
                    f"Multi-vector Token {multi_vector_tokens} already in tokenizer vocabulary. Please choose a different token name or remove the {multi_vector_tokens} and embedding from the tokenizer and text encoder."
                )

            is_multi_vector = len(embedding.shape) > 1 and embedding.shape[0] > 1

            if is_multi_vector:
                tokens = [token] + [f"{token}_{i}" for i in range(1, embedding.shape[0])]
                embeddings = [e for e in embedding]  # noqa: C416
            else:
                tokens = [token]
                embeddings = [embedding[0]] if len(embedding.shape) > 1 else [embedding]

            # add tokens and get ids
            self.tokenizer.add_tokens(tokens)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_ids_and_embeddings += zip(token_ids, embeddings)
            logger.info(f"Loaded textual inversion embedding for {token}.")

        # resize token embeddings and set all new embeddings
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        with paddle.no_grad():
            for token_id, embedding in token_ids_and_embeddings:
                self.text_encoder.get_input_embeddings().weight[token_id] = embedding


class LoraLoaderMixin:
    r"""
    Load LoRA layers into [`UNet2DConditionModel`] and
    [`CLIPTextModel`](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel).
    """
    text_encoder_name = TEXT_ENCODER_NAME
    unet_name = UNET_NAME

    def load_lora_weights(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, paddle.Tensor]], **kwargs):
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
        """
        state_dict, network_alphas = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)
        self.load_lora_into_unet(state_dict, network_alphas=network_alphas, unet=self.unet)
        self.load_lora_into_text_encoder(
            state_dict,
            network_alphas=network_alphas,
            text_encoder=self.text_encoder,
            lora_scale=self.lora_scale,
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
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.

        """
        # Load the main state dict first which has the LoRA layers for either of
        # UNet and text encoder or both.
        from_hf_hub = kwargs.pop("from_hf_hub", FROM_HF_HUB)
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
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

        if from_diffusers and use_safetensors and not is_safetensors_available():
            raise ValueError(
                "`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetensors"
            )
        if use_safetensors is None:
            use_safetensors = is_safetensors_available()

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch" if from_diffusers else "paddle",
        }

        model_file = None

        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            if from_diffusers:
                # Let's first try to load .safetensors weights
                if (use_safetensors and weight_name is None) or (
                    weight_name is not None and weight_name.endswith(".safetensors")
                ):
                    try:
                        model_file = _get_model_file(
                            pretrained_model_name_or_path_or_dict,
                            weights_name=weight_name or TORCH_LORA_WEIGHT_NAME_SAFE,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            resume_download=resume_download,
                            proxies=proxies,
                            local_files_only=local_files_only,
                            use_auth_token=use_auth_token,
                            revision=revision,
                            subfolder=subfolder,
                            user_agent=user_agent,
                            from_hf_hub=from_hf_hub,
                        )
                        state_dict = smart_load(model_file)
                    except Exception:
                        model_file = None
                        pass
                if model_file is None:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weight_name or TORCH_LORA_WEIGHT_NAME,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        from_hf_hub=from_hf_hub,
                    )
                    state_dict = smart_load(model_file)
            else:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or PADDLE_LORA_WEIGHT_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    from_hf_hub=from_hf_hub,
                )
                state_dict = smart_load(model_file)
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        network_alphas = None
        if all(
            (
                k.startswith("lora_te_")
                or k.startswith("lora_unet_")
                or k.startswith("lora_te1_")
                or k.startswith("lora_te2_")
            )
            for k in state_dict.keys()
        ):
            from_diffusers = True
            # Map SDXL blocks correctly.
            if unet_config is not None:
                # use unet config to remap block numbers
                state_dict = cls._map_sgm_blocks_to_diffusers(state_dict, unet_config)
            state_dict, network_alphas = cls._convert_kohya_lora_to_diffusers(state_dict)

        if from_diffusers:
            # convert diffusers name to pppdiffusers name
            name_mapping_dict = {
                ".encoder.": ".transformer.",
                ".layer_norm": ".norm",
                ".mlp.": ".",
                ".fc1.": ".linear1.",
                ".fc2.": ".linear2.",
                ".final_layer_norm.": ".ln_final.",
                ".embeddings.": ".",
                ".position_embedding.": ".positional_embedding.",
                ".patch_embedding.": ".conv1.",
                "visual_projection.weight": "vision_projection",
                "text_projection.weight": "text_projection",
                ".pre_layrnorm.": ".ln_pre.",
                ".post_layernorm.": ".ln_post.",
            }
            state_dict_keys = list(state_dict.keys())
            new_state_dict = {}
            for k in state_dict_keys:
                tensor = state_dict.pop(k)
                if tensor.ndim == 2:
                    tensor = tensor.T
                for oldk, newk in name_mapping_dict.items():
                    k = k.replace(oldk, newk)
                new_state_dict[k] = tensor
            state_dict = new_state_dict
        return state_dict, network_alphas

    @classmethod
    def _map_sgm_blocks_to_diffusers(cls, state_dict, unet_config, delimiter="_", block_slice_pos=5):
        is_all_unet = all(k.startswith("lora_unet") for k in state_dict)
        new_state_dict = {}
        inner_block_map = ["resnets", "attentions", "upsamplers"]

        # Retrieves # of down, mid and up blocks
        input_block_ids, middle_block_ids, output_block_ids = set(), set(), set()
        for layer in state_dict:
            if "text" not in layer:
                layer_id = int(layer.split(delimiter)[:block_slice_pos][-1])
                if "input_blocks" in layer:
                    input_block_ids.add(layer_id)
                elif "middle_block" in layer:
                    middle_block_ids.add(layer_id)
                elif "output_blocks" in layer:
                    output_block_ids.add(layer_id)
                else:
                    raise ValueError("Checkpoint not supported")

        input_blocks = {
            layer_id: [key for key in state_dict if f"input_blocks{delimiter}{layer_id}" in key]
            for layer_id in input_block_ids
        }
        middle_blocks = {
            layer_id: [key for key in state_dict if f"middle_block{delimiter}{layer_id}" in key]
            for layer_id in middle_block_ids
        }
        output_blocks = {
            layer_id: [key for key in state_dict if f"output_blocks{delimiter}{layer_id}" in key]
            for layer_id in output_block_ids
        }

        # Rename keys accordingly
        for i in input_block_ids:
            block_id = (i - 1) // (unet_config.layers_per_block + 1)
            layer_in_block_id = (i - 1) % (unet_config.layers_per_block + 1)

            for key in input_blocks[i]:
                inner_block_id = int(key.split(delimiter)[block_slice_pos])
                inner_block_key = inner_block_map[inner_block_id] if "op" not in key else "downsamplers"
                inner_layers_in_block = str(layer_in_block_id) if "op" not in key else "0"
                new_key = delimiter.join(
                    key.split(delimiter)[: block_slice_pos - 1]
                    + [str(block_id), inner_block_key, inner_layers_in_block]
                    + key.split(delimiter)[block_slice_pos + 1 :]
                )
                new_state_dict[new_key] = state_dict.pop(key)

        for i in middle_block_ids:
            key_part = None
            if i == 0:
                key_part = [inner_block_map[0], "0"]
            elif i == 1:
                key_part = [inner_block_map[1], "0"]
            elif i == 2:
                key_part = [inner_block_map[0], "1"]
            else:
                raise ValueError(f"Invalid middle block id {i}.")

            for key in middle_blocks[i]:
                new_key = delimiter.join(
                    key.split(delimiter)[: block_slice_pos - 1] + key_part + key.split(delimiter)[block_slice_pos:]
                )
                new_state_dict[new_key] = state_dict.pop(key)

        for i in output_block_ids:
            block_id = i // (unet_config.layers_per_block + 1)
            layer_in_block_id = i % (unet_config.layers_per_block + 1)

            for key in output_blocks[i]:
                inner_block_id = int(key.split(delimiter)[block_slice_pos])
                inner_block_key = inner_block_map[inner_block_id]
                inner_layers_in_block = str(layer_in_block_id) if inner_block_id < 2 else "0"
                new_key = delimiter.join(
                    key.split(delimiter)[: block_slice_pos - 1]
                    + [str(block_id), inner_block_key, inner_layers_in_block]
                    + key.split(delimiter)[block_slice_pos + 1 :]
                )
                new_state_dict[new_key] = state_dict.pop(key)

        if is_all_unet and len(state_dict) > 0:
            raise ValueError("At this point all state dict entries have to be converted.")
        else:
            # Remaining is the text encoder state dict.
            for k, v in state_dict.items():
                new_state_dict.update({k: v})

        return new_state_dict

    @classmethod
    def load_lora_into_unet(cls, state_dict, network_alphas, unet):
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
        """
        # If the serialization format is new (introduced in https://github.com/huggingface/diffusers/pull/2918),
        # then the `state_dict` keys should have `self.unet_name` and/or `self.text_encoder_name` as
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
            warn_message = "You have saved the LoRA weights using the old format. To convert the old LoRA weights to the new format, you can first load them in a dictionary and then create a new dictionary like the following: `new_state_dict = {f'unet'.{module_name}: params for module_name, params in old_state_dict.items()}`."
            warnings.warn(warn_message)

        # load loras into unet
        # make sure we set from_diffusers=False
        unet.load_attn_procs(state_dict, network_alphas=network_alphas, from_diffusers=False)

    @classmethod
    def load_lora_into_text_encoder(cls, state_dict, network_alphas, text_encoder, prefix=None, lora_scale=1.0):
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
        """

        # If the serialization format is new (introduced in https://github.com/huggingface/diffusers/pull/2918),
        # then the `state_dict` keys should have `self.unet_name` and/or `self.text_encoder_name` as
        # their prefixes.
        keys = list(state_dict.keys())
        prefix = cls.text_encoder_name if prefix is None else prefix

        if any(cls.text_encoder_name in key for key in keys):
            # Load the layers corresponding to text encoder and make necessary adjustments.
            text_encoder_keys = [k for k in keys if k.startswith(prefix)]
            text_encoder_lora_state_dict = {
                k.replace(f"{prefix}.", ""): v for k, v in state_dict.items() if k in text_encoder_keys
            }

            if len(text_encoder_lora_state_dict) > 0:
                logger.info(f"Loading {prefix}.")

                if any("to_out_lora" in k for k in text_encoder_lora_state_dict.keys()):
                    # Convert from the old naming convention to the new naming convention.
                    #
                    # Previously, the old LoRA layers were stored on the state dict at the
                    # same level as the attention block i.e.
                    # `text_model.encoder.layers.11.self_attn.to_out_lora.up.weight`.
                    #
                    # This is no actual module at that point, they were monkey patched on to the
                    # existing module. We want to be able to load them via their actual state dict.
                    # They're in `PatchedLoraProjection.lora_linear_layer` now.
                    for name, _ in text_encoder_attn_modules(text_encoder):
                        text_encoder_lora_state_dict[
                            f"{name}.q_proj.lora_linear_layer.up.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_q_lora.up.weight")
                        text_encoder_lora_state_dict[
                            f"{name}.k_proj.lora_linear_layer.up.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_k_lora.up.weight")
                        text_encoder_lora_state_dict[
                            f"{name}.v_proj.lora_linear_layer.up.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_v_lora.up.weight")
                        text_encoder_lora_state_dict[
                            f"{name}.out_proj.lora_linear_layer.up.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_out_lora.up.weight")

                        text_encoder_lora_state_dict[
                            f"{name}.q_proj.lora_linear_layer.down.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_q_lora.down.weight")
                        text_encoder_lora_state_dict[
                            f"{name}.k_proj.lora_linear_layer.down.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_k_lora.down.weight")
                        text_encoder_lora_state_dict[
                            f"{name}.v_proj.lora_linear_layer.down.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_v_lora.down.weight")
                        text_encoder_lora_state_dict[
                            f"{name}.out_proj.lora_linear_layer.down.weight"
                        ] = text_encoder_lora_state_dict.pop(f"{name}.to_out_lora.down.weight")

                rank = text_encoder_lora_state_dict[
                    "text_model.transformer.layers.0.self_attn.out_proj.lora_linear_layer.up.weight"
                ].shape[0]
                patch_mlp = any(".linear1." in key for key in text_encoder_lora_state_dict.keys())

                cls._modify_text_encoder(
                    text_encoder,
                    lora_scale,
                    network_alphas,
                    rank=rank,
                    patch_mlp=patch_mlp,
                )

                # set correct dtype & device
                text_encoder_lora_state_dict = {
                    k: v._to(dtype=text_encoder.dtype) for k, v in text_encoder_lora_state_dict.items()
                }
                text_encoder.load_dict(text_encoder_lora_state_dict)
                # load_state_dict_results = text_encoder.load_dict(text_encoder_lora_state_dict)
                # if len(load_state_dict_results.unexpected_keys) != 0:
                #     raise ValueError(
                #         f"failed to load text encoder state dict, unexpected keys: {load_state_dict_results.unexpected_keys}"
                #     )

    @property
    def lora_scale(self) -> float:
        # property function that returns the lora scale which can be set at run time by the pipeline.
        # if _lora_scale has not been set, return 1
        return self._lora_scale if hasattr(self, "_lora_scale") else 1.0

    def _remove_text_encoder_monkey_patch(self):
        self._remove_text_encoder_monkey_patch_classmethod(self.text_encoder)

    @classmethod
    def _remove_text_encoder_monkey_patch_classmethod(cls, text_encoder):
        for _, attn_module in text_encoder_attn_modules(text_encoder):
            if isinstance(attn_module.q_proj, PatchedLoraProjection):
                attn_module.q_proj = attn_module.q_proj.regular_linear_layer
                attn_module.k_proj = attn_module.k_proj.regular_linear_layer
                attn_module.v_proj = attn_module.v_proj.regular_linear_layer
                attn_module.out_proj = attn_module.out_proj.regular_linear_layer

        for _, mlp_module in text_encoder_mlp_modules(text_encoder):
            if isinstance(mlp_module.linear1, PatchedLoraProjection):
                mlp_module.linear1 = mlp_module.linear1.regular_linear_layer
            if isinstance(mlp_module.linear2, PatchedLoraProjection):
                mlp_module.linear2 = mlp_module.linear2.regular_linear_layer

    # @classmethod
    # def _modify_text_encoder(
    #     cls,
    #     text_encoder,
    #     lora_scale=1,
    #     network_alphas=None,
    #     rank: Union[Dict[str, int], int] = 4,
    #     dtype=None,
    #     patch_mlp=False,
    # ):
    #     r"""
    #     Monkey-patches the forward passes of attention modules of the text encoder.
    #     """

    #     # First, remove any monkey-patch that might have been applied before
    #     cls._remove_text_encoder_monkey_patch_classmethod(text_encoder)

    #     lora_parameters = []
    #     network_alphas = {} if network_alphas is None else network_alphas
    #     is_network_alphas_populated = len(network_alphas) > 0

    #     for name, attn_module in text_encoder_attn_modules(text_encoder):
    #         query_alpha = network_alphas.pop(name + ".to_q_lora.down.weight.alpha", None)
    #         key_alpha = network_alphas.pop(name + ".to_k_lora.down.weight.alpha", None)
    #         value_alpha = network_alphas.pop(name + ".to_v_lora.down.weight.alpha", None)
    #         out_alpha = network_alphas.pop(name + ".to_out_lora.down.weight.alpha", None)

    #         if isinstance(rank, dict):
    #             current_rank = rank.pop(f"{name}.out_proj.lora_linear_layer.up.weight")
    #         else:
    #             current_rank = rank

    #         q_linear_layer = (
    #             attn_module.q_proj.regular_linear_layer
    #             if isinstance(attn_module.q_proj, PatchedLoraProjection)
    #             else attn_module.q_proj
    #         )
    #         attn_module.q_proj = PatchedLoraProjection(
    #             q_linear_layer, lora_scale, network_alpha=query_alpha, rank=current_rank, dtype=dtype
    #         )
    #         lora_parameters.extend(attn_module.q_proj.lora_linear_layer.parameters())

    #         k_linear_layer = (
    #             attn_module.k_proj.regular_linear_layer
    #             if isinstance(attn_module.k_proj, PatchedLoraProjection)
    #             else attn_module.k_proj
    #         )
    #         attn_module.k_proj = PatchedLoraProjection(
    #             k_linear_layer, lora_scale, network_alpha=key_alpha, rank=current_rank, dtype=dtype
    #         )
    #         lora_parameters.extend(attn_module.k_proj.lora_linear_layer.parameters())

    #         v_linear_layer = (
    #             attn_module.v_proj.regular_linear_layer
    #             if isinstance(attn_module.v_proj, PatchedLoraProjection)
    #             else attn_module.v_proj
    #         )
    #         attn_module.v_proj = PatchedLoraProjection(
    #             v_linear_layer, lora_scale, network_alpha=value_alpha, rank=current_rank, dtype=dtype
    #         )
    #         lora_parameters.extend(attn_module.v_proj.lora_linear_layer.parameters())

    #         out_linear_layer = (
    #             attn_module.out_proj.regular_linear_layer
    #             if isinstance(attn_module.out_proj, PatchedLoraProjection)
    #             else attn_module.out_proj
    #         )
    #         attn_module.out_proj = PatchedLoraProjection(
    #             out_linear_layer, lora_scale, network_alpha=out_alpha, rank=current_rank, dtype=dtype
    #         )
    #         lora_parameters.extend(attn_module.out_proj.lora_linear_layer.parameters())

    #     if patch_mlp:
    #         for name, mlp_module in text_encoder_mlp_modules(text_encoder):
    #             fc1_alpha = network_alphas.pop(name + ".fc1.lora_linear_layer.down.weight.alpha", None)
    #             fc2_alpha = network_alphas.pop(name + ".fc2.lora_linear_layer.down.weight.alpha", None)

    #             current_rank_fc1 = rank.pop(f"{name}.fc1.lora_linear_layer.up.weight")
    #             current_rank_fc2 = rank.pop(f"{name}.fc2.lora_linear_layer.up.weight")

    #             fc1_linear_layer = (
    #                 mlp_module.fc1.regular_linear_layer
    #                 if isinstance(mlp_module.fc1, PatchedLoraProjection)
    #                 else mlp_module.fc1
    #             )
    #             mlp_module.fc1 = PatchedLoraProjection(
    #                 fc1_linear_layer, lora_scale, network_alpha=fc1_alpha, rank=current_rank_fc1, dtype=dtype
    #             )
    #             lora_parameters.extend(mlp_module.fc1.lora_linear_layer.parameters())

    #             fc2_linear_layer = (
    #                 mlp_module.fc2.regular_linear_layer
    #                 if isinstance(mlp_module.fc2, PatchedLoraProjection)
    #                 else mlp_module.fc2
    #             )
    #             mlp_module.fc2 = PatchedLoraProjection(
    #                 fc2_linear_layer, lora_scale, network_alpha=fc2_alpha, rank=current_rank_fc2, dtype=dtype
    #             )
    #             lora_parameters.extend(mlp_module.fc2.lora_linear_layer.parameters())

    #     if is_network_alphas_populated and len(network_alphas) > 0:
    #         raise ValueError(
    #             f"The `network_alphas` has to be empty at this point but has the following keys \n\n {', '.join(network_alphas.keys())}"
    #         )

    #     return lora_parameters

    @classmethod
    def _modify_text_encoder(
        cls,
        text_encoder,
        lora_scale=1,
        network_alphas=None,
        rank=4,
        dtype=None,
        patch_mlp=False,
    ):
        r"""
        Monkey-patches the forward passes of attention modules of the text encoder.
        """

        # First, remove any monkey-patch that might have been applied before
        cls._remove_text_encoder_monkey_patch_classmethod(text_encoder)

        lora_parameters = []
        network_alphas = {} if network_alphas is None else network_alphas

        for name, attn_module in text_encoder_attn_modules(text_encoder):
            query_alpha = network_alphas.get(name + ".k.proj.alpha")
            key_alpha = network_alphas.get(name + ".q.proj.alpha")
            value_alpha = network_alphas.get(name + ".v.proj.alpha")
            proj_alpha = network_alphas.get(name + ".out.proj.alpha")

            attn_module.q_proj = PatchedLoraProjection(
                attn_module.q_proj, lora_scale, network_alpha=query_alpha, rank=rank, dtype=dtype
            )
            lora_parameters.extend(attn_module.q_proj.lora_linear_layer.parameters())

            attn_module.k_proj = PatchedLoraProjection(
                attn_module.k_proj, lora_scale, network_alpha=key_alpha, rank=rank, dtype=dtype
            )
            lora_parameters.extend(attn_module.k_proj.lora_linear_layer.parameters())

            attn_module.v_proj = PatchedLoraProjection(
                attn_module.v_proj, lora_scale, network_alpha=value_alpha, rank=rank, dtype=dtype
            )
            lora_parameters.extend(attn_module.v_proj.lora_linear_layer.parameters())

            attn_module.out_proj = PatchedLoraProjection(
                attn_module.out_proj, lora_scale, network_alpha=proj_alpha, rank=rank, dtype=dtype
            )
            lora_parameters.extend(attn_module.out_proj.lora_linear_layer.parameters())

        if patch_mlp:
            for name, mlp_module in text_encoder_mlp_modules(text_encoder):
                fc1_alpha = network_alphas.get(name + ".linear1.alpha")
                fc2_alpha = network_alphas.get(name + ".linear2.alpha")

                mlp_module.linear1 = PatchedLoraProjection(
                    mlp_module.linear1, lora_scale, network_alpha=fc1_alpha, rank=rank, dtype=dtype
                )
                lora_parameters.extend(mlp_module.linear1.lora_linear_layer.parameters())

                mlp_module.linear2 = PatchedLoraProjection(
                    mlp_module.linear2, lora_scale, network_alpha=fc2_alpha, rank=rank, dtype=dtype
                )
                lora_parameters.extend(mlp_module.linear2.lora_linear_layer.parameters())

        return lora_parameters

    @classmethod
    def save_lora_weights(
        self,
        save_directory: Union[str, os.PathLike],
        unet_lora_layers: Dict[str, Union[nn.Layer, paddle.Tensor]] = None,
        text_encoder_lora_layers: Dict[str, nn.Layer] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = False,
        to_diffusers: Optional[bool] = None,
    ):
        r"""
        Save the LoRA parameters corresponding to the UNet and text encoder.
        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save LoRA parameters to. Will be created if it doesn't exist.
            unet_lora_layers (`Dict[str, paddle.nn.Layer]` or `Dict[str, paddle.Tensor]`):
                State dict of the LoRA layers corresponding to the `unet`.
            text_encoder_lora_layers (`Dict[str, paddle.nn.Layer]` or `Dict[str, paddle.Tensor]`):
                State dict of the LoRA layers corresponding to the `text_encoder`. Must explicitly pass the text
                encoder LoRA state dict because it comes from 🤗 Transformers.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `paddle.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
        """
        # Create a flat dictionary.
        state_dict = {}

        # Populate the dictionary.
        if unet_lora_layers is not None:
            weights = unet_lora_layers.state_dict() if isinstance(unet_lora_layers, nn.Layer) else unet_lora_layers

            unet_lora_state_dict = {f"{self.unet_name}.{module_name}": param for module_name, param in weights.items()}
            state_dict.update(unet_lora_state_dict)

        if text_encoder_lora_layers is not None:
            weights = (
                text_encoder_lora_layers.state_dict()
                if isinstance(text_encoder_lora_layers, nn.Layer)
                else text_encoder_lora_layers
            )

            text_encoder_lora_state_dict = {
                f"{self.text_encoder_name}.{module_name}": param for module_name, param in weights.items()
            }
            state_dict.update(text_encoder_lora_state_dict)

        # Save the model
        self.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
            to_diffusers=to_diffusers,
        )

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
        if to_diffusers and safe_serialization and not is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

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
                weight_name = PADDLE_LORA_WEIGHT_NAME

        # choose save_function
        if save_function is None:
            if to_diffusers:
                if safe_serialization:
                    if is_torch_available():
                        _save_function = safetensors.torch.save_file
                        state_dict = convert_state_dict(state_dict, framework="torch")
                    else:
                        _save_function = safetensors.numpy.save_file
                        state_dict = convert_state_dict(state_dict, framework="numpy")

                    def save_function(weights, filename):
                        return _save_function(weights, filename, metadata={"format": "pt"})

                else:
                    if not is_torch_available():
                        raise ImportError(
                            "`to_diffusers=True` with `safe_serialization=False` requires the `torch library: `pip install torch`."
                        )
                    save_function = torch.save
                    state_dict = convert_state_dict(state_dict, framework="torch")
                state_dict = transpose_state_dict(state_dict, name_mapping={".transformer.": ".encoder."})
            else:
                save_function = paddle.save

        save_function(state_dict, os.path.join(save_directory, weight_name))
        logger.info(f"Model weights saved in {os.path.join(save_directory, weight_name)}")

    @classmethod
    def _convert_kohya_lora_to_diffusers(cls, state_dict):
        unet_state_dict = {}
        te_state_dict = {}
        te2_state_dict = {}
        network_alphas = {}

        # every down weight has a corresponding up weight and potentially an alpha weight
        lora_keys = [k for k in state_dict.keys() if k.endswith("lora_down.weight")]
        for key in lora_keys:
            lora_name = key.split(".")[0]
            lora_name_up = lora_name + ".lora_up.weight"
            lora_name_alpha = lora_name + ".alpha"

            # if lora_name_alpha in state_dict:
            #     alpha = state_dict.pop(lora_name_alpha).item()
            #     network_alphas.update({lora_name_alpha: alpha})

            if lora_name.startswith("lora_unet_"):
                diffusers_name = key.replace("lora_unet_", "").replace("_", ".")

                if "input.blocks" in diffusers_name:
                    diffusers_name = diffusers_name.replace("input.blocks", "down_blocks")
                else:
                    diffusers_name = diffusers_name.replace("down.blocks", "down_blocks")

                if "middle.block" in diffusers_name:
                    diffusers_name = diffusers_name.replace("middle.block", "mid_block")
                else:
                    diffusers_name = diffusers_name.replace("mid.block", "mid_block")
                if "output.blocks" in diffusers_name:
                    diffusers_name = diffusers_name.replace("output.blocks", "up_blocks")
                else:
                    diffusers_name = diffusers_name.replace("up.blocks", "up_blocks")

                diffusers_name = diffusers_name.replace("transformer.blocks", "transformer_blocks")
                diffusers_name = diffusers_name.replace("to.q.lora", "to_q_lora")
                diffusers_name = diffusers_name.replace("to.k.lora", "to_k_lora")
                diffusers_name = diffusers_name.replace("to.v.lora", "to_v_lora")
                diffusers_name = diffusers_name.replace("to.out.0.lora", "to_out_lora")
                diffusers_name = diffusers_name.replace("proj.in", "proj_in")
                diffusers_name = diffusers_name.replace("proj.out", "proj_out")
                diffusers_name = diffusers_name.replace("emb.layers", "time_emb_proj")

                # SDXL specificity.
                if "emb" in diffusers_name:
                    pattern = r"\.\d+(?=\D*$)"
                    diffusers_name = re.sub(pattern, "", diffusers_name, count=1)
                if ".in." in diffusers_name:
                    diffusers_name = diffusers_name.replace("in.layers.2", "conv1")
                if ".out." in diffusers_name:
                    diffusers_name = diffusers_name.replace("out.layers.3", "conv2")
                if "downsamplers" in diffusers_name or "upsamplers" in diffusers_name:
                    diffusers_name = diffusers_name.replace("op", "conv")
                if "skip" in diffusers_name:
                    diffusers_name = diffusers_name.replace("skip.connection", "conv_shortcut")

                if "transformer_blocks" in diffusers_name:
                    if "attn1" in diffusers_name or "attn2" in diffusers_name:
                        diffusers_name = diffusers_name.replace("attn1", "attn1.processor")
                        diffusers_name = diffusers_name.replace("attn2", "attn2.processor")
                        unet_state_dict[diffusers_name] = state_dict.pop(key)
                        unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
                    elif "ff" in diffusers_name:
                        unet_state_dict[diffusers_name] = state_dict.pop(key)
                        unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
                elif any(key in diffusers_name for key in ("proj_in", "proj_out")):
                    unet_state_dict[diffusers_name] = state_dict.pop(key)
                    unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
                else:
                    unet_state_dict[diffusers_name] = state_dict.pop(key)
                    unet_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)

            elif lora_name.startswith("lora_te_"):
                diffusers_name = key.replace("lora_te_", "").replace("_", ".")
                diffusers_name = diffusers_name.replace("text.model", "text_model")
                diffusers_name = diffusers_name.replace("self.attn", "self_attn")
                diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
                diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
                diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
                diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
                if "self_attn" in diffusers_name:
                    te_state_dict[diffusers_name] = state_dict.pop(key)
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
                elif "mlp" in diffusers_name:
                    # Be aware that this is the new diffusers convention and the rest of the code might
                    # not utilize it yet.
                    diffusers_name = diffusers_name.replace(".lora.", ".lora_linear_layer.")
                    te_state_dict[diffusers_name] = state_dict.pop(key)
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)

            # (sayakpaul): Duplicate code. Needs to be cleaned.
            elif lora_name.startswith("lora_te1_"):
                diffusers_name = key.replace("lora_te1_", "").replace("_", ".")
                diffusers_name = diffusers_name.replace("text.model", "text_model")
                diffusers_name = diffusers_name.replace("self.attn", "self_attn")
                diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
                diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
                diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
                diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
                if "self_attn" in diffusers_name:
                    te_state_dict[diffusers_name] = state_dict.pop(key)
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
                elif "mlp" in diffusers_name:
                    # Be aware that this is the new diffusers convention and the rest of the code might
                    # not utilize it yet.
                    diffusers_name = diffusers_name.replace(".lora.", ".lora_linear_layer.")
                    te_state_dict[diffusers_name] = state_dict.pop(key)
                    te_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)

            # (sayakpaul): Duplicate code. Needs to be cleaned.
            elif lora_name.startswith("lora_te2_"):
                diffusers_name = key.replace("lora_te2_", "").replace("_", ".")
                diffusers_name = diffusers_name.replace("text.model", "text_model")
                diffusers_name = diffusers_name.replace("self.attn", "self_attn")
                diffusers_name = diffusers_name.replace("q.proj.lora", "to_q_lora")
                diffusers_name = diffusers_name.replace("k.proj.lora", "to_k_lora")
                diffusers_name = diffusers_name.replace("v.proj.lora", "to_v_lora")
                diffusers_name = diffusers_name.replace("out.proj.lora", "to_out_lora")
                if "self_attn" in diffusers_name:
                    te2_state_dict[diffusers_name] = state_dict.pop(key)
                    te2_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)
                elif "mlp" in diffusers_name:
                    # Be aware that this is the new diffusers convention and the rest of the code might
                    # not utilize it yet.
                    diffusers_name = diffusers_name.replace(".lora.", ".lora_linear_layer.")
                    te2_state_dict[diffusers_name] = state_dict.pop(key)
                    te2_state_dict[diffusers_name.replace(".down.", ".up.")] = state_dict.pop(lora_name_up)

            # Rename the alphas so that they can be mapped appropriately.
            if lora_name_alpha in state_dict:
                alpha = state_dict.pop(lora_name_alpha).cast("float32").item()
                if lora_name_alpha.startswith("lora_unet_"):
                    prefix = "unet."
                elif lora_name_alpha.startswith(("lora_te_", "lora_te1_")):
                    prefix = "text_encoder."
                else:
                    prefix = "text_encoder_2."
                new_name = prefix + diffusers_name.split(".lora.")[0] + ".alpha"
                network_alphas.update({new_name: alpha})

        if len(state_dict) > 0:
            raise ValueError(
                f"The following keys have not been correctly be renamed: \n\n {', '.join(state_dict.keys())}"
            )

        logger.info("Kohya-style checkpoint detected.")
        unet_state_dict = {f"{cls.unet_name}.{module_name}": params for module_name, params in unet_state_dict.items()}
        te_state_dict = {
            f"{cls.text_encoder_name}.{module_name}": params for module_name, params in te_state_dict.items()
        }
        te2_state_dict = (
            {f"text_encoder_2.{module_name}": params for module_name, params in te2_state_dict.items()}
            if len(te2_state_dict) > 0
            else None
        )
        if te2_state_dict is not None:
            te_state_dict.update(te2_state_dict)

        new_state_dict = {**unet_state_dict, **te_state_dict}
        return new_state_dict, network_alphas

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
        from .models.attention_processor import (
            LORA_ATTENTION_PROCESSORS,
            AttnProcessor,
            AttnProcessor2_5,
            LoRAAttnAddedKVProcessor,
            LoRAAttnProcessor,
            LoRAAttnProcessor2_5,
            LoRAXFormersAttnProcessor,
            XFormersAttnProcessor,
        )

        unet_attention_classes = {type(processor) for _, processor in self.unet.attn_processors.items()}

        if unet_attention_classes.issubset(LORA_ATTENTION_PROCESSORS):
            # Handle attention processors that are a mix of regular attention and AddedKV
            # attention.
            if len(unet_attention_classes) > 1 or LoRAAttnAddedKVProcessor in unet_attention_classes:
                self.unet.set_default_attn_processor()
            else:
                regular_attention_classes = {
                    LoRAAttnProcessor: AttnProcessor,
                    LoRAAttnProcessor2_5: AttnProcessor2_5,
                    LoRAXFormersAttnProcessor: XFormersAttnProcessor,
                }
                [attention_proc_class] = unet_attention_classes
                self.unet.set_attn_processor(regular_attention_classes[attention_proc_class]())

            for _, module in self.unet.named_sublayers():
                if hasattr(module, "set_lora_layer"):
                    module.set_lora_layer(None)

        # Safe to call the following regardless of LoRA.
        self._remove_text_encoder_monkey_patch()


class FromSingleFileMixin:
    """
    Load model weights saved in the `.ckpt` format into a [`DiffusionPipeline`].
    """

    @classmethod
    def from_ckpt(cls, *args, **kwargs):
        deprecation_message = "The function `from_ckpt` is deprecated in favor of `from_single_file` and will be removed in diffusers v.0.21. Please make sure to use `StableDiffusionPipeline.from_single_file(...)` instead."
        deprecate("from_ckpt", "0.21.0", deprecation_message, standard_warn=False)
        return cls.from_single_file(*args, **kwargs)

    @classmethod
    def from_single_file(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Instantiate a [`DiffusionPipeline`] from pretrained pipeline weights saved in the `.ckpt` or `.safetensors`
        format. The pipeline is set in evaluation mode (`model.eval()`) by default.
        Parameters:
            pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            paddle_dtype (`str` or `paddle.dtype`, *optional*):
                Override the default `paddle.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
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
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            extract_ema (`bool`, *optional*, defaults to `False`):
                Whether to extract the EMA weights or not. Pass `True` to extract the EMA weights which usually yield
                higher quality images for inference. Non-EMA weights are usually better for continuing finetuning.
            upcast_attention (`bool`, *optional*, defaults to `None`):
                Whether the attention computation should always be upcasted.
            image_size (`int`, *optional*, defaults to 512):
                The image size the model was trained on. Use 512 for all Stable Diffusion v1 models and the Stable
                Diffusion v2 base model. Use 768 for Stable Diffusion v2.
            prediction_type (`str`, *optional*):
                The prediction type the model was trained on. Use `'epsilon'` for all Stable Diffusion v1 models and
                the Stable Diffusion v2 base model. Use `'v_prediction'` for Stable Diffusion v2.
            num_in_channels (`int`, *optional*, defaults to `None`):
                The number of input channels. If `None`, it is automatically inferred.
            scheduler_type (`str`, *optional*, defaults to `"pndm"`):
                Type of scheduler to use. Should be one of `["pndm", "lms", "heun", "euler", "euler-ancestral", "dpm",
                "ddim"]`.
            load_safety_checker (`bool`, *optional*, defaults to `True`):
                Whether to load the safety checker or not.
            text_encoder ([`~transformers.CLIPTextModel`], *optional*, defaults to `None`):
                An instance of `CLIPTextModel` to use, specifically the
                [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant. If this
                parameter is `None`, the function loads a new instance of `CLIPTextModel` by itself if needed.
            vae (`AutoencoderKL`, *optional*, defaults to `None`):
                Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations. If
                this parameter is `None`, the function will load a new instance of [CLIP] by itself, if needed.
            tokenizer ([`~transformers.CLIPTokenizer`], *optional*, defaults to `None`):
                An instance of `CLIPTokenizer` to use. If this parameter is `None`, the function loads a new instance
                of `CLIPTokenizer` by itself if needed.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (for example the pipeline components of the
                specific pipeline class). The overwritten components are directly passed to the pipelines `__init__`
                method. See example below for more information.
        Examples:
        ```py
        >>> from ppdiffusers import StableDiffusionPipeline
        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = StableDiffusionPipeline.from_single_file(
        ...     "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
        ... )
        >>> # Download pipeline from local file
        >>> # file is downloaded under ./v1-5-pruned-emaonly.ckpt
        >>> pipeline = StableDiffusionPipeline.from_single_file("./v1-5-pruned-emaonly")
        >>> # Enable float16
        >>> pipeline = StableDiffusionPipeline.from_single_file(
        ...     "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt",
        ...     paddle_dtype=paddle.float16,
        ... )
        ```
        """
        # import here to avoid circular dependency
        from .pipelines.stable_diffusion.convert_from_ckpt import (
            download_from_original_stable_diffusion_ckpt,
        )

        from_hf_hub = "huggingface.co" in pretrained_model_link_or_path or "hf.co"
        cache_dir = (
            kwargs.pop("cache_dir", DIFFUSERS_CACHE) if from_hf_hub else kwargs.pop("cache_dir", PPDIFFUSERS_CACHE)
        )
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        extract_ema = kwargs.pop("extract_ema", False)
        image_size = kwargs.pop("image_size", None)
        scheduler_type = kwargs.pop("scheduler_type", "pndm")
        num_in_channels = kwargs.pop("num_in_channels", None)
        upcast_attention = kwargs.pop("upcast_attention", None)
        load_safety_checker = kwargs.pop("load_safety_checker", False)
        prediction_type = kwargs.pop("prediction_type", None)
        text_encoder = kwargs.pop("text_encoder", None)
        vae = kwargs.pop("vae", None)
        controlnet = kwargs.pop("controlnet", None)
        tokenizer = kwargs.pop("tokenizer", None)

        paddle_dtype = kwargs.pop("paddle_dtype", None)

        use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)

        pipeline_name = cls.__name__
        file_extension = pretrained_model_link_or_path.rsplit(".", 1)[-1]
        from_safetensors = file_extension == "safetensors"

        if from_safetensors and use_safetensors is False:
            raise ValueError("Make sure to install `safetensors` with `pip install safetensors`.")

        # TODO: For now we only support stable diffusion
        stable_unclip = None
        model_type = None

        if pipeline_name in [
            "StableDiffusionControlNetPipeline",
            "StableDiffusionControlNetImg2ImgPipeline",
            "StableDiffusionControlNetInpaintPipeline",
        ]:
            from .models.controlnet import ControlNetModel
            from .pipelines.controlnet.multicontrolnet import MultiControlNetModel

            # Model type will be inferred from the checkpoint.
            if not isinstance(controlnet, (ControlNetModel, MultiControlNetModel)):
                raise ValueError("ControlNet needs to be passed if loading from ControlNet pipeline.")
        elif "StableDiffusion" in pipeline_name:
            # Model type will be inferred from the checkpoint.
            pass
        elif pipeline_name == "StableUnCLIPPipeline":
            model_type = "FrozenOpenCLIPEmbedder"
            stable_unclip = "txt2img"
        elif pipeline_name == "StableUnCLIPImg2ImgPipeline":
            model_type = "FrozenOpenCLIPEmbedder"
            stable_unclip = "img2img"
        elif pipeline_name == "PaintByExamplePipeline":
            model_type = "PaintByExample"
        elif pipeline_name == "LDMTextToImagePipeline":
            model_type = "LDMTextToImage"
        else:
            raise ValueError(f"Unhandled pipeline class: {pipeline_name}")

        pretrained_model_link_or_path = str(pretrained_model_link_or_path)
        if os.path.isfile(pretrained_model_link_or_path):
            checkpoint_path = pretrained_model_link_or_path
        elif pretrained_model_link_or_path.startswith("http://") or pretrained_model_link_or_path.startswith(
            "https://"
        ):
            # HF Hub models
            if any(p in pretrained_model_link_or_path for p in ["huggingface.co", "hf.co", "hf-mirror.com"]):
                # remove huggingface url
                for prefix in [
                    "https://huggingface.co/",
                    "huggingface.co/",
                    "hf.co/",
                    "https://hf.co/",
                    "https://hf-mirror.com",
                ]:
                    if pretrained_model_link_or_path.startswith(prefix):
                        pretrained_model_link_or_path = pretrained_model_link_or_path[len(prefix) :]

                # Code based on diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained
                ckpt_path = Path(pretrained_model_link_or_path)
                if not ckpt_path.is_file():
                    # get repo_id and (potentially nested) file path of ckpt in repo
                    repo_id = os.path.join(*ckpt_path.parts[:2])
                    file_path = os.path.join(*ckpt_path.parts[2:])

                    if file_path.startswith("blob/"):
                        file_path = file_path[len("blob/") :]

                    if file_path.startswith("main/"):
                        file_path = file_path[len("main/") :]

                    checkpoint_path = hf_hub_download(
                        repo_id,
                        filename=file_path,
                        cache_dir=cache_dir,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        force_download=force_download,
                    )
                else:
                    checkpoint_path = ckpt_path
            else:
                checkpoint_path = ppdiffusers_url_download(
                    pretrained_model_link_or_path,
                    cache_dir=cache_dir,
                    filename=http_file_name(pretrained_model_link_or_path).strip('"'),
                    force_download=force_download,
                    resume_download=resume_download,
                )
        else:
            checkpoint_path = pretrained_model_link_or_path

        pipe = download_from_original_stable_diffusion_ckpt(
            checkpoint_path,
            pipeline_class=cls,
            model_type=model_type,
            stable_unclip=stable_unclip,
            controlnet=controlnet,
            extract_ema=extract_ema,
            image_size=image_size,
            scheduler_type=scheduler_type,
            num_in_channels=num_in_channels,
            upcast_attention=upcast_attention,
            load_safety_checker=load_safety_checker,
            prediction_type=prediction_type,
            paddle_dtype=paddle_dtype,
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
        )

        return pipe


class FromOriginalVAEMixin:
    @classmethod
    def from_single_file(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Instantiate a [`AutoencoderKL`] from pretrained controlnet weights saved in the original `.ckpt` or
        `.safetensors` format. The pipeline is format. The pipeline is set in evaluation mode (`model.eval()`) by
        default.
        Parameters:
            pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            paddle_dtype (`str` or `paddle.dtype`, *optional*):
                Override the default `paddle.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to True, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            image_size (`int`, *optional*, defaults to 512):
                The image size the model was trained on. Use 512 for all Stable Diffusion v1 models and the Stable
                Diffusion v2 base model. Use 768 for Stable Diffusion v2.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            upcast_attention (`bool`, *optional*, defaults to `None`):
                Whether the attention computation should always be upcasted.
            scaling_factor (`float`, *optional*, defaults to 0.18215):
                The component-wise standard deviation of the trained latent space computed using the first batch of the
                training set. This is used to scale the latent space to have unit variance when training the diffusion
                model. The latents are scaled with the formula `z = z * scaling_factor` before being passed to the
                diffusion model. When decoding, the latents are scaled back to the original scale with the formula: `z
                = 1 / scaling_factor * z`. For more details, refer to sections 4.3.2 and D.1 of the [High-Resolution
                Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) paper.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (for example the pipeline components of the
                specific pipeline class). The overwritten components are directly passed to the pipelines `__init__`
                method. See example below for more information.
        <Tip warning={true}>
            Make sure to pass both `image_size` and `scaling_factor` to `from_single_file()` if you want to load
            a VAE that does accompany a stable diffusion model of v2 or higher or SDXL.
        </Tip>
        Examples:
        ```py
        from diffusers import AutoencoderKL
        url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"  # can also be local file
        model = AutoencoderKL.from_single_file(url)
        ```
        """
        if not is_omegaconf_available():
            raise ValueError(BACKENDS_MAPPING["omegaconf"][1])

        from omegaconf import OmegaConf

        from .models import AutoencoderKL

        # import here to avoid circular dependency
        from .pipelines.stable_diffusion.convert_from_ckpt import (
            convert_diffusers_vae_unet_to_ppdiffusers,
            convert_ldm_vae_checkpoint,
            create_vae_diffusers_config,
        )

        config_file = kwargs.pop("config_file", None)
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)
        image_size = kwargs.pop("image_size", None)
        scaling_factor = kwargs.pop("scaling_factor", None)
        kwargs.pop("upcast_attention", None)

        paddle_dtype = kwargs.pop("paddle_dtype", None)

        use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)

        file_extension = pretrained_model_link_or_path.rsplit(".", 1)[-1]
        from_safetensors = file_extension == "safetensors"

        if from_safetensors and use_safetensors is False:
            raise ValueError("Make sure to install `safetensors` with `pip install safetensors`.")

        pretrained_model_link_or_path = str(pretrained_model_link_or_path)
        if os.path.isfile(pretrained_model_link_or_path):
            checkpoint_path = pretrained_model_link_or_path
        elif pretrained_model_link_or_path.startswith("http://") or pretrained_model_link_or_path.startswith(
            "https://"
        ):
            # download hf models
            if any(p in pretrained_model_link_or_path for p in ["huggingface.co", "hf.co", "hf-mirror.com"]):
                # remove huggingface url
                for prefix in [
                    "https://huggingface.co/",
                    "huggingface.co/",
                    "hf.co/",
                    "https://hf.co/",
                    "https://hf-mirror.com",
                ]:
                    if pretrained_model_link_or_path.startswith(prefix):
                        pretrained_model_link_or_path = pretrained_model_link_or_path[len(prefix) :]

                # Code based on diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained
                ckpt_path = Path(pretrained_model_link_or_path)
                if not ckpt_path.is_file():
                    # get repo_id and (potentially nested) file path of ckpt in repo
                    repo_id = "/".join(ckpt_path.parts[:2])
                    file_path = "/".join(ckpt_path.parts[2:])

                    if file_path.startswith("blob/"):
                        file_path = file_path[len("blob/") :]

                    if file_path.startswith("main/"):
                        file_path = file_path[len("main/") :]

                    checkpoint_path = hf_hub_download(
                        repo_id,
                        filename=file_path,
                        cache_dir=cache_dir,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        force_download=force_download,
                    )
            else:
                checkpoint_path = ppdiffusers_url_download(
                    pretrained_model_link_or_path,
                    cache_dir=cache_dir,
                    filename=http_file_name(pretrained_model_link_or_path).strip('"'),
                    force_download=force_download,
                    resume_download=resume_download,
                )
        else:
            checkpoint_path = pretrained_model_link_or_path

        checkpoint = smart_load(checkpoint_path)

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        if config_file is None:
            config_url = "https://paddlenlp.bj.bcebos.com/models/community/junnyu/develop/v1-inference.yaml"
            config_file = BytesIO(requests.get(config_url).content)

        original_config = OmegaConf.load(config_file)

        # default to sd-v1-5
        image_size = image_size or 512

        vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

        if scaling_factor is None:
            if (
                "model" in original_config
                and "params" in original_config.model
                and "scale_factor" in original_config.model.params
            ):
                vae_scaling_factor = original_config.model.params.scale_factor
            else:
                vae_scaling_factor = 0.18215  # default SD scaling factor

        vae_config["scaling_factor"] = vae_scaling_factor

        vae = AutoencoderKL(**vae_config)

        # we must transpose linear layer
        vae.load_dict(convert_diffusers_vae_unet_to_ppdiffusers(vae, converted_vae_checkpoint))

        if paddle_dtype is not None:
            vae.to(paddle_dtype=paddle_dtype)

        return vae


class FromOriginalControlnetMixin:
    @classmethod
    def from_single_file(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Instantiate a [`ControlNetModel`] from pretrained controlnet weights saved in the original `.ckpt` or
        `.safetensors` format. The pipeline is set in evaluation mode (`model.eval()`) by default.
        Parameters:
            pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            paddle_dtype (`str` or `paddle.dtype`, *optional*):
                Override the default `paddle.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to True, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            image_size (`int`, *optional*, defaults to 512):
                The image size the model was trained on. Use 512 for all Stable Diffusion v1 models and the Stable
                Diffusion v2 base model. Use 768 for Stable Diffusion v2.
            upcast_attention (`bool`, *optional*, defaults to `None`):
                Whether the attention computation should always be upcasted.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (for example the pipeline components of the
                specific pipeline class). The overwritten components are directly passed to the pipelines `__init__`
                method. See example below for more information.
        Examples:
        ```py
        from ppdiffusers import StableDiffusionControlnetPipeline, ControlNetModel
        url = "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth"  # can also be a local path
        model = ControlNetModel.from_single_file(url)
        url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.safetensors"  # can also be a local path
        pipe = StableDiffusionControlnetPipeline.from_single_file(url, controlnet=controlnet)
        ```
        """
        # import here to avoid circular dependency
        from .pipelines.stable_diffusion.convert_from_ckpt import (
            download_controlnet_from_original_ckpt,
        )

        config_file = kwargs.pop("config_file", None)
        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        num_in_channels = kwargs.pop("num_in_channels", None)
        use_linear_projection = kwargs.pop("use_linear_projection", None)
        revision = kwargs.pop("revision", None)
        extract_ema = kwargs.pop("extract_ema", False)
        image_size = kwargs.pop("image_size", None)
        upcast_attention = kwargs.pop("upcast_attention", None)

        paddle_dtype = kwargs.pop("paddle_dtype", None)

        use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)

        file_extension = pretrained_model_link_or_path.rsplit(".", 1)[-1]
        from_safetensors = file_extension == "safetensors"

        if from_safetensors and use_safetensors is False:
            raise ValueError("Make sure to install `safetensors` with `pip install safetensors`.")

        pretrained_model_link_or_path = str(pretrained_model_link_or_path)
        if os.path.isfile(pretrained_model_link_or_path):
            checkpoint_path = pretrained_model_link_or_path
        elif pretrained_model_link_or_path.startswith("http://") or pretrained_model_link_or_path.startswith(
            "https://"
        ):
            # HF Hub models
            if any(p in pretrained_model_link_or_path for p in ["huggingface.co", "hf.co", "hf-mirror.com"]):
                # remove huggingface url
                for prefix in [
                    "https://huggingface.co/",
                    "huggingface.co/",
                    "hf.co/",
                    "https://hf.co/",
                    "https://hf-mirror.com",
                ]:
                    if pretrained_model_link_or_path.startswith(prefix):
                        pretrained_model_link_or_path = pretrained_model_link_or_path[len(prefix) :]

                # Code based on diffusers.pipelines.pipeline_utils.DiffusionPipeline.from_pretrained
                ckpt_path = Path(pretrained_model_link_or_path)
                if not ckpt_path.is_file():
                    # get repo_id and (potentially nested) file path of ckpt in repo
                    repo_id = "/".join(ckpt_path.parts[:2])
                    file_path = "/".join(ckpt_path.parts[2:])

                    if file_path.startswith("blob/"):
                        file_path = file_path[len("blob/") :]

                    if file_path.startswith("main/"):
                        file_path = file_path[len("main/") :]

                    checkpoint_path = hf_hub_download(
                        repo_id,
                        filename=file_path,
                        cache_dir=cache_dir,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=use_auth_token,
                        revision=revision,
                        force_download=force_download,
                    )
            else:
                checkpoint_path = ppdiffusers_url_download(
                    pretrained_model_link_or_path,
                    cache_dir=cache_dir,
                    filename=http_file_name(pretrained_model_link_or_path).strip('"'),
                    force_download=force_download,
                    resume_download=resume_download,
                )
        else:
            checkpoint_path = pretrained_model_link_or_path

        if config_file is None:
            config_url = "https://raw.githubusercontent.com/lllyasviel/ControlNet/main/models/cldm_v15.yaml"
            config_file = BytesIO(requests.get(config_url).content)

        image_size = image_size or 512

        controlnet = download_controlnet_from_original_ckpt(
            checkpoint_path,
            original_config_file=config_file,
            image_size=image_size,
            extract_ema=extract_ema,
            num_in_channels=num_in_channels,
            upcast_attention=upcast_attention,
            # from_safetensors=from_safetensors,
            use_linear_projection=use_linear_projection,
        )

        if paddle_dtype is not None:
            controlnet.to(paddle_dtype=paddle_dtype)

        return controlnet


def http_file_name(
    url: str,
    *,
    proxies=None,
    headers: Optional[Dict[str, str]] = None,
    timeout=10.0,
    max_retries=0,
):
    """
    Get a remote file name.
    """
    headers = copy.deepcopy(headers) or {}
    r = _request_wrapper(
        method="GET",
        url=url,
        stream=True,
        proxies=proxies,
        headers=headers,
        timeout=timeout,
        max_retries=max_retries,
    )
    hf_raise_for_status(r)
    displayed_name = url.split("/")[-1]
    content_disposition = r.headers.get("Content-Disposition")
    if content_disposition is not None and "filename=" in content_disposition:
        # Means file is on CDN
        displayed_name = content_disposition.split("filename=")[-1]
    return displayed_name
