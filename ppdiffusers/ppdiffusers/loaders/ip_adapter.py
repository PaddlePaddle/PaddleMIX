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
from typing import Dict, Union

import paddle
from safetensors import safe_open

from ..utils import (
    DIFFUSERS_CACHE,
    FROM_AISTUDIO,
    FROM_DIFFUSERS,
    FROM_HF_HUB,
    HF_HUB_OFFLINE,
    PPDIFFUSERS_CACHE,
    _get_model_file,
    is_paddlenlp_available,
    logging,
    smart_load,
)

if is_paddlenlp_available():
    from ppdiffusers.transformers import (
        CLIPImageProcessor,
        CLIPVisionModelWithProjection,
    )

    from ..models.attention_processor import (
        IPAdapterAttnProcessor,
        IPAdapterAttnProcessor2_5,
    )

logger = logging.get_logger(__name__)


class IPAdapterMixin:
    """Mixin for handling IP Adapters."""

    def load_ip_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, paddle.Tensor]],
        subfolder: str = "",
        weight_name: str = None,
        **kwargs,
    ):
        """
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
        """

        # Load the main state dict first.
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
        if subfolder is None:
            subfolder = ""
        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch" if from_diffusers else "paddle",
        }
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
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
                from_aistudio=from_aistudio,
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {"image_proj": {}, "ip_adapter": {}}
                with safe_open(model_file, framework="np") as f:
                    metadata = f.metadata()
                    if metadata is None:
                        metadata = {}
                    if metadata.get("format", "pt") not in ["pt", "pd", "np"]:
                        raise OSError(
                            f"The safetensors archive passed at {model_file} does not contain the valid metadata. Make sure "
                            "you save your model with the `save_pretrained` method."
                        )
                    data_format = metadata.get("format", "pt")
                    if data_format == "pt" and not from_diffusers:
                        logger.warning(
                            "Detect the weight is in diffusers format, but currently, `from_diffusers` is set to `False`. To proceed, we will change the value of `from_diffusers` to `True`!"
                        )
                        from_diffusers = True
                    for key in f.keys():
                        if key.startswith("image_proj."):
                            state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                        elif key.startswith("ip_adapter."):
                            state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
            else:
                state_dict = smart_load(model_file, return_numpy=True, return_is_torch_weight=True)
                is_torch_weight = state_dict.pop("is_torch_weight", False)
                if not from_diffusers and is_torch_weight:
                    logger.warning(
                        "Detect the weight is in diffusers format, but currently, `from_diffusers` is set to `False`. To proceed, we will change the value of `from_diffusers` to `True`!"
                    )
                    from_diffusers = True
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        keys = list(state_dict.keys())
        if sorted(keys) != ["image_proj", "ip_adapter"]:
            raise ValueError("Required keys are (`image_proj` and `ip_adapter`) missing from the state dict.")

        # load CLIP image encoer here if it has not been registered to the pipeline yet
        if hasattr(self, "image_encoder") and getattr(self, "image_encoder", None) is None:
            if not isinstance(pretrained_model_name_or_path_or_dict, dict):
                logger.info(f"loading image_encoder from {pretrained_model_name_or_path_or_dict}")
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    pretrained_model_name_or_path_or_dict,
                    subfolder=os.path.join(subfolder, "image_encoder"),
                    from_hf_hub=from_hf_hub,
                    from_aistudio=from_aistudio,
                    # from_diffusers=from_diffusers, # we must disable this !
                ).to(dtype=self.dtype)
                self.image_encoder = image_encoder
            else:
                raise ValueError("`image_encoder` cannot be None when using IP Adapters.")

        # create feature extractor if it has not been registered to the pipeline yet
        if hasattr(self, "feature_extractor") and getattr(self, "feature_extractor", None) is None:
            self.feature_extractor = CLIPImageProcessor()

        # load ip-adapter into unet
        self.unet._load_ip_adapter_weights(state_dict, from_diffusers=from_diffusers)

    def set_ip_adapter_scale(self, scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_5)):
                attn_processor.scale = scale
