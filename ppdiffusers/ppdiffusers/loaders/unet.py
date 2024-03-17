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
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from typing import Callable, Dict, List, Optional, Union

import paddle

from ..models.embeddings import ImageProjection
from ..models.modeling_pytorch_paddle_utils import (
    convert_paddle_state_dict_to_pytorch,
    convert_pytorch_state_dict_to_paddle,
)
from ..models.modeling_utils import faster_set_state_dict, load_state_dict
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
    delete_adapter_layers,
    is_paddle_version,
    is_ppxformers_available,
    is_safetensors_available,
    is_torch_available,
    logging,
    set_adapter_layers,
    set_weights_and_activate_adapters,
)
from .utils import AttnProcsLayers

if is_safetensors_available():
    from safetensors.numpy import save_file as np_safe_save_file

    if is_torch_available():
        from safetensors.torch import save_file as torch_safe_save_file

logger = logging.get_logger(__name__)


if is_torch_available():
    import torch
TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"

TORCH_LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
TORCH_LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"

PADDLE_LORA_WEIGHT_NAME = "paddle_lora_weights.pdparams"
PADDLE_LORA_WEIGHT_NAME_SAFE = "paddle_lora_weights.safetensors"

TORCH_CUSTOM_DIFFUSION_WEIGHT_NAME = "pytorch_custom_diffusion_weights.bin"
TORCH_CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE = "pytorch_custom_diffusion_weights.safetensors"

PADDLE_CUSTOM_DIFFUSION_WEIGHT_NAME = "paddle_custom_diffusion_weights.pdparams"
PADDLE_CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE = "paddle_custom_diffusion_weights.safetensors"


class UNet2DConditionLoadersMixin:
    """
    Load LoRA layers into a [`UNet2DCondtionModel`].
    """

    text_encoder_name = TEXT_ENCODER_NAME
    unet_name = UNET_NAME

    def load_attn_procs(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, paddle.Tensor]], **kwargs):
        r"""
        Load pretrained attention processor layers into [`UNet2DConditionModel`]. Attention processor layers have to be
        defined in
        [`attention_processor.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
        and be a `nn.Layer` class.

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
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.

        Example:

        ```py
        from ppdiffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", paddle_dtype=paddle.float16
        )
        pipeline.unet.load_attn_procs(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        ```
        """
        from ..models.attention_processor import CustomDiffusionAttnProcessor
        from ..models.lora import (
            LoRACompatibleConv,
            LoRACompatibleLinear,
            LoRAConv2dLayer,
            LoRALinearLayer,
        )

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
        if subfolder is None:
            subfolder = ""
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", LOW_CPU_MEM_USAGE_DEFAULT)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        network_alphas = kwargs.pop("network_alphas", None)

        _pipeline = kwargs.pop("_pipeline", None)  # noqa: F841

        is_network_alphas_none = network_alphas is None

        if use_safetensors is None:
            use_safetensors = True

        if weight_name is not None:
            if "paddle" in weight_name.lower() or ".pdparams" in weight_name.lower():
                if from_diffusers:
                    logger.warning(
                        "Detect the weight is in ppdiffusers format, but currently, `from_diffusers` is set to `True`. To proceed, we will change the value of `from_diffusers` to `False`!"
                    )
                    from_diffusers = False
            elif "torch" in weight_name.lower() or ".bin" in weight_name.lower() or ".pt" in weight_name.lower():
                if not from_diffusers:
                    logger.warning(
                        "Detect the weight is in diffusers format, but currently, `from_diffusers` is set to `False`. To proceed, we will change the value of `from_diffusers` to `True`!"
                    )
                    from_diffusers = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch" if from_diffusers else "paddle",
        }

        if low_cpu_mem_usage and (not is_paddle_version(">=", "2.5.0") and not is_paddle_version("==", "0.0.0")):
            raise NotImplementedError(
                "Low memory initialization requires paddlepaddle-gpu >= 2.5.0. Please either update your PaddlePaddle version or set"
                " `low_cpu_mem_usage=False`."
            )

        model_file = None
        state_dict = {}
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
            data_format = load_state_dict(model_file, state_dict)
            if not from_diffusers and data_format == "pt":
                logger.warning(
                    "Detect the weight is in diffusers format, but currently, `from_diffusers` is set to `False`. To proceed, we will change the value of `from_diffusers` to `True`!"
                )
                from_diffusers = True
            if from_diffusers and data_format in ["pd", "np"]:
                logger.warning(
                    "Detect the weight is in ppdiffusers format, but currently, `from_diffusers` is set to `True`. To proceed, we will change the value of `from_diffusers` to `False`!"
                )
                from_diffusers = False
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        # fill attn processors
        lora_layers_list = []

        is_lora = all(("lora" in k or k.endswith(".alpha")) for k in state_dict.keys()) and not USE_PEFT_BACKEND
        is_custom_diffusion = any("custom_diffusion" in k for k in state_dict.keys())

        if is_lora:
            # correct keys
            state_dict, network_alphas = self.convert_state_dict_legacy_attn_format(state_dict, network_alphas)

            if network_alphas is not None:
                network_alphas_keys = list(network_alphas.keys())
                used_network_alphas_keys = set()

            lora_grouped_dict = defaultdict(dict)
            mapped_network_alphas = {}

            all_keys = list(state_dict.keys())
            for key in all_keys:
                value = state_dict.pop(key)
                attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                lora_grouped_dict[attn_processor_key][sub_key] = value

                # Create another `mapped_network_alphas` dictionary so that we can properly map them.
                if network_alphas is not None:
                    for k in network_alphas_keys:
                        if k.replace(".alpha", "") in key:
                            mapped_network_alphas.update({attn_processor_key: network_alphas.get(k)})
                            used_network_alphas_keys.add(k)

            if not is_network_alphas_none:
                if len(set(network_alphas_keys) - used_network_alphas_keys) > 0:
                    raise ValueError(
                        f"The `network_alphas` has to be empty at this point but has the following keys \n\n {', '.join(network_alphas.keys())}"
                    )

            if len(state_dict) > 0:
                raise ValueError(
                    f"The `state_dict` has to be empty at this point but has the following keys \n\n {', '.join(state_dict.keys())}"
                )

            for key, value_dict in lora_grouped_dict.items():
                attn_processor = self
                for sub_key in key.split("."):
                    attn_processor = getattr(attn_processor, sub_key)

                # Process non-attention layers, which don't have to_{k,v,q,out_proj}_lora layers
                # or add_{k,v,q,out_proj}_proj_lora layers.
                if from_diffusers:
                    rank = value_dict["lora.down.weight"].shape[0]
                else:
                    rank = value_dict["lora.down.weight"].shape[1]

                if isinstance(attn_processor, LoRACompatibleConv):
                    in_features = attn_processor._in_channels
                    out_features = attn_processor._out_channels
                    kernel_size = attn_processor._kernel_size

                    ctx = paddle.LazyGuard if low_cpu_mem_usage else nullcontext
                    with ctx():
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
                    ctx = paddle.LazyGuard if low_cpu_mem_usage else nullcontext
                    with ctx():
                        lora = LoRALinearLayer(
                            attn_processor.in_features,
                            attn_processor.out_features,
                            rank,
                            mapped_network_alphas.get(key),
                        )
                else:
                    raise ValueError(f"Module {key} is not a LoRACompatibleConv or LoRACompatibleLinear module.")

                value_dict = {k.replace("lora.", ""): v for k, v in value_dict.items()}
                lora_layers_list.append((attn_processor, lora))
                if from_diffusers:
                    convert_pytorch_state_dict_to_paddle(lora, value_dict)
                faster_set_state_dict(lora, value_dict)

        elif is_custom_diffusion:
            attn_processors = {}
            custom_diffusion_grouped_dict = defaultdict(dict)
            for key, value in state_dict.items():
                if len(value) == 0:
                    custom_diffusion_grouped_dict[key] = {}
                else:
                    if "to_out" in key:
                        attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                    else:
                        attn_processor_key, sub_key = ".".join(key.split(".")[:-2]), ".".join(key.split(".")[-2:])
                    custom_diffusion_grouped_dict[attn_processor_key][sub_key] = value

            for key, value_dict in custom_diffusion_grouped_dict.items():
                if len(value_dict) == 0:
                    attn_processors[key] = CustomDiffusionAttnProcessor(
                        train_kv=False, train_q_out=False, hidden_size=None, cross_attention_dim=None
                    )
                else:
                    if from_diffusers:
                        cross_attention_dim = value_dict["to_k_custom_diffusion.weight"].shape[1]
                        hidden_size = value_dict["to_k_custom_diffusion.weight"].shape[0]
                    else:
                        cross_attention_dim = value_dict["to_k_custom_diffusion.weight"].shape[0]
                        hidden_size = value_dict["to_k_custom_diffusion.weight"].shape[1]
                    train_q_out = True if "to_q_custom_diffusion.weight" in value_dict else False
                    attn_processors[key] = CustomDiffusionAttnProcessor(
                        train_kv=True,
                        train_q_out=train_q_out,
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                    )
                    if from_diffusers:
                        convert_pytorch_state_dict_to_paddle(attn_processors[key], value_dict)
                    faster_set_state_dict(attn_processors[key], value_dict)
        elif USE_PEFT_BACKEND:
            # In that case we have nothing to do as loading the adapter weights is already handled above by `set_peft_model_state_dict`
            # on the Unet
            pass
        else:
            raise ValueError(
                f"{model_file} does not seem to be in the correct format expected by LoRA or Custom Diffusion training."
            )

        # For PEFT backend the Unet is already offloaded at this stage as it is handled inside `lora_lora_weights_into_unet`
        if not USE_PEFT_BACKEND:

            # only custom diffusion needs to set attn processors
            if is_custom_diffusion:
                self.set_attn_processor(attn_processors)

            # set lora layers
            for target_module, lora_layer in lora_layers_list:
                target_module.set_lora_layer(lora_layer)

            self.to(dtype=self.dtype)

    def convert_state_dict_legacy_attn_format(self, state_dict, network_alphas):
        is_new_lora_format = all(
            key.startswith(self.unet_name) or key.startswith(self.text_encoder_name) for key in state_dict.keys()
        )
        if is_new_lora_format:
            # Strip the `"unet"` prefix.
            is_text_encoder_present = any(key.startswith(self.text_encoder_name) for key in state_dict.keys())
            if is_text_encoder_present:
                warn_message = "The state_dict contains LoRA params corresponding to the text encoder which are not being used here. To use both UNet and text encoder related LoRA params, use [`pipe.load_lora_weights()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.load_lora_weights)."
                logger.warn(warn_message)
            unet_keys = [k for k in state_dict.keys() if k.startswith(self.unet_name)]
            state_dict = {k.replace(f"{self.unet_name}.", ""): v for k, v in state_dict.items() if k in unet_keys}

        # change processor format to 'pure' LoRACompatibleLinear format
        if any("processor" in k.split(".") for k in state_dict.keys()):

            def format_to_lora_compatible(key):
                if "processor" not in key.split("."):
                    return key
                return key.replace(".processor", "").replace("to_out_lora", "to_out.0.lora").replace("_lora", ".lora")

            state_dict = {format_to_lora_compatible(k): v for k, v in state_dict.items()}

            if network_alphas is not None:
                network_alphas = {format_to_lora_compatible(k): v for k, v in network_alphas.items()}
        return state_dict, network_alphas

    def save_attn_procs(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        to_diffusers: Optional[bool] = None,
        **kwargs,
    ):
        r"""
        Save attention processor layers to a directory so that it can be reloaded with the
        [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`] method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save an attention processor to (will be created if it doesn't exist).
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or with `pickle`.

        Example:

        ```py
        import paddle
        from ppdiffusers import DiffusionPipeline

        pipeline = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            paddle_dtype=paddle.float16,
        )
        pipeline.unet.load_attn_procs("path-to-save-model", weight_name="paddle_custom_diffusion_weights.pdparams")
        pipeline.unet.save_attn_procs("path-to-save-model", weight_name="paddle_custom_diffusion_weights.pdparams")
        ```
        """
        from ..models.attention_processor import (
            CustomDiffusionAttnProcessor,
            CustomDiffusionAttnProcessor2_5,
            CustomDiffusionXFormersAttnProcessor,
        )

        if to_diffusers is None:
            to_diffusers = TO_DIFFUSERS
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        is_custom_diffusion = any(
            isinstance(
                x,
                (CustomDiffusionAttnProcessor, CustomDiffusionAttnProcessor2_5, CustomDiffusionXFormersAttnProcessor),
            )
            for (_, x) in self.attn_processors.items()
        )
        if is_custom_diffusion:
            model_to_save = AttnProcsLayers(
                {
                    y: x
                    for (y, x) in self.attn_processors.items()
                    if isinstance(
                        x,
                        (
                            CustomDiffusionAttnProcessor,
                            CustomDiffusionAttnProcessor2_5,
                            CustomDiffusionXFormersAttnProcessor,
                        ),
                    )
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
                if safe_serialization:
                    weight_name = (
                        PADDLE_CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE
                        if is_custom_diffusion
                        else PADDLE_LORA_WEIGHT_NAME_SAFE
                    )
                else:
                    weight_name = (
                        PADDLE_CUSTOM_DIFFUSION_WEIGHT_NAME if is_custom_diffusion else PADDLE_LORA_WEIGHT_NAME
                    )
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

                convert_paddle_state_dict_to_pytorch(self, state_dict)
            else:
                if safe_serialization:
                    for k, v in state_dict.items():
                        if isinstance(v, paddle.Tensor):
                            state_dict[k] = v.cpu().numpy()

                    save_function = partial(np_safe_save_file, metadata={"format": "pd"})
                else:
                    save_function = paddle.save

        # Save the model
        save_function(state_dict, os.path.join(save_directory, weight_name))
        logger.info(f"Model weights saved in {os.path.join(save_directory, weight_name)}")

    def fuse_lora(self, lora_scale=1.0, safe_fusing=False):
        self.lora_scale = lora_scale
        self._safe_fusing = safe_fusing
        self.apply(self._fuse_lora_apply)

    def _fuse_lora_apply(self, module):
        if not USE_PEFT_BACKEND:
            if hasattr(module, "_fuse_lora"):
                module._fuse_lora(self.lora_scale, self._safe_fusing)
        else:
            from ppdiffusers.peft.tuners.tuners_utils import BaseTunerLayer

            if isinstance(module, BaseTunerLayer):
                if self.lora_scale != 1.0:
                    module.scale_layer(self.lora_scale)
                module.merge(safe_merge=self._safe_fusing)

    def unfuse_lora(self):
        self.apply(self._unfuse_lora_apply)

    def _unfuse_lora_apply(self, module):
        if not USE_PEFT_BACKEND:
            if hasattr(module, "_unfuse_lora"):
                module._unfuse_lora()
        else:
            from ppdiffusers.peft.tuners.tuners_utils import BaseTunerLayer

            if isinstance(module, BaseTunerLayer):
                module.unmerge()

    def set_adapters(
        self,
        adapter_names: Union[List[str], str],
        weights: Optional[Union[List[float], float]] = None,
    ):
        """
        Set the currently active adapters for use in the UNet.

        Args:
            adapter_names (`List[str]` or `str`):
                The names of the adapters to use.
            adapter_weights (`Union[List[float], float]`, *optional*):
                The adapter(s) weights to use with the UNet. If `None`, the weights are set to `1.0` for all the
                adapters.

        Example:

        ```py
        from ppdiffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", paddle_dtype=paddle.float16
        )
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipeline.set_adapters(["cinematic", "pixel"], adapter_weights=[0.5, 0.5])
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `set_adapters()`.")

        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

        if weights is None:
            weights = [1.0] * len(adapter_names)
        elif isinstance(weights, float):
            weights = [weights] * len(adapter_names)

        if len(adapter_names) != len(weights):
            raise ValueError(
                f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}."
            )

        set_weights_and_activate_adapters(self, adapter_names, weights)

    def disable_lora(self):
        """
        Disable the UNet's active LoRA layers.

        Example:

        ```py
        from ppdiffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", paddle_dtype=paddle.float16
        )
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.disable_lora()
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        set_adapter_layers(self, enabled=False)

    def enable_lora(self):
        """
        Enable the UNet's active LoRA layers.

        Example:

        ```py
        from ppdiffusers import AutoPipelineForText2Image
        import paddle

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", paddle_dtype=paddle.float16
        )
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.enable_lora()
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        set_adapter_layers(self, enabled=True)

    def delete_adapters(self, adapter_names: Union[List[str], str]):
        """
        Delete an adapter's LoRA layers from the UNet.

        Args:
            adapter_names (`Union[List[str], str]`):
                The names (single string or list of strings) of the adapter to delete.

        Example:

        ```py
        from ppdiffusers import AutoPipelineForText2Image
        import paddle

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", paddle_dtype=paddle.float16
        )
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_names="cinematic"
        )
        pipeline.delete_adapters("cinematic")
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        for adapter_name in adapter_names:
            delete_adapter_layers(self, adapter_name)

            # Pop also the corresponding adapter from the config
            if hasattr(self, "peft_config"):
                self.peft_config.pop(adapter_name, None)

    def _load_ip_adapter_weights(self, state_dict, from_diffusers=None):
        if from_diffusers is None:
            from_diffusers = FROM_DIFFUSERS

        str_dtype = str(self.dtype).replace("paddle.", "")
        from ..models.attention_processor import (
            AttnProcessor,
            AttnProcessor2_5,
            IPAdapterAttnProcessor,
            IPAdapterAttnProcessor2_5,
        )

        # set ip-adapter cross-attention processors & load state_dict
        attn_procs = {}
        key_id = 1
        for name in self.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.config.block_out_channels[block_id]
            if cross_attention_dim is None or "motion_modules" in name:
                attn_processor_class = AttnProcessor2_5 if is_ppxformers_available() else AttnProcessor
                attn_procs[name] = attn_processor_class()
            else:
                attn_processor_class = (
                    IPAdapterAttnProcessor2_5 if is_ppxformers_available() else IPAdapterAttnProcessor
                )
                attn_procs[name] = attn_processor_class(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0
                ).to(dtype=self.dtype)

                value_dict = {}
                for k, w in attn_procs[name].state_dict().items():
                    value_dict.update({f"{k}": state_dict["ip_adapter"][f"{key_id}.{k}"].astype(str_dtype)})

                if from_diffusers:
                    convert_pytorch_state_dict_to_paddle(attn_procs[name], value_dict)
                attn_procs[name].load_dict(value_dict)
                key_id += 2

        self.set_attn_processor(attn_procs)

        # create image projection layers.
        if from_diffusers:
            clip_embeddings_dim = state_dict["image_proj"]["proj.weight"].shape[-1]
            cross_attention_dim = state_dict["image_proj"]["proj.weight"].shape[0] // 4
        else:
            clip_embeddings_dim = state_dict["image_proj"]["proj.weight"].shape[0]
            cross_attention_dim = state_dict["image_proj"]["proj.weight"].shape[-1] // 4
        image_projection = ImageProjection(
            cross_attention_dim=cross_attention_dim, image_embed_dim=clip_embeddings_dim, num_image_text_embeds=4
        )
        image_projection.to(dtype=self.dtype)

        # load image projection layer weights
        image_proj_state_dict = {}
        image_proj_state_dict.update(
            {
                "image_embeds.weight": state_dict["image_proj"]["proj.weight"].astype(str_dtype),
                "image_embeds.bias": state_dict["image_proj"]["proj.bias"].astype(str_dtype),
                "norm.weight": state_dict["image_proj"]["norm.weight"].astype(str_dtype),
                "norm.bias": state_dict["image_proj"]["norm.bias"].astype(str_dtype),
            }
        )
        if from_diffusers:
            convert_pytorch_state_dict_to_paddle(image_projection, image_proj_state_dict)
        image_projection.load_dict(image_proj_state_dict)

        self.encoder_hid_proj = image_projection.to(dtype=self.dtype)
        self.config.encoder_hid_dim_type = "ip_image_proj"

    delete_adapter_layers
