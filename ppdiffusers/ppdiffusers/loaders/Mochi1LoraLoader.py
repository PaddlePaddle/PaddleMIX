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



class Mochi1LoraLoaderMixin(LoraLoaderMixin):
    _lora_loadable_modules = ["transformer"]
    transformer_name = TRANSFORMER_NAME

    @classmethod
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, paddle.Tensor]],
        **kwargs,
    ):
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
            allow_pickle = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "paddle",
        }

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
        self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, paddle.Tensor]], adapter_name=None, **kwargs
    ):
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
        cls, state_dict, transformer, adapter_name=None, _pipeline=None, low_cpu_mem_usage=False
    ):
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
        save_function: Callable = None,
        safe_serialization: bool = True,
    ):
        state_dict = {}

        if not transformer_lora_layers:
            raise ValueError("You must pass `transformer_lora_layers`.")

        if transformer_lora_layers:
            state_dict.update(cls.pack_weights(transformer_lora_layers, cls.transformer_name))

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
        super().fuse_lora(
            components=components,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
            adapter_names=adapter_names,
            **kwargs,
        )

    def unfuse_lora(self, components: List[str] = ["transformer"], **kwargs):
        super().unfuse_lora(components=components, **kwargs)