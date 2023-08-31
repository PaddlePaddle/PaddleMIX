# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The Salesforce Team Authors and The HuggingFace Team. All rights reserved.
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
""" Paddle BLIP2 model."""
import gc
import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Type, Union

import paddle
import paddle.distributed as dist
import paddle.nn as nn
from paddlenlp.transformers import AutoTokenizer, PretrainedModel
from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.transformers.llama.modeling import LlamaForCausalLM
from paddlenlp.transformers.model_outputs import ModelOutput
from paddlenlp.transformers.model_utils import _add_variant, weight_name_suffix
from paddlenlp.transformers.opt.modeling import OPTForCausalLM
from paddlenlp.transformers.t5.modeling import T5ForConditionalGeneration
from paddlenlp.utils.env import (
    PADDLE_WEIGHTS_INDEX_NAME,
    PADDLE_WEIGHTS_NAME,
    PYTORCH_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
)
from tqdm.auto import tqdm

from paddlemix.examples.blip2.utils import blip2_load
from paddlemix.models.blip2.modeling_utils import (
    all_gather_with_grad,
    concat_all_gather,
    disabled_train,
    masked_fill,
)
from paddlemix.models.blip2.Qformer import BertLMHeadModel
from paddlemix.models.model_utils import MixPretrainedModel
from paddlemix.utils.log import logger

from .configuration import Blip2Config

VISION_WEIGHTS = {
    "eva_vit_g": "https://bj.bcebos.com/paddlenlp/models/community/paddlemix/blip2-stage2/eva_vit_g/model_state.pdparams"
}
BRIDGE_WEIGHTS = {
    "qformer-stage2": "https://bj.bcebos.com/paddlenlp/models/community/paddlemix/blip2-stage2/Qformer/model_state.pdparams",
    "qformer-stage1": "https://bj.bcebos.com/paddlenlp/models/community/paddlemix/blip2-stage1/Qformer/model_state.pdparams",
}
from paddle.utils.download import is_url as is_remote_url
from paddlenlp.transformers.utils import cached_file, resolve_cache_dir
from paddlenlp.utils.downloader import get_path_from_url_with_filelock

__all__ = [
    "Blip2ForConditionalGeneration",
]


def Parameter(tensor):
    return paddle.create_parameter(
        tensor.shape,
        dtype=tensor.dtype,
        default_initializer=nn.initializer.Assign(tensor),
    )


@dataclass
class Blip2ForConditionalGenerationModelOutput(ModelOutput):
    """
    Class defining the outputs of [`Blip2ForConditionalGeneration`].
    Args:
        loss (`paddle.Tensor`, *optional*, returned when `labels` is provided, `paddle.Tensor` of shape `(1,)`):
            Language modeling loss from the language model.
        logits (`paddle.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head of the language model.
        vision_outputs (`BaseModelOutputWithPooling`):
            Outputs of the vision encoder.
        qformer_outputs (`BaseModelOutputWithPoolingAndCrossAttentions`):
            Outputs of the Q-Former (Querying Transformer).
        language_model_outputs (`CausalLMOutputWithPast` or `Seq2SeqLMOutput`):
            Outputs of the language model.
    """

    loss: Optional[Tuple[paddle.Tensor]] = None
    logits: Optional[Tuple[paddle.Tensor]] = None
    vision_outputs: Optional[paddle.Tensor] = None
    qformer_outputs: Optional[Tuple[paddle.Tensor]] = None
    language_model_outputs: Optional[Tuple[paddle.Tensor]] = None

    def to_tuple(self) -> Tuple[Any]:
        return tuple(
            self[k]
            if k not in ["vision_outputs", "qformer_outputs", "language_model_outputs"]
            else getattr(self, k).to_tuple()
            for k in self.keys()
        )


@dataclass
class Blip2ForStage1ModelOutput(Blip2ForConditionalGenerationModelOutput):
    """
    Class defining the outputs of [`Blip2ForStage1ModelOutput`].
    """

    loss: Optional[Tuple[paddle.Tensor]] = None
    loss_itc: Optional[Tuple[paddle.Tensor]] = None
    loss_itm: Optional[paddle.Tensor] = None
    loss_lm: Optional[Tuple[paddle.Tensor]] = None


class Blip2PretrainedModel(MixPretrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Blip2Config

    def _init_weights(self, module):
        """Initialize the weights"""
        pass

    @classmethod
    def init_tokenizer(cls, tokenizer_name="bert-base-uncased"):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        return tokenizer

    @classmethod
    def refine_state_dict(self, model, state_dict):
        from paddlemix.models.blip2.eva_vit import interpolate_pos_embed

        interpolate_pos_embed(model, state_dict)

    def get_expected_keys(self, model_state_dict, name=None):
        model_list = []
        if name == "Qformer":
            self._keys_to_ignore_on_load_unexpected = ["visual_encoder", "language_model"]
            for key in model_state_dict.keys():
                if "visual_encoder" not in key and "language_model" not in key:
                    model_list.append(key)

        elif name == "visual_encoder":
            self._keys_to_ignore_on_load_unexpected = ["Qformer", "language_model"]
            for key in model_state_dict.keys():
                if "visual_encoder" in key:
                    model_list.append(key)
        else:
            self._keys_to_ignore_on_load_unexpected = ["language_model"]
            for key in model_state_dict.keys():
                if "language_model" not in key:
                    model_list.append(key)

        return model_list

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, from_hf_hub: bool = False, subfolder: str = None, *args, **kwargs
    ):
        """
        Creates an instance of `PretrainedModel`. Model weights are loaded
        by specifying name of a built-in pretrained model, a pretrained model from HF Hub, a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model or dir path
                to load from. The string can be:

                - Name of a built-in pretrained model
                - Name of a pretrained model from HF Hub
                - Name of a community-contributed pretrained model.
                - Local directory path which contains model weights file("model_state.pdparams")
                  and model config file ("model_config.json").
            from_hf_hub (bool): load model from huggingface hub. Default to `False`.
            subfolder (str, optional) An optional value corresponding to a folder inside the repo.
                Only works when loading from Huggingface Hub.
            *args (tuple): Position arguments for model `__init__`. If provided,
                use these as position argument values for model initialization.
            **kwargs (dict): Keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for model
                initialization. If the keyword is in `__init__` argument names of
                base model, update argument values of the base model; else update
                argument values of derived model.
            load_state_as_np (bool, optional): The weights read in can be choosed
                to place on CPU or GPU though the model is on the default device.
                If `True`, load the model weights as `numpy.ndarray` on CPU.
                Otherwise, weights would be loaded as tensors on the default
                device. Note that if on GPU, the latter would creates extra
                temporary tensors in addition to the model weights, which
                doubles the memory usage . Thus it is suggested to use `True`
                for big models on GPU. Default to `False`.

        Returns:
            PretrainedModel: An instance of `PretrainedModel`.

        Example:
            .. code-block::

                from paddlenlp.transformers import BertForSequenceClassification

                # Name of built-in pretrained model
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

                # Name of pretrained model from PaddleHub
                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

                # Name of community-contributed pretrained model
                model = BertForSequenceClassification.from_pretrained('yingyibiao/bert-base-uncased-sst-2-finetuned', num_labels=3)

                # Load from local directory path
                model = BertForSequenceClassification.from_pretrained('./my_bert/'
        """
        import os

        from paddlenlp.transformers.configuration_utils import PretrainedConfig
        from paddlenlp.transformers.model_utils import load_state_dict, no_init_weights
        from paddlenlp.transformers.utils import (
            ContextManagers,
            device_guard,
            is_paddle_support_lazy_init,
            is_safetensors_available,
            resolve_cache_dir,
        )
        from paddlenlp.utils.env import (
            CONFIG_NAME,
            PADDLE_WEIGHTS_NAME,
            PYTORCH_WEIGHTS_NAME,
        )

        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        dtype = kwargs.pop("dtype", None)
        subfolder = kwargs.pop("subfolder", "")
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None if is_safetensors_available() else False)

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
        convert_from_torch = kwargs.pop("convert_from_torch", None)
        load_state_as_np = kwargs.pop("load_state_as_np", None)
        mp_degree = kwargs.pop("mp_degree", 1)
        if load_state_as_np is not None:
            logger.warning("`load_state_as_np` is deprecated,  please delete it!")

        model_kwargs = kwargs

        # from_hf_hub defalut enable convert_from_torch
        if from_hf_hub and convert_from_torch is None:
            logger.warning(
                "If you are attempting to load weights from Hugging Face Hub and want to disable the default behavior of considering torch weights,"
                " you can set ·convert_from_torch=False·. By default, `convert_from_torch` is set to `True`. "
            )
            convert_from_torch = True
        # convert_from_torch defalut is False
        if convert_from_torch is None:
            convert_from_torch = False

        cache_dir = resolve_cache_dir(pretrained_model_name_or_path, from_hf_hub, cache_dir)
        # 1. get the PretrainedConfig to init model
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                from_hf_hub=from_hf_hub,
                subfolder=subfolder,
                **kwargs,
            )
        if not os.path.exists(os.path.join(cache_dir, CONFIG_NAME)):
            config.save_pretrained(cache_dir)

        # refine options for config
        config.mp_degree = mp_degree
        convert_from_torch = cls.support_conversion(config) and convert_from_torch

        if dtype is None:
            dtype = config.dtype
        else:
            config.dtype = dtype

        init_contexts = []
        if low_cpu_mem_usage:
            # Instantiate model.
            init_contexts.append(no_init_weights(_enable=True))
            if is_paddle_support_lazy_init():
                init_contexts.append(paddle.LazyGuard())

        if dtype:
            init_contexts.append(dtype_guard(dtype))

        # Keep in fp32 modules
        keep_in_fp32_modules = None
        use_keep_in_fp32_modules = False

        # resolve model_weight file
        resolved_archive_file, sharded_metadata, is_sharded = cls._resolve_model_file_path(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            subfolder=subfolder,
            from_hf_hub=from_hf_hub,
            config=config,
            convert_from_torch=convert_from_torch,
            use_safetensors=use_safetensors,
            variant=variant,
        )

        # load pt weights early so that we know which dtype to init the model under
        if not is_sharded and state_dict is None:
            # Time to load the checkpoint
            if resolved_archive_file.endswith(PYTORCH_WEIGHTS_NAME):
                if convert_from_torch:
                    # try to get the name-mapping info
                    logger.info(
                        f"Starting to convert pytorch weight file<{resolved_archive_file}> to "
                        f"paddle weight file<{os.path.join(cache_dir, PADDLE_WEIGHTS_NAME)}> ..."
                    )
                    state_dict = cls.convert(resolved_archive_file, config, cache_dir)
                else:
                    raise ValueError(
                        f"download the {PYTORCH_WEIGHTS_NAME} weight file, but model<{cls}> "
                        "don't support conversion from pytorch weight file to paddle weight file "
                    )
            else:
                # 4. loading non-sharded ckpt from the state dict
                if config.tensor_parallel_degree > 1 and resolved_archive_file.endswith("model_state.pdparams"):
                    state_dict = cls.convert_tensor_parallel(resolved_archive_file, config)
                else:
                    state_dict = load_state_dict(resolved_archive_file)

                logger.info("Loaded weights file from disk, setting weights to model.")

        # Check if `_keep_in_fp32_modules` is not None
        use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and dtype == "float16"

        if is_sharded:
            loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
        else:
            loaded_state_dict_keys = [k for k in state_dict.keys()]

        if low_cpu_mem_usage:  # or use_keep_in_fp32_modules:
            state_dict = None

        # will only support load paddle.Tensor to model.
        if state_dict is not None:
            for k in list(state_dict.keys()):
                if not isinstance(state_dict[k], paddle.Tensor):
                    with device_guard():
                        state_dict[k] = paddle.Tensor(state_dict.pop(k), zero_copy=True)

        # 3. init the model
        init_args = config["init_args"] or ()
        with ContextManagers(init_contexts):
            model = cls(config, *init_args, **model_kwargs)
        cls.refine_state_dict(model, state_dict)
        if use_keep_in_fp32_modules:
            # low_cpu_mem_usage = True
            keep_in_fp32_modules = model._keep_in_fp32_modules
        else:
            keep_in_fp32_modules = []

        model, missing_keys, unexpected_keys, mismatched_keys = cls._load_pretrained_model(
            model=model,
            state_dict=state_dict,
            loaded_keys=loaded_state_dict_keys,
            resolved_archive_file=resolved_archive_file,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config=config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            low_cpu_mem_usage=low_cpu_mem_usage,
            dtype=dtype,
            keep_in_fp32_modules=keep_in_fp32_modules,
        )

        if paddle.in_dynamic_mode():
            return model

        return model, state_dict

    def load_pretrained(
        self,
        config,
        model,
        training_args,
        state_dict=None,
        ignore_mismatched_sizes=False,
        low_cpu_mem_usage=False,
        dtype=None,
        cache_dir=None,
        subfolder="",
        variant=None,
        *args,
        **kwargs,
    ) -> Tuple[List[str]]:
        """load the state_dict into model, and do the following things:

            * resolve the pretrained model name or path by checking if they exist in the cache and then
            download them.
            * load the pretrained model and refine the state dict if necessary.
            * filter the weight keys and set the state_dict to the model.

        Args:
            pretrained_model_name_or_path (str): the pretrained model name or path.
            state_dict (Dict[str, Tensor]): the model state dict data.
            ignore_mismatched_sizes (bool, optional): whether ignore error when tensor size mismatched. Defaults to False.
            low_cpu_mem_usage (bool, optional): whether use low cpu memory usage for loading pretrained model。 Defautls to False.
            dtype (_type_, optional): the dtype of model state dict. Defaults to None.
            cahce_cache_dir (str, optional): the cache directory for loading pretrained model. Defaults to None.
            sufolder (str, optional): the subfolder of pretrained model name. Defaults "".
            variant (str, optional): the pretrained model variant. Defaults to None.

        Returns:
            Tuple[List[str]]: _description_
        """
        qformer_model_name_or_path = config.get("bridge_name_or_path", None)
        vision_model_name_or_path = config.get("vision_name_or_path", None)
        model_name_or_path = config.get("model_name_or_path", None)
        if (not qformer_model_name_or_path and not vision_model_name_or_path) and model_name_or_path is None:
            ValueError(
                "either model_name_or_path or (bridge_model_name_or_path and vision_model_name_or_path) should be set."
            )

        def load_blip2_model_state(
            model_name_or_path=None,
            model=None,
            training_args=None,
            weight_name=None,
            state_dict=None,
            ignore_mismatched_sizes=False,
            low_cpu_mem_usage=False,
            dtype=None,
            cache_dir=None,
            subfolder="",
            variant=None,
        ):
            cache_dir = resolve_cache_dir(model_name_or_path, cache_dir)

            # Keep in fp32 modules
            keep_in_fp32_modules = None
            use_keep_in_fp32_modules = False

            # resolve model_weight file
            resolved_archive_file, sharded_metadata, is_sharded = self._resolve_model_file_path_mix(
                model_name_or_path, cache_dir=cache_dir, subfolder=subfolder, variant=variant, weight_name=weight_name
            )

            # Check if `_keep_in_fp32_modules` is not None
            use_keep_in_fp32_modules = (self._keep_in_fp32_modules is not None) and dtype == "float16"

            loaded_state_dict_keys = self.state_dict()

            if use_keep_in_fp32_modules:
                # low_cpu_mem_usage = True
                keep_in_fp32_modules = self._keep_in_fp32_modules
            else:
                keep_in_fp32_modules = []

            # load_pretrained_model
            model_state_dict = self.state_dict()

            expected_keys = self.get_expected_keys(model_state_dict, weight_name)

            prefix = self.base_model_prefix

            if len(prefix) > 0:
                has_prefix_module = any(s.startswith(prefix) for s in loaded_state_dict_keys)
                expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
            else:
                has_prefix_module = False
                expects_prefix_module = False

            # key re-naming operations are never done on the keys
            # that are loaded, but always on the keys of the newly initialized model
            remove_prefix_from_model = not has_prefix_module and expects_prefix_module
            add_prefix_to_model = has_prefix_module and not expects_prefix_module

            if remove_prefix_from_model:
                _prefix = f"{prefix}."
                expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(_prefix)]
                expected_keys = [s[len(_prefix) :] if s.startswith(_prefix) else s for s in expected_keys]
            elif add_prefix_to_model:
                expected_keys = [".".join([prefix, s]) for s in expected_keys]

            missing_keys = list(set(expected_keys) - set(loaded_state_dict_keys))
            unexpected_keys = list(set(loaded_state_dict_keys) - set(expected_keys))

            # Some models may have keys that are not in the state by design, removing them before needlessly warning
            # the user.
            if self._keys_to_ignore_on_load_missing is not None:
                for pat in self._keys_to_ignore_on_load_missing:
                    missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

            if self._keys_to_ignore_on_load_unexpected is not None:
                for pat in self._keys_to_ignore_on_load_unexpected:
                    unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

            # Set some modules to fp32 if any
            if keep_in_fp32_modules is not None:
                for name, param in self.named_parameters():
                    if any(module_to_keep_in_fp32 in name for module_to_keep_in_fp32 in keep_in_fp32_modules):
                        param = param.to(dtype=paddle.float32)

            # Make sure we are able to load base models as well as derived models (with heads)
            model_to_load = self
            if len(self.base_model_prefix) > 0 and hasattr(self, self.base_model_prefix) and not has_prefix_module:
                model_to_load = getattr(self, self.base_model_prefix)
                base_model_expected_keys = list(model_to_load.state_dict().keys())
                if any(
                    key in expected_keys_not_prefixed and key not in base_model_expected_keys
                    for key in loaded_state_dict_keys
                ):
                    raise ValueError(
                        "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
                        "properly saved?"
                    )

            def _find_mismatched_keys(
                state_dict,
                model_state_dict,
                loaded_keys,
                add_prefix_to_model,
                remove_prefix_from_model,
                ignore_mismatched_sizes,
            ):
                mismatched_keys = []
                if ignore_mismatched_sizes:
                    for checkpoint_key in loaded_keys:
                        # If the checkpoint is sharded, we may not have the key here.
                        if checkpoint_key not in state_dict:
                            continue
                        model_key = checkpoint_key
                        if remove_prefix_from_model:
                            # The model key starts with `prefix` but `checkpoint_key` doesn't so we add it.
                            model_key = f"{prefix}.{checkpoint_key}"
                        elif add_prefix_to_model:
                            # The model key doesn't start with `prefix` but `checkpoint_key` does so we remove it.
                            model_key = ".".join(checkpoint_key.split(".")[1:])

                        if (
                            model_key in model_state_dict
                            and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                        ):
                            mismatched_keys.append(
                                (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                            )
                            del state_dict[checkpoint_key]
                return mismatched_keys

            if state_dict is not None:
                # DONT Hold tensor parallel here, only hold afer load state dict.
                # Whole checkpoint
                # For model parallel if FastGeneration
                # To avoid recursive import temporarily.
                import paddlenlp.ops.fast_transformer.transformer.decoding as ft_decoding

                state_dict = ft_decoding.get_ft_para_conf().fit_partial_model(model_to_load, state_dict)

                mismatched_keys = _find_mismatched_keys(
                    state_dict,
                    model_state_dict,
                    loaded_state_dict_keys,
                    add_prefix_to_model,
                    remove_prefix_from_model,
                    ignore_mismatched_sizes,
                )
            else:
                # Sharded checkpoint or whole but low_cpu_mem_usage==True

                # This should always be a list but, just to be sure.
                if not isinstance(resolved_archive_file, list):
                    resolved_archive_file = [resolved_archive_file]

                mismatched_keys = []

                if len(resolved_archive_file) > 1:
                    resolved_archive_file = tqdm(resolved_archive_file, desc="Loading checkpoint shards")

                for shard_file in resolved_archive_file:
                    state_dict = blip2_load(
                        shard_file, model, training_args, map_location="cpu", weight_name=weight_name
                    )

                    # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
                    # matching the weights in the model.
                    mismatched_keys += _find_mismatched_keys(
                        state_dict,
                        model_state_dict,
                        loaded_state_dict_keys,
                        add_prefix_to_model,
                        remove_prefix_from_model,
                        ignore_mismatched_sizes,
                    )

                    # force memory release
                    del state_dict
                    gc.collect()

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint at {model_name_or_path} were not used when"
                    f" initializing {self.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                    f" initializing {self.__class__.__name__} from the checkpoint of a model trained on another task or"
                    " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                    " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                    f" {self.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                    " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
                )
            else:
                logger.info(f"All model checkpoint weights were used when initializing {self.__class__.__name__}.\n")

            if len(missing_keys) > 0:
                logger.warning(
                    f"Some weights of {self.__class__.__name__} were not initialized from the model checkpoint at"
                    f" {model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                    " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
                )
            elif len(mismatched_keys) == 0:
                logger.info(
                    f"All the weights of {self.__class__.__name__} were initialized from the model checkpoint at"
                    f" {model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                    f" was trained on, you can already use {self.__class__.__name__} for predictions without further"
                    " training."
                )
            if len(mismatched_keys) > 0:
                mismatched_warning = "\n".join(
                    [
                        f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                        for key, shape1, shape2 in mismatched_keys
                    ]
                )
                logger.warning(
                    f"Some weights of {self.__class__.__name__} were not initialized from the model checkpoint at"
                    f" {model_name_or_path} and are newly initialized because the shapes did not"
                    f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                    " to use it for predictions and inference."
                )

            return missing_keys, unexpected_keys, mismatched_keys

        if vision_model_name_or_path is not None:
            logger.info("loading a vision model from path{}".format(vision_model_name_or_path))
            load_blip2_model_state(vision_model_name_or_path, model, training_args, "visual_encoder")
        if qformer_model_name_or_path is not None:
            logger.info("loading a bridge model from path{}".format(qformer_model_name_or_path))
            load_blip2_model_state(qformer_model_name_or_path, model, training_args, "Qformer")
        if model_name_or_path is not None:
            logger.info("loading vision and bridge model from path{}".format(model_name_or_path))
            load_blip2_model_state(model_name_or_path, model, training_args, "model")

    @classmethod
    def _resolve_model_file_path_mix(
        cls: Type[PretrainedModel],
        pretrained_model_name_or_path: str,
        weight_name=None,
        from_hf_hub: bool = False,
        cache_dir=None,
        subfolder: str = "",
        config: PretrainedConfig = None,
        convert_from_torch: bool = False,
        use_safetensors=None,
        variant=None,
    ) -> str:

        """resolve model target file path from `` and `cache_dir`

        1. when it is file path:
            return the weight file

        2. when it is model-name:
            2.1 check default `MODEL_HOME` + `model-mame` + model_state.pdparams
            2.2 get the url from `pretrained_resource_files_map`, and set it to `pretrained_model_name_or_path`

        3. when it is local dir:
            check whether the file<local_dir + weight_file> exist

        Args:
            cls (Type[PretrainedModel]): the inherited PretrainedModel class
            pretrained_model_name_or_path (str): the model-name/url/local_dir/local_dir
            cache_dir (Optional[str], optional): cache_dir is used when name_or_path is model-name/url. Defaults to None.
            convert_from_torch (bool, optional): whether support convert pytorch model to paddle model

        Returns:
            str: the model weight file path
        """
        is_sharded = False
        sharded_metadata = None

        is_local_file = False
        if pretrained_model_name_or_path is not None:

            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            if weight_name == "visual_encoder":
                if pretrained_model_name_or_path in VISION_WEIGHTS:
                    is_local_file = os.path.isfile(VISION_WEIGHTS[pretrained_model_name_or_path])
                    pretrained_model_name_or_path = VISION_WEIGHTS[pretrained_model_name_or_path]
                else:
                    is_local = os.path.isdir(pretrained_model_name_or_path)
                    pretrained_model_name_or_path = pretrained_model_name_or_path
            elif weight_name == "Qformer":
                if pretrained_model_name_or_path in BRIDGE_WEIGHTS:
                    is_local_file = os.path.isfile(BRIDGE_WEIGHTS[pretrained_model_name_or_path])
                    pretrained_model_name_or_path = BRIDGE_WEIGHTS[pretrained_model_name_or_path]
                else:
                    is_local = os.path.isdir(pretrained_model_name_or_path)
                    pretrained_model_name_or_path = pretrained_model_name_or_path

            def get_file_path(pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_NAME, variant):
                return os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))

            # pretrained_model_name_or_path is dir
            if is_local:
                if use_safetensors is not False and os.path.isfile(
                    get_file_path(pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_NAME, variant)
                ):
                    # Load from a safetensors checkpoint
                    archive_file = get_file_path(pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_NAME, variant)
                elif use_safetensors is not False and os.path.isfile(
                    get_file_path(pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_NAME, weight_name_suffix())
                ):
                    # Load from a safetensors checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_NAME, weight_name_suffix()
                    )
                elif use_safetensors is not False and os.path.isfile(
                    get_file_path(pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_INDEX_NAME, variant)
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_INDEX_NAME, variant
                    )
                    is_sharded = True
                elif use_safetensors is not False and os.path.isfile(
                    get_file_path(
                        pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_INDEX_NAME, weight_name_suffix()
                    )
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path, subfolder, SAFE_WEIGHTS_INDEX_NAME, weight_name_suffix()
                    )
                    is_sharded = True
                elif os.path.isfile(
                    get_file_path(pretrained_model_name_or_path, subfolder, PADDLE_WEIGHTS_NAME, variant)
                ):
                    # Load from a PaddlePaddle checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path, subfolder, PADDLE_WEIGHTS_NAME, variant
                    )
                elif os.path.isfile(
                    get_file_path(pretrained_model_name_or_path, subfolder, PADDLE_WEIGHTS_INDEX_NAME, variant)
                ):
                    # Load from a sharded PaddlePaddle checkpoint
                    archive_file = get_file_path(
                        pretrained_model_name_or_path, subfolder, PADDLE_WEIGHTS_INDEX_NAME, variant
                    )
                    is_sharded = True
                elif os.path.isfile(
                    get_file_path(
                        pretrained_model_name_or_path, subfolder, PADDLE_WEIGHTS_INDEX_NAME, weight_name_suffix()
                    )
                ):
                    # Load from a sharded PaddlePaddle checkpoint for hybrid parallel model
                    archive_file = get_file_path(
                        pretrained_model_name_or_path, subfolder, PADDLE_WEIGHTS_INDEX_NAME, weight_name_suffix()
                    )
                    is_sharded = True
                elif os.path.isfile(
                    get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        PADDLE_WEIGHTS_NAME,
                        weight_name_suffix(),
                    )
                ):
                    # Load from a PaddlePaddle checkpoint for hybrid parallel model
                    archive_file = get_file_path(
                        pretrained_model_name_or_path,
                        subfolder,
                        PADDLE_WEIGHTS_NAME,
                        weight_name_suffix(),
                    )
                # At this stage we don't have a weight file so we will raise an error.
                elif os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(PYTORCH_WEIGHTS_NAME, variant))
                ):
                    raise ValueError(
                        f"Found {_add_variant(PYTORCH_WEIGHTS_NAME, variant)} in directory"
                        f" {pretrained_model_name_or_path}. Please set convert_from_torch=True in from_pretrained. eg, Model.from_pretrained(model_name, convert_from_torch=True) "
                    )
                else:
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(PADDLE_WEIGHTS_NAME, variant)}, found in directory"
                        f" {pretrained_model_name_or_path}."
                    )
            # pretrained_model_name_or_path is file
            elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)) or is_local_file:
                archive_file = pretrained_model_name_or_path
                is_local = True
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = get_path_from_url_with_filelock(pretrained_model_name_or_path)
            else:
                # set correct filename
                if use_safetensors is not False:
                    filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
                else:
                    filename = _add_variant(PADDLE_WEIGHTS_NAME, variant)

                try:
                    # Load from URL or cache if already cached
                    cached_file_kwargs = dict(
                        cache_dir=cache_dir,
                        subfolder=subfolder,
                        _raise_exceptions_for_missing_entries=False,
                    )
                    resolved_archive_file = None
                    if pretrained_model_name_or_path in cls.pretrained_init_configuration:
                        # fetch the weight url from the `pretrained_resource_files_map`
                        resource_file_url = cls.pretrained_resource_files_map["model_state"][
                            pretrained_model_name_or_path
                        ]
                        resolved_archive_file = cached_file(
                            resource_file_url, _add_variant(PADDLE_WEIGHTS_NAME, variant), **cached_file_kwargs
                        )

                    if resolved_archive_file is None:
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, filename, **cached_file_kwargs
                        )
                    else:
                        # xxx.pdparams in pretrained_resource_files_map renamed model_state.pdparams
                        filename = _add_variant(PADDLE_WEIGHTS_NAME, variant)

                    # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                    # result when internet is up, the repo and revision exist, but the file does not.
                    if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        if resolved_archive_file is not None:
                            is_sharded = True
                        elif use_safetensors:
                            raise EnvironmentError(
                                f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} and thus cannot be loaded with `safetensors`. Please make sure that the model has been saved with `safe_serialization=True` or do not set `use_safetensors=True`."
                            )
                        else:
                            # This repo has no safetensors file of any kind, we switch to PyTorch.
                            filename = _add_variant(PADDLE_WEIGHTS_NAME, variant)
                            resolved_archive_file = cached_file(
                                pretrained_model_name_or_path, filename, **cached_file_kwargs
                            )
                    if resolved_archive_file is None and filename == _add_variant(PADDLE_WEIGHTS_NAME, variant):
                        # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path,
                            _add_variant(PADDLE_WEIGHTS_INDEX_NAME, variant),
                            **cached_file_kwargs,
                        )
                        # raise ValueError(resolved_archive_file)
                        if resolved_archive_file is not None:
                            is_sharded = True
                    if resolved_archive_file is None:
                        # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
                        # message.
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(PADDLE_WEIGHTS_NAME, variant)}."
                        )
                except Exception:
                    # For any other exception, we throw a generic error.
                    raise EnvironmentError(
                        f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                        " from 'https://paddlenlp.bj.bcebos.com'"
                    )

            if is_local:
                logger.info(f"Loading weights file {archive_file}")
                resolved_archive_file = archive_file
            else:
                logger.info(f"Loading weights file {filename} from cache at {resolved_archive_file}")
        else:
            resolved_archive_file = None

            def get_file_path(pretrained_model_name_or_path):
                return os.path.join(pretrained_model_name_or_path)

            # pretrained_model_name_or_path is dir
            if is_local_file:
                if os.path.isfile(get_file_path(pretrained_model_name_or_path)):
                    # Load from a PaddlePaddle checkpoint
                    resolved_archive_file = get_file_path(pretrained_model_name_or_path)
            elif is_remote_url(pretrained_model_name_or_path):
                resolved_archive_file = get_path_from_url_with_filelock(pretrained_model_name_or_path)
            else:
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                    " from 'https://paddlenlp.bj.bcebos.com'"
                )

        return resolved_archive_file, sharded_metadata, is_sharded


class Blip2ForConditionalGeneration(Blip2PretrainedModel):
    config_class = Blip2Config
    main_input_name = "pixel_values"

    def __init__(
        self,
        config: Blip2Config,
    ):
        super().__init__(config)
        from paddlemix.models.blip2.eva_vit import VisionTransformer

        config.vision_config.update({"mp_degree": config.mp_degree})
        self.visual_encoder = VisionTransformer(config=config.vision_config)
        self.freeze_vit = config.freeze_vit
        self.train_stage1 = False
        if self.freeze_vit:
            # freeze vit except the post layer norm layer.
            for name, param in self.visual_encoder.named_parameters():
                param.stop_gradient = True
            self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logger.info("freeze vision encoder")
        if config.get("train_mode", None) == "stage1":
            self.train_stage1 = True
            self.tokenizer = self.init_tokenizer()
            self.Qformer = BertLMHeadModel(
                config=config.qformer_config,
                encoder_width=self.visual_encoder.num_features,
                train_in_satge1=True,
                tokenizer_length=len(self.tokenizer),
            )

            state_dict = self.Qformer.state_dict()
            for name, param in self.Qformer.named_parameters():
                if "_query" in name:
                    key_orig = name.replace("_query", "")
                    param.copy_(state_dict[key_orig], False)

            self.temp = self.create_parameter(
                shape=(1,), default_initializer=paddle.nn.initializer.Constant(value=0.07)
            )
            self.max_txt_len = config.get("max_txt_len")
        else:
            if config.use_decoder_only_language_model:
                if "opt" in config.text_config:
                    language_model = OPTForCausalLM.from_pretrained(
                        config.text_config,
                        load_state_as_np=True,
                        ignore_mismatched_sizes=True,
                    )
                elif "llama" in config.text_config:
                    from paddlenlp.transformers.llama.configuration import LlamaConfig

                    if config.mp_degree > 1:
                        import paddle.distributed.fleet as fleet

                        hcg = fleet.get_hybrid_communicate_group()
                        language_model = LlamaForCausalLM.from_pretrained(
                            config.text_config,
                            tensor_parallel_degree=config.mp_degree,
                            tensor_parallel_rank=hcg.get_model_parallel_rank(),
                            tensor_parallel_output=False,
                        )
                    else:
                        language_model = LlamaForCausalLM.from_pretrained(
                            config.text_config,
                            tensor_parallel_output=False,
                        )
                    language_model.hidden_size = LlamaConfig.from_pretrained(config.text_config).hidden_size
                    language_model.pad_token_id = LlamaConfig.from_pretrained(config.text_config).pad_token_id
                else:
                    raise NotImplementedError
            else:
                from paddlenlp.transformers import T5Config

                t5_config = T5Config(config.text_config)
                for key, value in config.text_config.items():
                    t5_config[key] = config.text_config[key]
                language_model = T5ForConditionalGeneration.from_pretrained(config.text_config, load_state_as_np=True)
                language_model.hidden_size = t5_config["d_model"]

            self.language_model = language_model
            for name, param in self.language_model.named_parameters():
                param.stop_gradient = True
            self.pad_token_id = self.language_model.pad_token_id

            self.Qformer = BertLMHeadModel(
                config=config.qformer_config,
                encoder_width=self.visual_encoder.num_features,
                train_in_satge1=False,
                text_hidden_size=self.language_model.hidden_size,
            )
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None

    def get_input_embeddings(self) -> nn.Layer:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: paddle.Tensor,
        input_ids: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
        text_input_stage1: Optional[paddle.Tensor] = None,
        **kwargs
    ):

        if self.train_stage1:
            return self.forward_stage1(pixel_values, text_input_stage1)
        else:
            return self.forward_stage2(
                pixel_values,
                input_ids,
                attention_mask,
                return_dict,
                decoder_input_ids=kwargs.get("decoder_input_ids", None),
                decoder_attention_mask=kwargs.get("decoder_attention_mask", None),
            )

    def forward_stage2(
        self,
        pixel_values: paddle.Tensor,
        input_ids: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        return_dict: Optional[bool] = None,
        decoder_input_ids: Optional[paddle.Tensor] = None,
        decoder_attention_mask: Optional[paddle.Tensor] = None,
    ) -> Union[Tuple, Blip2ForConditionalGenerationModelOutput]:
        r"""
        Returns:
        Examples:
        Image captioning (without providing a text prompt):
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from paddlenlp.transformers import Blip2Processor, Blip2ForConditionalGeneration
        >>> import paddle
        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        >>> model = Blip2ForConditionalGeneration.from_pretrained(
        ...     "Salesforce/blip2-flan-t5-xl"
        ... )
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(images=image, return_tensors="pd")
        >>> generated_ids, scores = model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        two cats laying on a couch
        ```
        Visual question answering (prompt = question):
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from paddlenlp.transformers import Blip2Processor, Blip2ForConditionalGeneration
        >>> import paddle
        >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        >>> model = Blip2ForConditionalGeneration.from_pretrained(
        ...     "Salesforce/blip2-flan-t5-xl"
        ... )
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> prompt = "Question: how many cats are there? Answer:"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pd")
        >>> generated_ids, scores= model.generate(**inputs)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        two
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        with paddle.amp.auto_cast(level="O2"):
            image_embeds = self.Qformer.ln_vision(self.visual_encoder(pixel_values))
        image_embeds = image_embeds.astype("float32")

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = paddle.ones(image_embeds.shape[:-1], dtype="int64")
        query_tokens = self.Qformer.query_tokens.expand([image_embeds.shape[0], -1, -1])
        query_outputs = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.Qformer.language_projection(query_output)
        language_model_attention_mask = paddle.ones(language_model_inputs.shape[:-1], dtype="int64")
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = paddle.concat([language_model_inputs, inputs_embeds], axis=1)
        if attention_mask is None:
            attention_mask = paddle.ones_like(input_ids)

        attention_mask = paddle.concat([language_model_attention_mask, attention_mask], axis=1)

        if self.config.use_decoder_only_language_model:
            targets = input_ids * (1 - (input_ids == self.pad_token_id).astype(input_ids.dtype)) + (
                input_ids == self.pad_token_id
            ).astype(input_ids.dtype) * (-100)

            empty_targets = paddle.ones(language_model_attention_mask.shape, dtype="int64").fill_(-100)
            labels = paddle.concat([empty_targets, targets], axis=1)
            labels.stop_gradient = True
            with paddle.amp.auto_cast(level="O2"):
                outputs = self.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=labels,
                )
                loss = outputs.loss
        else:
            targets = decoder_input_ids * (
                1 - (decoder_input_ids == self.pad_token_id).astype(decoder_input_ids.dtype)
            ) + (decoder_input_ids == self.pad_token_id).astype(input_ids.dtype) * (-100)
            targets.stop_gradient = True
            with paddle.amp.auto_cast(level="O2"):
                outputs = self.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    decoder_attention_mask=decoder_attention_mask,
                    return_dict=return_dict,
                    labels=targets,
                )
                loss = outputs[0]
        return Blip2ForConditionalGenerationModelOutput(
            loss=loss,
        )

    def forward_stage1(self, pixel_values, text_input):
        text = text_input

        image = pixel_values
        image_embeds = self.Qformer.ln_vision(self.visual_encoder(image))

        image_atts = paddle.ones(image_embeds.shape[:-1], dtype="int64")
        query_tokens = self.Qformer.query_tokens.expand(shape=[image_embeds.shape[0], -1, -1])
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )
        image_feats = paddle.nn.functional.normalize(
            x=self.Qformer.vision_proj(query_output.last_hidden_state), axis=-1
        )

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_attention_mask=True,
            return_tensors="pd",
        )
        text_output = self.Qformer.bert(
            text_tokens.input_ids, attention_mask=text_tokens.attention_mask, return_dict=True
        )
        text_feat = paddle.nn.functional.normalize(
            self.Qformer.text_proj(text_output.last_hidden_state[:, 0, :]), axis=-1
        )

        # Image-text Contrastive
        # image_feats_all = image_feats
        # text_feat_all = text_feat
        image_feats_all = concat_all_gather(image_feats)
        text_feat_all = concat_all_gather(text_feat)
        sim_q2t = paddle.matmul(image_feats.unsqueeze(axis=1), text_feat_all.unsqueeze(axis=-1)).squeeze()
        sim_i2t = sim_q2t.max(axis=-1)
        sim_i2t = sim_i2t / self.temp
        sim_t2q = paddle.matmul(
            x=text_feat.unsqueeze(axis=1).unsqueeze(axis=1), y=image_feats_all.transpose(perm=[0, 2, 1])
        ).squeeze()
        sim_t2i = sim_t2q.max(axis=-1)
        sim_t2i = sim_t2i / self.temp

        rank = dist.get_rank()
        bs = image.shape[0]

        targets = paddle.linspace(start=rank * bs, stop=rank * bs + bs - 1, num=bs).astype(int)
        one_hot_label = paddle.nn.functional.one_hot(targets, num_classes=sim_i2t.shape[1])
        smooth_label = paddle.nn.functional.label_smooth(label=one_hot_label, epsilon=0.1)
        loss_itc = (
            paddle.nn.functional.cross_entropy(input=sim_i2t, label=smooth_label, soft_label=True)
            + paddle.nn.functional.cross_entropy(input=sim_t2i, label=smooth_label, soft_label=True)
        ) / 2
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        image_embeds_world = all_gather_with_grad(image_embeds)
        with paddle.no_grad():
            weights_t2i = paddle.nn.functional.softmax(x=sim_t2i, axis=1) + 0.0001
            weights_t2i_list = paddle.chunk(weights_t2i, chunks=paddle.distributed.get_world_size(), axis=-1)
            weights_t2i_list[rank].fill_diagonal_(value=0)
            weights_t2i = paddle.concat(weights_t2i_list, axis=-1)
            weights_i2t = paddle.nn.functional.softmax(x=sim_i2t, axis=1) + 0.0001
            weights_i2t_list = paddle.chunk(weights_i2t, chunks=paddle.distributed.get_world_size(), axis=-1)
            weights_i2t_list[rank].fill_diagonal_(value=0)
            weights_i2t = paddle.concat(weights_i2t_list, axis=-1)
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = paddle.multinomial(x=weights_t2i[b], num_samples=1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = paddle.stack(x=image_embeds_neg, axis=0)
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = paddle.multinomial(x=weights_i2t[b], num_samples=1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])
        text_ids_neg = paddle.stack(x=text_ids_neg, axis=0)
        text_atts_neg = paddle.stack(x=text_atts_neg, axis=0)
        text_ids_all = paddle.concat(x=[text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], axis=0)
        text_atts_all = paddle.concat(
            x=[text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg], axis=0
        )
        query_tokens_itm = self.Qformer.query_tokens.expand(shape=[text_ids_all.shape[0], -1, -1])
        query_atts_itm = paddle.ones(shape=query_tokens_itm.shape[:-1], dtype="int64")
        attention_mask_all = paddle.concat(x=[query_atts_itm, text_atts_all], axis=1)
        image_embeds_all = paddle.concat(x=[image_embeds, image_embeds_neg, image_embeds], axis=0)
        image_atts_all = paddle.ones(shape=image_embeds_all.shape[:-1], dtype="int64")
        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.shape[1], :]
        vl_output = self.Qformer.itm_head(vl_embeddings)
        logits = vl_output.mean(axis=1)

        itm_labels = paddle.concat([paddle.ones([bs], dtype="int64"), paddle.zeros([2 * bs], dtype="int64")], axis=0)
        loss_itm = paddle.nn.functional.cross_entropy(input=logits, label=itm_labels)
        # Image Captioning

        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, (0)] = self.tokenizer.bos_token_id
        labels = masked_fill(decoder_input_ids, decoder_input_ids == self.tokenizer.pad_token_id, -100)
        query_atts = paddle.ones(shape=query_tokens.shape[:-1], dtype="int64")
        attention_mask = paddle.concat(x=[query_atts, text_tokens.attention_mask], axis=1)
        lm_output = self.Qformer(
            decoder_input_ids,
            attention_mask=attention_mask,
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
        )
        loss_lm = lm_output.loss
        return Blip2ForStage1ModelOutput(
            loss=loss_itc + loss_itm + loss_lm, loss_itc=loss_itc, loss_itm=loss_itm, loss_lm=loss_lm
        )

    @paddle.no_grad()
    def generate_stage1(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))
        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, axis=0)
        else:
            num_beams = 1
        image_atts = paddle.ones(shape=image_embeds.shape[:-1], dtype="int64")
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
        input_ids = paddle.empty(shape=[image.shape[0], 1], dtype="int64").fill_(value=self.tokenizer.bos_token_id)
        query_tokens = self.query_tokens.expand(shape=[image_embeds.shape[0], -1, -1])
        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs,
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    @paddle.no_grad()
    def generate(
        self,
        pixel_values: paddle.Tensor,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        **generate_kwargs,
    ) -> paddle.Tensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.
        Args:
            pixel_values (`paddle.Tensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            input_ids (`paddle.Tensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`paddle.Tensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        batch_size = pixel_values.shape[0]
        image_embeds = self.Qformer.ln_vision(self.visual_encoder(pixel_values))
        image_attention_mask = paddle.ones(image_embeds.shape[:-1], dtype="int64")

        query_tokens = self.Qformer.query_tokens.expand([image_embeds.shape[0], -1, -1])
        query_outputs = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs = self.Qformer.language_projection(query_output)
        language_attention_mask = paddle.ones(language_model_inputs.shape[:-1], dtype="int64")
        if input_ids is None:
            input_ids = paddle.to_tensor([[self.config.text_config.bos_token_id]]).tile([batch_size, 1])
        if attention_mask is None:
            attention_mask = paddle.ones_like(input_ids)
        attention_mask = paddle.concat([language_attention_mask, attention_mask], axis=1)
        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = paddle.concat([language_model_inputs, inputs_embeds], axis=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            top_p=0.9,
            decode_strategy="greedy_search",
            temperature=1,
            num_beams=5,
            max_length=30,
            min_length=8,
            eos_token_id=50118,
            repetition_penalty=1,
            length_penalty=1,
            num_return_sequences=1,
        )

        return outputs

    @paddle.no_grad()
    def encode_image(
        self,
        pixel_values: paddle.Tensor,
        **kwargs,
    ):
        image_embeds = self.ln_vision(self.visual_encoder(pixel_values.astype("float16")))
        image_embeds = image_embeds.astype("float32")

        image_attention_mask = paddle.ones(image_embeds.shape[:-1], dtype="int64")

        query_tokens = self.query_tokens.expand([image_embeds.shape[0], -1, -1])
        query_outputs = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs[0]
        return query_output

    @paddle.no_grad()
    def predict_answers(
        self,
        pixel_values: paddle.Tensor,
        input_ids: Optional[paddle.Tensor] = None,
        attention_mask: Optional[paddle.Tensor] = None,
        max_len=10,
        min_len=1,
        **kwargs
    ):
        # batch_size = pixel_values.shape[0]
        image_embeds = self.Qformer.ln_vision(self.visual_encoder(pixel_values))
        image_attention_mask = paddle.ones(image_embeds.shape[:-1], dtype="int64")

        query_tokens = self.Qformer.query_tokens.expand([image_embeds.shape[0], -1, -1])
        query_outputs = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state

        language_model_inputs = self.Qformer.language_projection(query_output)
        language_attention_mask = paddle.ones(language_model_inputs.shape[:-1], dtype="int64")

        attention_mask = paddle.concat([language_attention_mask, attention_mask], axis=1)
        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = paddle.concat([language_model_inputs, inputs_embeds], axis=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=False,
            decode_strategy="greedy_search",
            temperature=1,
            num_beams=5,
            max_length=max_len,
            min_length=min_len,
            eos_token_id=50118,
            repetition_penalty=1,
            length_penalty=0,
        )

        return outputs


from contextlib import contextmanager


@contextmanager
def dtype_guard(dtype="float32"):
    origin_dtype = paddle.get_default_dtype()
    paddle.set_default_dtype(dtype)
    try:
        yield
    finally:
        paddle.set_default_dtype(origin_dtype)
