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

""" Paddle BLIP2 base model."""
import gc
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import paddle
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.transformers.model_outputs import ModelOutput
from paddlenlp.transformers.model_utils import _add_variant, weight_name_suffix
from paddlenlp.utils.env import PADDLE_WEIGHTS_NAME, SAFE_WEIGHTS_NAME

from paddlemix.models.model_utils import MixPretrainedModel
from paddlemix.utils.log import logger

from .configuration import Blip2Config
from .utils import blip2_load

VISION_WEIGHTS = {"eva_vit_g": "https://bj.bcebos.com/paddlenlp/models/community/paddlemix/blip2-stage2/eva_vit_g"}
BRIDGE_WEIGHTS = {
    "qformer-stage2": "https://bj.bcebos.com/paddlenlp/models/community/paddlemix/blip2-stage2/Qformer",
    "qformer-stage1": "https://bj.bcebos.com/paddlenlp/models/community/paddlemix/blip2-stage1/Qformer",
}
from paddlenlp.transformers.utils import resolve_cache_dir

__all__ = ["Blip2ForConditionalGenerationModelOutput", "Blip2ForStage1ModelOutput", "Blip2PretrainedModel"]


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


from contextlib import contextmanager


@contextmanager
def dtype_guard(dtype="float32"):
    origin_dtype = paddle.get_default_dtype()
    paddle.set_default_dtype(dtype)
    try:
        yield
    finally:
        paddle.set_default_dtype(origin_dtype)


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
    The class for pretrained model used in BLIP2.
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
        Returns:
            PretrainedModel: An instance of `PretrainedModel`.

        Example:
            .. code-block::
                from paddlemix.models.blip2.modeling import Blip2ForConditionalGeneration
                model = Blip2ForConditionalGeneration.from_pretrained(config.model_name_or_path)
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

        # from_hf_hub default enable convert_from_torch
        if from_hf_hub and convert_from_torch is None:
            logger.warning(
                "If you are attempting to load weights from Hugging Face Hub and want to disable the default behavior of considering torch weights,"
                " you can set ·convert_from_torch=False·. By default, `convert_from_torch` is set to `True`. "
            )
            convert_from_torch = True
        # convert_from_torch default is False
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
        resolved_archive_file, *_, sharded_metadata, is_sharded = cls._resolve_model_file_path(
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
        training_args=None,
        **kwargs,
    ) -> Tuple[List[str]]:
        """load the state_dict into model, and do the following things:

            * resolve the pretrained model name or path by checking if they exist in the cache and then
            download them.
            * load the pretrained model and refine the state dict if necessary.
            * filter the weight keys and set the state_dict to the model.

        Args:
            training_args (bool, optional): whether ignore error when tensor size mismatched. Defaults to False.
        Returns:
            Tuple[List[str]]: _description_
        """
        qformer_model_name_or_path = (
            kwargs.get("bridge_name_or_path", None)
            if kwargs.get("bridge_name_or_path", None)
            else self.config.get("bridge_name_or_path", None)
        )
        vision_model_name_or_path = (
            kwargs.get("vision_name_or_path", None)
            if kwargs.get("vision_name_or_path", None)
            else self.config.get("vision_name_or_path", None)
        )
        model_name_or_path = (
            kwargs.get("vision_and_bridge_name_or_path", None)
            if kwargs.get("vision_and_bridge_name_or_path", None)
            else self.config.get("vision_and_bridge_name_or_path", None)
        )

        if (not qformer_model_name_or_path and not vision_model_name_or_path) and model_name_or_path is None:
            ValueError(
                "either vision_and_bridge_name_or_path or (bridge_name_or_path and vision_name_or_path) should be set."
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
            if model_name_or_path is not None:

                model_name_or_path = str(model_name_or_path)
                if weight_name == "visual_encoder":
                    if model_name_or_path in VISION_WEIGHTS:
                        model_name_or_path = VISION_WEIGHTS[model_name_or_path]
                    else:
                        model_name_or_path = model_name_or_path
                elif weight_name == "Qformer":
                    if model_name_or_path in BRIDGE_WEIGHTS:
                        model_name_or_path = BRIDGE_WEIGHTS[model_name_or_path]
                    else:
                        model_name_or_path = model_name_or_path

            # resolve model_weight file
            resolved_archive_file, *_ = self._resolve_model_file_path(
                pretrained_model_name_or_path=model_name_or_path,
                cache_dir=cache_dir,
                subfolder=subfolder,
                variant=variant,
            )

            # Check if `_keep_in_fp32_modules` is not None
            use_keep_in_fp32_modules = (self._keep_in_fp32_modules is not None) and dtype == "float16"

            if use_keep_in_fp32_modules:
                # low_cpu_mem_usage = True
                keep_in_fp32_modules = self._keep_in_fp32_modules
            else:
                keep_in_fp32_modules = []

            # load_pretrained_model
            model_state_dict = self.state_dict()

            expected_keys = self.get_expected_keys(model_state_dict, weight_name)

            # Set some modules to fp32 if any
            if keep_in_fp32_modules is not None:
                for name, param in self.named_parameters():
                    if any(module_to_keep_in_fp32 in name for module_to_keep_in_fp32 in keep_in_fp32_modules):
                        param = param.to(dtype=paddle.float32)
            # Make sure we are able to load base models as well as derived models (with heads)

            def _find_mismatched_keys(
                state_dict,
                model_state_dict,
                loaded_keys,
                ignore_mismatched_sizes,
            ):
                mismatched_keys = []
                if ignore_mismatched_sizes:
                    for checkpoint_key in loaded_keys:
                        # If the checkpoint is sharded, we may not have the key here.
                        if checkpoint_key not in state_dict:
                            continue
                        model_key = checkpoint_key
                        if (
                            model_key in model_state_dict
                            and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                        ):
                            mismatched_keys.append(
                                (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                            )
                            del state_dict[checkpoint_key]
                return mismatched_keys

            resolved_archive_file = [resolved_archive_file]
            mismatched_keys = []
            for shard_file in resolved_archive_file:
                loaded_state_dict_keys = blip2_load(
                    shard_file, model, training_args, map_location="cpu", weight_name=weight_name
                )
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

                mismatched_keys += _find_mismatched_keys(
                    state_dict,
                    model_state_dict,
                    loaded_state_dict_keys,
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
            load_blip2_model_state(vision_model_name_or_path, self, training_args, "visual_encoder")
        if qformer_model_name_or_path is not None:
            logger.info("loading a bridge model from path{}".format(qformer_model_name_or_path))
            load_blip2_model_state(qformer_model_name_or_path, self, training_args, "Qformer")
        if model_name_or_path is not None:
            logger.info("loading vision and bridge model from path{}".format(model_name_or_path))
            load_blip2_model_state(model_name_or_path, self, training_args)

    def save_pretrained(
        self,
        save_dir: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = paddle.save,
        max_shard_size: Union[int, str] = "10GB",
        safe_serialization: bool = False,
        variant: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Saves model configuration and related resources (model state) as files
        under `save_dir`. The model configuration would be saved into a file named
        "model_config.json", and model state would be saved into a file
        named "model_state.pdparams".

        The `save_dir` can be used in `from_pretrained` as argument value
        of `pretrained_model_name_or_path` to re-load the trained model.

        Args:
            save_dir (str): Directory to save files into.

        Example:
            .. code-block::

                from paddlenlp.transformers import BertForSequenceClassification

                model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
                model.save_pretrained('./trained_model/')
                # reload from save_directory
                model = BertForSequenceClassification.from_pretrained('./trained_model/')
        """
        import copy

        from paddlenlp.transformers.model_utils import shard_checkpoint, unwrap_model

        assert not os.path.isfile(save_dir), "Saving directory ({}) should be a directory, not a file".format(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        merge_tensor_parallel = kwargs.get("merge_tensor_parallel", False)
        config_to_save = kwargs.get("config_to_save", None)
        shard_format = kwargs.get("shard_format", "naive")  # support naive pipeline

        save_directory = save_dir

        os.makedirs(save_directory, exist_ok=True)
        # Save model config

        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)
        if config_to_save is None:
            config_to_save = copy.deepcopy(model_to_save.config)
        config_to_save.mp_degree = getattr(config_to_save, "config_to_save", 1)
        # Save the model
        state_dict = model_to_save.state_dict()
        self._keys_to_ignore_on_save = []
        ignore_visual_encoder = False
        for k, v in state_dict.items():
            if v.stop_gradient:
                self._keys_to_ignore_on_save.append(k)
                if "visual_encoder" in k:
                    ignore_visual_encoder = True
        if config_to_save.mp_degree > 1:
            if merge_tensor_parallel:
                state_dict = model_to_save.merge_tensor_parallel(state_dict, config_to_save)
                if config_to_save.tensor_parallel_rank != 0:
                    logger.info("Saving with merge_tensor_parallel, tensor_parallel_rank > 0 don't need save")
                    return
                if variant is not None and "tp" in variant:
                    variant = "_".join([x for x in variant.split("_") if "tp" not in x])
            else:
                variant = weight_name_suffix() if variant is None else variant
        if not ignore_visual_encoder:
            config_to_save.model_name_or_path = save_dir
        else:
            config_to_save.bridge_name_or_path = save_dir

        # Attach architecture to the config
        config_to_save.architectures = [model_to_save.__class__.__name__]
        # Save the config
        if is_main_process:
            config_to_save.save_pretrained(save_directory)
        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            for ignore_key in self._keys_to_ignore_on_save:
                if ignore_key in state_dict.keys():
                    del state_dict[ignore_key]

        # Shard the model if it is too big.
        weights_name = SAFE_WEIGHTS_NAME if safe_serialization else PADDLE_WEIGHTS_NAME
        weights_name = _add_variant(weights_name, variant)

        # Save model
        shards, index = shard_checkpoint(
            state_dict, max_shard_size=max_shard_size, weights_name=weights_name, shard_format=shard_format
        )

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".pdparams", "").replace(".safetensors", "")

            # make sure that file to be deleted matches format of sharded file, e.g. paddle_model-00001-of-00005
            filename_no_suffix = filename.replace(".pdparams", "").replace(".safetensors", "")
            reg = re.compile("(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in shards.keys()
                and is_main_process
                and reg.fullmatch(filename_no_suffix) is not None
            ):
                os.remove(full_filename)

        # Save the model
        for shard_file, shard in shards.items():
            save_function(shard, os.path.join(save_directory, shard_file))
        path_to_weights = os.path.join(save_directory, _add_variant(PADDLE_WEIGHTS_NAME, variant))
        logger.info(f"Model weights saved in {path_to_weights}")
