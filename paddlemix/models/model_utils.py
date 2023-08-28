# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import os
import re
import warnings
from typing import List, Optional, Tuple

import numpy as np
import paddle
from paddlenlp.transformers import PretrainedModel
from tqdm.auto import tqdm

from paddlemix.utils import device_guard, paddlemix_load
from paddlemix.utils.env import MODEL_HOME
from paddlemix.utils.log import logger
from typing import Type
from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.transformers.model_utils import PretrainedModel,resolve_weight_file_from_hf_hub,_add_variant,weight_name_suffix
from paddlenlp.utils.env import (
    CONFIG_NAME,
    LEGACY_CONFIG_NAME,
    PADDLE_WEIGHTS_INDEX_NAME,
    PADDLE_WEIGHTS_NAME,
    PYTORCH_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
)
VISION_WEIGHTS={"eva_vit_g":"blip2-stage2/eva_vit_g/model_state.pdparams"}
BRIDGE_WEIGHTS={"qformer-stage2":"blip2-stage2/Qformer/model_state.pdparams"}
from paddlenlp.transformers.utils import     (ContextManagers,
                                                InitTrackerMeta,
                                                adapt_stale_fwd_patch,
                                                cached_file,
                                                cached_file_for_hf_hub,
                                                convert_file_size_to_int,
                                                dtype_byte_size,
                                                fn_args_to_dict,
                                                get_checkpoint_shard_files,
                                                is_paddle_support_lazy_init,
                                                is_safetensors_available,
                                                paddlenlp_load,
                                                resolve_cache_dir,
                                                weight_name_suffix,
                                            )
from paddle.utils.download import is_url as is_remote_url
from paddlenlp.utils.downloader import get_path_from_url_with_filelock, hf_file_exists

__all__ = ["MixPretrainedModel"]


def resolve_cache_dir(pretrained_model_name_or_path: str, cache_dir: Optional[str] = None) -> str:
    """resolve cache dir for PretrainedModel and PretrainedConfig

    Args:
        pretrained_model_name_or_path (str): the name or path of pretrained model
        cache_dir (str): cache_dir for models
    """
    if os.path.isdir(pretrained_model_name_or_path):
        return pretrained_model_name_or_path

    if cache_dir is not None:
        # since model_clas.from_pretrained calls config_clas.from_pretrained, the model_name may get appended twice
        if cache_dir.endswith(pretrained_model_name_or_path):
            return cache_dir
        else:
            return os.path.join(cache_dir, pretrained_model_name_or_path)
    return os.path.join(MODEL_HOME, pretrained_model_name_or_path)


def _load_state_dict_into_model(model_to_load, state_dict, start_prefix):
    # torch will cast dtype in load_state_dict, but paddle strictly check dtype
    _convert_state_dict_dtype_and_shape(state_dict, model_to_load)

    error_msgs = []

    if len(start_prefix) > 0:
        for key in list(state_dict.keys()):
            if key.startswith(start_prefix):
                state_dict[key.replace(start_prefix, "")] = state_dict.pop(key)

    # TODO: add return status to state_dict
    with warnings.catch_warnings(record=True) as w:
        warnings.resetwarnings()
        # paddlenlp hold  missing_keys , just ignore not found warnings.
        warnings.filterwarnings("ignore", message=r".*is not found in the provided dict.*")
        model_to_load.set_state_dict(state_dict)
        error_msgs.extend([str(x.message) for x in w])

    del state_dict

    return error_msgs


def _convert_state_dict_dtype_and_shape(state_dict, model_to_load):
    # convert the dtype of state dict
    def is_0d_or_1d(tensor):
        return len(tensor.shape) == 0 or list(tensor.shape) == [1]

    for key, value in model_to_load.state_dict().items():
        if key in state_dict:
            if isinstance(state_dict[key], np.ndarray):
                raise ValueError(
                    "convert_state_dict_dtype expected paddle.Tensor not numpy.ndarray, plase convert numpy.ndarray to paddle.Tensor"
                )
            if state_dict[key].is_floating_point() and state_dict[key].dtype != value.dtype:
                state_dict[key] = paddle.cast(state_dict.pop(key), value.dtype)

            # unified 0d and 1d tensor
            if is_0d_or_1d(value) and is_0d_or_1d(state_dict[key]):
                if list(value.shape) != list(state_dict[key].shape):
                    state_dict[key] = paddle.reshape(state_dict.pop(key), value.shape)


def _load_state_dict_into_meta_model(
    model,
    state_dict,
    loaded_state_dict_keys,  # left for now but could be removed, see below
    start_prefix,
    expected_keys,
    dtype=None,
    is_safetensors=False,
    keep_in_fp32_modules=None,
):
    """
    This is somewhat similar to `_load_state_dict_into_model`, but deals with a model that has some or all of its
    params on a `meta` device. It replaces the model params with the data from the `state_dict`, while moving the
    params back to the normal device, but only for `loaded_state_dict_keys`.

    `start_prefix` is used for models which insert their name into model keys, e.g. `bert` in
    `bert.pooler.dense.weight`

    """
    from paddle.common_ops_import import convert_np_dtype_to_dtype_

    dtype = convert_np_dtype_to_dtype_(dtype)
    error_msgs = []

    for param_name, param in state_dict.items():
        # First part of the test is always true as loaded_state_dict_keys always contains state_dict keys.
        if param_name not in loaded_state_dict_keys or param_name not in expected_keys:
            continue

        if param_name.startswith(start_prefix):
            param_name = param_name[len(start_prefix) :]

        if param.place != paddle.framework._current_expected_place():
            param = param._copy_to(paddle.framework._current_expected_place(), False)

        # # We convert floating dtypes to the `dtype` passed. We want to keep the buffers/params
        # # in int/uint/bool and not cast them.
        if dtype is not None and paddle.is_floating_point(param):
            if (
                keep_in_fp32_modules is not None
                and any(module_to_keep_in_fp32 in param_name for module_to_keep_in_fp32 in keep_in_fp32_modules)
                and dtype == paddle.float16
            ):
                param = param.astype(dtype=paddle.float32)
            else:
                param = param.astype(dtype=dtype)

        if dtype is None:
            old_param = model
            splits = param_name.split(".")
            for split in splits:
                old_param = getattr(old_param, split)
                if old_param is None:
                    break

            if old_param is not None:
                param = param.astype(dtype=old_param.dtype)

        with paddle.no_grad():
            model.state_dict()[param_name].get_tensor()._share_data_with(param.value().get_tensor())
            param.value().get_tensor()._clear()

    return error_msgs


class MixPretrainedModel(PretrainedModel):
    """
    The base class for all pretrained models used in PaddleMIX. It mainly provides common methods
    for loading (construction and loading) and saving pretrained models. Loading can be
    customized in loaded_pretrained when the pretrained model is used for different stages.

    The most difference between `PretrainedModel` and `MixPretrainedModel` is that
    `MixPretrainedModel` increaces `load_pretrained` method to support loading pretrained weights
    in differenet stages after construction. The other methods are the same as class
    `paddlenlp.transformers.model_utils.PretrainedModel`.
    """

    def __init__(self, *args, **kwargs):
        super(MixPretrainedModel, self).__init__(*args, **kwargs)

    def get_expected_keys(self, model_state_dict,name=None):
        model_list=[]
        if name=="bridge":
            self._keys_to_ignore_on_load_unexpected=["visual_encoder","language_model"]
            for key in model_state_dict.keys():
                if "visual_encoder" not in key and "language_model" not in key:
                    model_list.append(key)
                    
        elif name=="vision":
            self._keys_to_ignore_on_load_unexpected=["Qformer","language_model"]
            for key in model_state_dict.keys():
                if "visual_encoder" in key:
                    model_list.append(key)
        else:
            self._keys_to_ignore_on_load_unexpected=["language_model"]
            for key in model_state_dict.keys():
                if "language_model" not in key:
                    model_list.append(key)
                
            
        return model_list

    def refine_state_dict(self, state_dict):
        # preprocess the weight loaded here, such as interpolatation
        pass

    def load_pretrained(
        self,
        config,
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
        qformer_model_name_or_path =  config.get("bridge_name_or_path",None)
        vision_model_name_or_path = config.get("vision_name_or_path",None)
        model_name_or_path = config.get("model_name_or_path",None)
        if qformer_model_name_or_path and  vision_model_name_or_path and model_name_or_path is None:
            ValueError("either model_name_or_path or (bridge_model_name_or_path and vision_model_name_or_path) should be set.")
        def load_blip2_model_state(model_name_or_path=None,name=None,
            state_dict=None,
            ignore_mismatched_sizes=False,                    
            low_cpu_mem_usage=False,
            dtype=None,
            cache_dir=None,
            subfolder="",
            variant=None,):
            cache_dir = resolve_cache_dir(model_name_or_path, cache_dir)

            # Keep in fp32 modules
            keep_in_fp32_modules = None
            use_keep_in_fp32_modules = False

            # resolve model_weight file
            resolved_archive_file, sharded_metadata, is_sharded = self._resolve_model_file_path(
                model_name_or_path,
                cache_dir=cache_dir,
                subfolder=subfolder,
                variant=variant,
                name=name
            )

            # Check if `_keep_in_fp32_modules` is not None
            use_keep_in_fp32_modules = (self._keep_in_fp32_modules is not None) and dtype == "float16"
            
            loaded_state_dict_keys=self.state_dict()


            if use_keep_in_fp32_modules:
                # low_cpu_mem_usage = True
                keep_in_fp32_modules = self._keep_in_fp32_modules
            else:
                keep_in_fp32_modules = []

            # load_pretrained_model
            is_safetensors = False

            model_state_dict = self.state_dict()

            expected_keys = self.get_expected_keys(model_state_dict,name)

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
            start_prefix = ""
            model_to_load = self
            if len(self.base_model_prefix) > 0 and not hasattr(self, self.base_model_prefix) and has_prefix_module:
                start_prefix = self.base_model_prefix + "."
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
                error_msgs = _load_state_dict_into_model(model_to_load, state_dict, start_prefix)
            else:
                # Sharded checkpoint or whole but low_cpu_mem_usage==True

                # This should always be a list but, just to be sure.
                if not isinstance(resolved_archive_file, list):
                    resolved_archive_file = [resolved_archive_file]

                error_msgs = []
                mismatched_keys = []

                if len(resolved_archive_file) > 1:
                    resolved_archive_file = tqdm(resolved_archive_file, desc="Loading checkpoint shards")

                for shard_file in resolved_archive_file:
                    pre_tensor_parallel_split = False
                    state_dict = paddlemix_load(shard_file,map_location="cpu")

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
        if  model_name_or_path is not None:  
            load_blip2_model_state(model_name_or_path,'model')
        else:
            load_blip2_model_state(vision_model_name_or_path,'vision') 
            load_blip2_model_state(qformer_model_name_or_path,'bridge')
      
    @classmethod
    def _resolve_model_file_path(
        cls: Type[PretrainedModel],
        pretrained_model_name_or_path: str,
        name=None,
        from_hf_hub: bool = False,
        cache_dir = None,
        subfolder: str = "",
        config: PretrainedConfig = None,
        convert_from_torch: bool = False,
        use_safetensors = None,
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


        if pretrained_model_name_or_path is not None:
            
            
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            if name == "vision":
                if pretrained_model_name_or_path in VISION_WEIGHTS:
                   is_local_file = os.path.isfile(VISION_WEIGHTS[pretrained_model_name_or_path])
                   pretrained_model_name_or_path=VISION_WEIGHTS[pretrained_model_name_or_path]
                else:
                    is_local = os.path.isdir(pretrained_model_name_or_path)
                    pretrained_model_name_or_path=pretrained_model_name_or_path
            elif name == "bridge":
                if pretrained_model_name_or_path in BRIDGE_WEIGHTS:
                   is_local_file = os.path.isfile(BRIDGE_WEIGHTS[pretrained_model_name_or_path])
                   pretrained_model_name_or_path=BRIDGE_WEIGHTS[pretrained_model_name_or_path]
                else:
                    is_local = os.path.isdir(pretrained_model_name_or_path)
                    pretrained_model_name_or_path=pretrained_model_name_or_path

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
                if os.path.isfile(
                    get_file_path(pretrained_model_name_or_path)
                ):
                    # Load from a PaddlePaddle checkpoint
                    resolved_archive_file = get_file_path(
                        pretrained_model_name_or_path
                    )
            elif is_remote_url(pretrained_model_name_or_path):
                resolved_archive_file = get_path_from_url_with_filelock(pretrained_model_name_or_path)
            else:
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                    " from 'https://paddlenlp.bj.bcebos.com'"
                )

        return resolved_archive_file, sharded_metadata, is_sharded