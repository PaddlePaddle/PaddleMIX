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
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from paddlenlp.transformers.tokenizer_utils_base import AddedToken, TextInput
from tqdm.auto import tqdm

from paddlemix.utils import device_guard, paddlemix_load
from paddlemix.utils.env import MODEL_HOME
from paddlemix.utils.log import logger

__all__ = [
    "MixPretrainedModel",
    "MIXPretrainedTokenizer",
    "NPUCrossEntropyLoss",
]


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
                    "convert_state_dict_dtype expected paddle.Tensor not numpy.ndarray, please convert numpy.ndarray to paddle.Tensor"
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
    `MixPretrainedModel` increases `load_pretrained` method to support loading pretrained weights
    in different stages after construction. The other methods are the same as class
    `paddlenlp.transformers.model_utils.PretrainedModel`.
    """

    def __init__(self, *args, **kwargs):
        super(MixPretrainedModel, self).__init__(*args, **kwargs)

    def get_expected_keys(self, model_state_dict):
        # override when model needs to load different pretrain model at different stages, such as BLIP-2
        return list(model_state_dict.keys())

    def refine_state_dict(self, state_dict):
        # preprocess the weight loaded here, such as interpolation
        pass

    def load_pretrained(
        self,
        pretrained_model_name_or_path=None,
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
            low_cpu_mem_usage (bool, optional): whether use low cpu memory usage for loading pretrained model。 Defaults to False.
            dtype (_type_, optional): the dtype of model state dict. Defaults to None.
            cache_dir (str, optional): the cache directory for loading pretrained model. Defaults to None.
            subfolder (str, optional): the subfolder of pretrained model name. Defaults "".
            variant (str, optional): the pretrained model variant. Defaults to None.

        Returns:
            Tuple[List[str]]: _description_
        """
        if pretrained_model_name_or_path is None and state_dict is None:
            ValueError("Either pretrained_model_name_or_path or state_dict should be set.")

        cache_dir = resolve_cache_dir(pretrained_model_name_or_path, cache_dir)

        # Keep in fp32 modules
        keep_in_fp32_modules = None
        use_keep_in_fp32_modules = False

        # resolve model_weight file
        resolved_archive_file, sharded_metadata, is_sharded = self._resolve_model_file_path(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            subfolder=subfolder,
            variant=variant,
        )

        if not is_sharded and state_dict is None:
            # Time to load the checkpoint
            # loading non-sharded ckpt from the state dict
            if self.config.tensor_parallel_degree > 1 and resolved_archive_file.endswith("model_state.pdparams"):
                state_dict = self.convert_tensor_parallel(resolved_archive_file, self.config)
            else:
                state_dict = paddlemix_load(resolved_archive_file)

            self.refine_state_dict(state_dict)

            logger.info("Loaded weights file from disk, setting weights to model.")

        # Check if `_keep_in_fp32_modules` is not None
        use_keep_in_fp32_modules = (self._keep_in_fp32_modules is not None) and dtype == "float16"

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

        if use_keep_in_fp32_modules:
            # low_cpu_mem_usage = True
            keep_in_fp32_modules = self._keep_in_fp32_modules
        else:
            keep_in_fp32_modules = []

        # load_pretrained_model
        is_safetensors = False

        model_state_dict = self.state_dict()

        expected_keys = self.get_expected_keys(model_state_dict)

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
            # DONT Hold tensor parallel here, only hold after load state dict.
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
                state_dict = paddlemix_load(shard_file)

                # Mismatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
                # matching the weights in the model.
                mismatched_keys += _find_mismatched_keys(
                    state_dict,
                    model_state_dict,
                    loaded_state_dict_keys,
                    add_prefix_to_model,
                    remove_prefix_from_model,
                    ignore_mismatched_sizes,
                )

                if (
                    self.config.tensor_parallel_degree > 1
                    and ".tp" not in shard_file
                    and not pre_tensor_parallel_split
                ):
                    logger.info("Converting state_dict to Tensor Parallel Format")
                    # ignore error for multi shard, since only parts of data
                    state_dict = self.convert_tensor_parallel(
                        None, self.config, state_dict=state_dict, ignore_error=len(resolved_archive_file) > 1
                    )
                    logger.info("Converted state_dict to Tensor Parallel Format")

                if low_cpu_mem_usage:
                    new_error_msgs = _load_state_dict_into_meta_model(
                        model_to_load,
                        state_dict,
                        loaded_state_dict_keys,
                        start_prefix,
                        expected_keys,
                        dtype=dtype,
                        is_safetensors=is_safetensors,
                        keep_in_fp32_modules=keep_in_fp32_modules,
                    )
                    error_msgs += new_error_msgs
                else:
                    error_msgs += _load_state_dict_into_model(model_to_load, state_dict, start_prefix)

                # force memory release
                del state_dict
                gc.collect()

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if " but the expected shape is" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            logger.warning(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
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
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {self.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
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
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        return missing_keys, unexpected_keys, mismatched_keys


class MIXPretrainedTokenizer(PretrainedTokenizer):
    """
    The base class for all pretrained models used in PaddleMIX.

    The most difference between `PretrainedTokenizer` and `MIXPretrainedTokenizer` is that
    `MIXPretrainedTokenizer` tokenize has '/n' same as PyTorch. The other methods are the same as class
    `paddlenlp.transformers.tokenizer_utils.PretrainedTokenizer`.
    """

    def __init__(self, *args, **kwargs):
        super(MIXPretrainedTokenizer, self).__init__(*args, **kwargs)

    def tokenize(self, text: TextInput, **kwargs) -> List[str]:
        """
        Converts a string in a sequence of tokens, using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific `prepare_for_tokenization` preprocessing method.

        Returns:
            `List[str]`: The list of tokens.
        """

        split_special_tokens = kwargs.pop("split_special_tokens", self.split_special_tokens)

        # Simple mapping string => AddedToken for special tokens with specific tokenization behaviors
        all_special_tokens_extended = dict(
            (str(t), t) for t in self.all_special_tokens_extended if isinstance(t, AddedToken)
        )

        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        # TODO: should this be in the base class?
        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase
            escaped_special_toks = [
                re.escape(s_tok) for s_tok in (self.unique_no_split_tokens + self.all_special_tokens)
            ]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)

        if split_special_tokens:
            no_split_token = []
            tokens = [text]
        else:
            no_split_token = set(self.unique_no_split_tokens)  # don't split on any of the added tokens
            # "This is something<special_token_1>  else"
            tokens = self.tokens_trie.split(text)

        # ["This is something", "<special_token_1>", "  else"]
        for i, token in enumerate(tokens):
            if token in no_split_token:
                tok_extended = all_special_tokens_extended.get(token, None)
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.rstrip and right:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        tokens[i + 1] = right.lstrip()
                    # Strip white spaces on the left
                    if tok_extended.lstrip and left:
                        tokens[i - 1] = left.rstrip()  # Opposite here
                # else:
                #     # We strip left and right by default
                #     if right:
                #         tokens[i + 1] = right.lstrip()
                #     if left:
                #         tokens[i - 1] = left.rstrip()
        # ["This is something", "<special_token_1>", "else"]
        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text


class NPUCrossEntropyLoss(paddle.nn.Layer):
    """
    Make cross_entropy_loss compatible with npu device
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.reduction = kwargs.get("reduction", "mean")
        kwargs["reduction"] = "none"
        self.nll_loss = paddle.nn.NLLLoss(**kwargs)
        self.log_softmax = paddle.nn.functional.log_softmax

    def forward(self, logits, labels):
        loss = self.nll_loss(self.log_softmax(logits, axis=-1), labels)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unexcepted reduction method: {self.reduction}")
