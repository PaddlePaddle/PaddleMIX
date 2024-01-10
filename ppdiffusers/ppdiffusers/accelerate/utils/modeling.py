# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import contextlib

import paddle
import paddle.amp
import paddle.nn as nn

from ..state import AcceleratorState
from .dataclasses import AutocastKwargs, DistributedType

WEIGHTS_INDEX_NAME = "model_state.pdparams.index.json"


from paddlenlp.transformers.model_utils import shard_checkpoint  # noqa: F401


def named_module_tensors(
    module: nn.Layer, include_buffers: bool = True, recurse: bool = False, remove_non_persistent: bool = False
):
    """
    A helper function that gathers all the tensors (parameters + buffers) of a given module. If `include_buffers=True`
    it's the same as doing `module.named_parameters(recurse=recurse) + module.named_buffers(recurse=recurse)`.

    Args:
        module (`paddle.nn.Layer`):
            The module we want the tensors on.
        include_buffer (`bool`, *optional*, defaults to `True`):
            Whether or not to include the buffers in the result.
        recurse (`bool`, *optional`, defaults to `False`):
            Whether or not to go look in every submodule or just return the direct parameters and buffers.
        remove_non_persistent (`bool`, *optional*, defaults to `False`):
            Whether or not to remove the non persistent buffer from the buffers. Useful only when include_buffers =
            True
    """
    for named_parameter in module.named_parameters():
        yield named_parameter

    if include_buffers:
        non_persistent_buffers = set()
        if remove_non_persistent:
            non_persistent_buffers = get_non_persistent_buffers(module)
        for named_buffer in module.named_buffers():
            name, _ = named_buffer
            if name not in non_persistent_buffers:
                yield named_buffer


def get_non_persistent_buffers(module: nn.Layer, recurse: bool = False):
    """
    Gather all non persistent buffers of a given modules into a set

    Args:
        module (`nn.Layer`):
            The module we want the non persistent buffers on.
        recurse (`bool`, *optional*, defaults to `False`):
            Whether or not to go look in every submodule or just return the direct non persistent buffers.
    """

    non_persistent_buffers_set = module._non_persistable_buffer_names_set
    if recurse:
        for _, m in module.named_sublayers(include_self=True):
            non_persistent_buffers_set |= m._non_persistable_buffer_names_set

    return non_persistent_buffers_set


def get_mixed_precision_context_manager(
    native_amp: bool = False, autocast_kwargs: AutocastKwargs = None, fp16_opt_level="O1"
):
    """
    Return a context manager for autocasting mixed precision

    Args:
        native_amp (`bool`, *optional*, defaults to False):
            Whether mixed precision is actually enabled.
        cache_enabled (`bool`, *optional*, defaults to True):
            Whether the weight cache inside autocast should be enabled.
    """
    state = AcceleratorState()
    if autocast_kwargs is None:
        autocast_kwargs = {}
    else:
        autocast_kwargs = autocast_kwargs.to_kwargs()
    if native_amp:
        autocast_kwargs["level"] = fp16_opt_level
        if state.mixed_precision == "fp16":
            autocast_kwargs["dtype"] = "float16"
            return paddle.amp.amp_guard(**autocast_kwargs)
        elif state.mixed_precision == "bf16" and state.distributed_type in [
            DistributedType.NO,
            DistributedType.MULTI_GPU,
        ]:
            autocast_kwargs["dtype"] = "bfloat16"
            return paddle.amp.amp_guard(**autocast_kwargs)
        else:
            return paddle.amp.amp_guard(**autocast_kwargs)
    else:
        return contextlib.nullcontext()
