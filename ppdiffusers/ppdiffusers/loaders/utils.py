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

import functools
import weakref
from collections import OrderedDict

import paddle
import paddle.nn as nn

try:
    from paddle.nn.layer.layers import HookRemoveHelper
except ImportError:
    from paddle.fluid.dygraph.layers import HookRemoveHelper

from typing import Any, Callable, Dict, Optional


class _WrappedHook:
    def __init__(self, hook: Callable, module: Optional["nn.Layer"] = None):
        self.hook: Callable = hook
        functools.update_wrapper(self, hook)

        self.with_module: bool = False

        if module is not None:
            self.module: weakref.ReferenceType["nn.Layer"] = weakref.ref(module)
            self.with_module = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.with_module:
            module = self.module()
            if module is None:
                raise RuntimeError("You are trying to call the hook of a dead Module!")
            return self.hook(module, *args, **kwargs)
        return self.hook(*args, **kwargs)

    def __getstate__(self) -> Dict:
        result = {"hook": self.hook, "with_module": self.with_module}
        if self.with_module:
            result["module"] = self.module()

        return result

    def __setstate__(self, state: Dict):
        self.hook = state["hook"]
        self.with_module = state["with_module"]

        if self.with_module:
            if state["module"] is None:
                raise RuntimeError("You are trying to revive the hook of a dead Module!")
            self.module = weakref.ref(state["module"])


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

    def register_load_state_dict_pre_hook(self, hook, with_module=False):
        if getattr(self, "load_state_dict_pre_hooks", None) is None:
            self.load_state_dict_pre_hooks = OrderedDict()
        handle = HookRemoveHelper(self.load_state_dict_pre_hooks)
        self.load_state_dict_pre_hooks[handle._hook_id] = _WrappedHook(hook, self if with_module else None)
        return handle

    def set_state_dict(self, state_dict, use_structured_name: bool = True):
        if getattr(self, "load_state_dict_pre_hooks", None) is not None:
            for hook in self.load_state_dict_pre_hooks.values():
                hook(state_dict)
        return super().set_state_dict(self, state_dict, use_structured_name=use_structured_name)

    load_dict = set_state_dict
    set_dict = set_state_dict
