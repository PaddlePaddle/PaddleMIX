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

import functools
from typing import Optional, Union

import paddle
import paddle.nn as nn

from .state import PartialState
from .utils import send_to_device


class ModelHook:
    """
    A hook that contains callbacks to be executed just before and after the forward method of a model. The difference
    with PyTorch existing hooks is that they get passed along the kwargs.

    Class attribute:
    - **no_grad** (`bool`, *optional*, defaults to `False`) -- Whether or not to execute the actual forward pass under
      the `paddle.no_grad()` context manager.
    """

    no_grad = False

    def init_hook(self, module):
        """
        To be executed when the hook is attached to the module.

        Args:
            module (`nn.Layer`): The module attached to this hook.
        """
        return module

    def pre_forward(self, module, *args, **kwargs):
        """
        To be executed just before the forward method of the model.

        Args:
            module (`nn.Layer`): The module whose forward pass will be executed just after this event.
            args (`Tuple[Any]`): The positional arguments passed to the module.
            kwargs (`Dict[Str, Any]`): The keyword arguments passed to the module.

        Returns:
            `Tuple[Tuple[Any], Dict[Str, Any]]`: A tuple with the treated `args` and `kwargs`.
        """
        return args, kwargs

    def post_forward(self, module, output):
        """
        To be executed just after the forward method of the model.

        Args:
            module (`nn.Layer`): The module whose forward pass been executed just before this event.
            output (`Any`): The output of the module.

        Returns:
            `Any`: The processed `output`.
        """
        return output

    def detach_hook(self, module):
        """
        To be executed when the hook is detached from a module.

        Args:
            module (`nn.Layer`): The module detached from this hook.
        """
        return module


class SequentialHook(ModelHook):
    """
    A hook that can contain several hooks and iterates through them at each event.
    """

    def __init__(self, *hooks):
        self.hooks = hooks

    def init_hook(self, module):
        for hook in self.hooks:
            module = hook.init_hook(module)
        return module

    def pre_forward(self, module, *args, **kwargs):
        for hook in self.hooks:
            args, kwargs = hook.pre_forward(module, *args, **kwargs)
        return args, kwargs

    def post_forward(self, module, output):
        for hook in self.hooks:
            output = hook.post_forward(module, output)
        return output

    def detach_hook(self, module):
        for hook in self.hooks:
            module = hook.detach_hook(module)
        return module


def add_hook_to_module(module: nn.Layer, hook: ModelHook, append: bool = False):
    """
    Adds a hook to a given module. This will rewrite the `forward` method of the module to include the hook, to remove
    this behavior and restore the original `forward` method, use `remove_hook_from_module`.

    <Tip warning={true}>

    If the module already contains a hook, this will replace it with the new hook passed by default. To chain two hooks
    together, pass `append=True`, so it chains the current and new hook into an instance of the `SequentialHook` class.

    </Tip>

    Args:
        module (`nn.Layer`):
            The module to attach a hook to.
        hook (`ModelHook`):
            The hook to attach.
        append (`bool`, *optional*, defaults to `False`):
            Whether the hook should be chained with an existing one (if module already contains a hook) or not.

    Returns:
        `nn.Layer`: The same module, with the hook attached (the module is modified in place, so the result can
        be discarded).
    """

    if append and (getattr(module, "_pp_hook", None) is not None):
        old_hook = module._pp_hook
        remove_hook_from_module(module)
        hook = SequentialHook(old_hook, hook)

    if hasattr(module, "_pp_hook") and hasattr(module, "_old_forward"):
        # If we already put some hook on this module, we replace it with the new one.
        old_forward = module._old_forward
    else:
        old_forward = module.forward
        module._old_forward = old_forward

    module = hook.init_hook(module)
    module._pp_hook = hook

    def new_forward(module, *args, **kwargs):
        args, kwargs = module._pp_hook.pre_forward(module, *args, **kwargs)
        if module._pp_hook.no_grad:
            with paddle.no_grad():
                output = module._old_forward(*args, **kwargs)
        else:
            output = module._old_forward(*args, **kwargs)
        return module._pp_hook.post_forward(module, output)

    module.forward = functools.update_wrapper(functools.partial(new_forward, module), old_forward)

    return module


def remove_hook_from_module(module: nn.Layer, recurse=False):
    """
    Removes any hook attached to a module via `add_hook_to_module`.

    Args:
        module (`nn.Layer`): The module to attach a hook to.
        recurse (`bool`, **optional**): Whether to remove the hooks recursively

    Returns:
        `nn.Layer`: The same module, with the hook detached (the module is modified in place, so the result can
        be discarded).
    """

    if hasattr(module, "_pp_hook"):
        module._pp_hook.detach_hook(module)
        delattr(module, "_pp_hook")

    if hasattr(module, "_old_forward"):
        module.forward = module._old_forward
        delattr(module, "_old_forward")

    if recurse:
        for child in module.children():
            remove_hook_from_module(child, recurse)

    return module


def remove_hook_from_submodules(module: nn.Layer):
    """
    Recursively removes all hooks attached on the submodules of a given model.

    Args:
        module (`nn.Layer`): The module on which to remove all hooks.
    """
    remove_hook_from_module(module)
    for child in module.children():
        remove_hook_from_submodules(child)


class CpuOffload(ModelHook):
    """
    Offloads a model on the CPU until its forward pass is called. The model will not be offloaded back to the CPU after
    the forward, the user needs to call the `init_hook` method again for this.

    Args:
        execution_device(`str` or `int`, *optional*):
            The device on which the model should be executed. Will default to the MPS device if it's available, then
            GPU 0 if there is a GPU, and finally to the CPU.
        prev_module_hook (`UserCpuOffloadHook`, *optional*):
            The hook sent back by [`cpu_offload_with_hook`] for a previous model in the pipeline you are running. If
            passed, its offload method will be called just before the forward of the model to which this hook is
            attached.
    """

    def __init__(
        self,
        execution_device: Optional[Union[str, int]] = None,
        prev_module_hook: Optional["UserCpuOffloadHook"] = None,
    ):
        self.prev_module_hook = prev_module_hook

        self.execution_device = execution_device if execution_device is not None else PartialState().default_device

    def init_hook(self, module):
        return module.to(device="cpu")

    def pre_forward(self, module, *args, **kwargs):
        if self.prev_module_hook is not None:
            self.prev_module_hook.offload()
        module.to(device=self.execution_device)
        return send_to_device(args, self.execution_device), send_to_device(kwargs, self.execution_device)


class UserCpuOffloadHook:
    """
    A simple hook grouping a model and a `ModelHook`, which provides easy APIs for to call the init method of the hook
    or remove it entirely.
    """

    def __init__(self, model, hook):
        self.model = model
        self.hook = hook

    def offload(self):
        self.hook.init_hook(self.model)

    def remove(self):
        remove_hook_from_module(self.model)
