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
import builtins
import contextlib
import copy
import functools
import math
import os
from collections import OrderedDict
from collections import abc as container_abcs
from types import FunctionType, MethodType
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn as nn

try:
    from paddle.base.dygraph.base import param_guard
except ImportError:

    @contextlib.contextmanager
    def param_guard(parameters):
        yield


try:
    from paddle.base.framework import Parameter as ParameterBase
except ImportError:
    from paddle.framework import Parameter as ParameterBase


def str2bool(v):
    if isinstance(v, bool):
        return v
    if not isinstance(v, str):
        v = str(v)
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Not supported value: {}".format(v))


if not hasattr(paddle, "masked_fill"):

    def masked_fill(x, mask, value):
        y = paddle.full(x.shape, value, x.dtype)
        return paddle.where(mask, y, x)

    paddle.masked_fill = masked_fill


if not hasattr(paddle, "finfo"):

    def finfo(dtype: paddle.dtype = None):
        if dtype is None:
            dtype = paddle.get_default_dtype()

        if dtype in [paddle.bfloat16, "bfloat16"]:
            # Numpy do not support `np.finfo(np.uint16)`, so try to construct a finfo object to fetch min value
            class BFloatFInfo:
                min = -3.3895313892515355e38

            return BFloatFInfo
        if dtype in [paddle.float32, "float32"]:
            return np.finfo(np.float32)
        if dtype in [paddle.float16, "float16"]:
            return np.finfo(np.float16)
        if dtype in [paddle.float64, "float64"]:
            return np.finfo(np.float64)

    paddle.finfo = finfo


def is_floating_point(x):
    if not isinstance(x, (paddle.Tensor, paddle.static.Variable)):
        raise TypeError("Expected Tensor, but received type of x: {}".format(type(x)))
    dtype = x.dtype
    is_fp_dtype = (
        dtype == paddle.float32 or dtype == paddle.float64 or dtype == paddle.float16 or dtype == paddle.bfloat16
    )
    return is_fp_dtype


if not hasattr(paddle, "is_floating_point"):
    paddle.is_floating_point = is_floating_point

if not hasattr(paddle.Tensor, "data_ptr"):
    paddle.Tensor.data_ptr = lambda x: x.value().get_tensor()._ptr()

if not hasattr(paddle.Tensor, "data"):
    paddle.Tensor.data = lambda x: x


def permute_pt(x, *perm: builtins.int, name=None):
    return paddle.transpose(x, perm=perm, name=name)


paddle.permute = permute_pt
paddle.Tensor.permute = permute_pt
paddle.Tensor.softmax = nn.functional.softmax

########################################################
if not hasattr(paddle.nn.Layer, "requires_grad_"):

    def requires_grad_(self, value=True):
        for v in self.parameters():
            v.stop_gradient = not value

        return self

    paddle.nn.Layer.requires_grad_ = requires_grad_

if not hasattr(paddle.Tensor, "requires_grad_"):

    def requires_grad_(self, value=True):
        self.stop_gradient = not value
        return self

    paddle.Tensor.requires_grad_ = requires_grad_

if not hasattr(paddle.Tensor, "requires_grad"):

    @property
    def requires_grad_getter(self):
        return not self.stop_gradient

    @requires_grad_getter.setter
    def requires_grad_setter(self, value=True):
        self.stop_gradient = not value
        return self

    paddle.Tensor.requires_grad = requires_grad_getter
    paddle.Tensor.requires_grad = requires_grad_setter
########################################################

# patch repeat_interleave
raw_repeat_interleave = paddle.repeat_interleave


@paddle.jit.not_to_static
def repeat_interleave(x, repeats, axis=None, name=None):
    fp16 = False
    if x.dtype == paddle.float16:
        x = x.cast(paddle.float32)
        fp16 = True

    out = raw_repeat_interleave(x, repeats=repeats, axis=axis, name=name)

    if fp16:
        out = out.cast(paddle.float16)
    return out


paddle.repeat_interleave = repeat_interleave
paddle.Tensor.repeat_interleave = repeat_interleave

# patch max
raw_max = paddle.max


@paddle.jit.not_to_static
def max(x, axis=None, keepdim=False, name=None):
    fp16 = False
    if x.dtype == paddle.float16:
        x = x.cast(paddle.float32)
        fp16 = True

    out = raw_max(x, axis=axis, keepdim=keepdim, name=name)

    if fp16:
        out = out.cast(paddle.float16)
    return out


paddle.max = max
paddle.Tensor.max = max

# patch gather_nd support bfloat16
raw_gather_nd = paddle.gather_nd


@paddle.jit.not_to_static
def gather_nd(x, index, name=None):
    out = raw_gather_nd(x, index=index, name=name)
    return out


paddle.gather_nd = gather_nd
paddle.Tensor.gather_nd = gather_nd
if not hasattr(paddle.Tensor, "contiguous"):
    paddle.Tensor.contiguous = lambda x: x


def eval(self):
    # Layer-level setting
    self.training = False
    for layer in self.sublayers():
        layer.training = False
    return self


nn.Layer.eval = eval


def in_features(self):
    return self.weight.shape[0]


def out_features(self):
    return self.weight.shape[1]


nn.Linear.in_features = property(in_features)
nn.Linear.out_features = property(out_features)


def Parameter(data: paddle.Tensor, requires_grad=True):
    tensor = paddle.create_parameter(data.shape, dtype=data.dtype, default_initializer=nn.initializer.Assign(data))
    if not requires_grad:
        tensor.stop_gradient = True
    return tensor


nn.Parameter = Parameter

if not hasattr(nn, "TorchLinear"):

    class TorchLinear(nn.Layer):
        """
        Same as paddle.layer.Linear, except weight matrix is stored as [out_features, in_features] (same as torch),
        instead of [in_features, out_features]
        """

        def __init__(
            self,
            in_features,
            out_features,
            weight_attr=None,
            bias_attr=None,
            name=None,
            bias=None,
        ):
            super().__init__()
            self._dtype = self._helper.get_default_dtype()
            self._weight_attr = weight_attr
            if bias is not None:
                bias_attr = bias
            self._bias_attr = bias_attr
            self.in_features = self._in_features = in_features
            self.out_features = self.out_features = out_features
            self.weight = self.create_parameter(
                shape=[out_features, in_features],  # regular linear has shape [in_features, out_features]
                attr=self._weight_attr,
                dtype=self._dtype,
                is_bias=False,
            )
            self.bias = self.create_parameter(
                shape=[out_features],
                attr=self._bias_attr,
                dtype=self._dtype,
                is_bias=True,
            )
            self.name = name

        def forward(self, input):
            out = paddle.nn.functional.linear(x=input, weight=self.weight.T, bias=self.bias, name=self.name)
            return out

        def extra_repr(self):
            name_str = ", name={}".format(self.name) if self.name else ""
            return "in_features={}, out_features={}, dtype={}{}".format(
                self.weight.shape[1], self.weight.shape[0], self._dtype, name_str
            )

    nn.TorchLinear = TorchLinear


@contextlib.contextmanager
def device_scope(device="cpu"):
    new_device = device.replace("cuda", "gpu")
    old_device = paddle.get_device()
    try:
        paddle.set_device(new_device)
        yield
    finally:
        paddle.set_device(old_device)


@contextlib.contextmanager
def requires_grad_and_without_random(*tensors, stop_gradient=False):
    raw_rng_state = paddle.get_cuda_rng_state()
    raw_stop_gradient = [each_tensor.stop_gradient for each_tensor in tensors]
    need_switch_stop_gradient = len(set(raw_stop_gradient)) > 1
    if need_switch_stop_gradient:
        for each_tensor in tensors:
            each_tensor.stop_gradient = stop_gradient
    yield
    if need_switch_stop_gradient:
        for index, each_tensor in enumerate(tensors):
            each_tensor.stop_gradient = raw_stop_gradient[index]
    paddle.set_cuda_rng_state(raw_rng_state)


paddle.device_scope = device_scope

if not hasattr(nn.Layer, "get_sublayer"):

    def get_sublayer(self, target: str):
        if target == "":
            return self

        atoms = target.split(".")
        mod: nn.Layer = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(mod.__class__.__name__ + " has no " "attribute `" + item + "`")

            mod = getattr(mod, item)

            if not isinstance(mod, nn.Layer):
                raise AttributeError("`" + item + "` is not " "an nn.Layer")
        return mod

    nn.Layer.get_sublayer = get_sublayer


def to(self=None, device=None, dtype=None, blocking=None):
    return self._to_impl(
        device=device,
        dtype=dtype,
        blocking=blocking,
        include_sublayers=True,
        floating_only=True,
    )


nn.Layer.to = to

from ..utils.import_utils import is_ppxformers_available, is_npu_available

if is_npu_available():
    for lib in os.listdir(os.getenv("CUSTOM_DEVICE_ROOT")):
        if lib.endswith(".so"):
            paddle.utils.cpp_extension.extension_utils.load_op_meta_info_and_register_op(
                lib
            )
    from paddle.base import core
    def scaled_dot_product_attention_npu(query,
                                         key,
                                         value,
                                         attn_mask=None,
                                         dropout_p=0.0,
                                         is_causal=False,
                                         training=True,
                                         name=None,
                                         fixed_seed_offset=None,
                                         return_softmax=False,
                                         is_triangle_upper_mask=True,
                                         ):
        out = core.eager._run_custom_op(
            "flash_attention_npu",
            query,
            key,
            value,
            fixed_seed_offset,
            attn_mask,
            dropout_p,
            is_causal,
            return_softmax,
            not training,
            is_triangle_upper_mask,
        )[0]
        return out
    paddle.nn.functional.scaled_dot_product_attention_npu = scaled_dot_product_attention_npu

if is_ppxformers_available() or is_npu_available():
    from paddle.incubate.nn.memory_efficient_attention import memory_efficient_attention

    try:
        from paddle.incubate.nn.functional import (
            variable_length_memory_efficient_attention,
        )
    except ImportError:
        variable_length_memory_efficient_attention = None

    is_support_flash_attention = True
    flash_attn_error = None
    try:
        _ = paddle.nn.functional.scaled_dot_product_attention(
            paddle.ones((1, 1, 2, 40), dtype=paddle.float16),
            paddle.ones((1, 1, 2, 40), dtype=paddle.float16),
            paddle.ones((1, 1, 2, 40), dtype=paddle.float16),
            attn_mask=paddle.ones((1, 2, 1, 1), dtype=paddle.float16),
        )
        
        from paddle.nn.functional.flash_attention import flash_attention
        _ = flash_attention(
            paddle.ones((1, 1, 2, 40), dtype=paddle.float16),
            paddle.ones((1, 1, 2, 40), dtype=paddle.float16),
            paddle.ones((1, 1, 2, 40), dtype=paddle.float16),
        )
    except Exception as error:
        flash_attn_error = error
        is_support_flash_attention = False

    def scaled_dot_product_attention_(
        query,
        key,
        value,
        attn_mask=None,  # shape [bs, num_heads, query_len, key_len]
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        training=True,
        attention_op=None,
    ):

        if attention_op in [None, "auto"]:
            attention_op = "cutlass"
            if is_support_flash_attention and query.dtype not in [paddle.float32]:
                attention_op = "flash"
            elif is_npu_available() and query.dtype not in [paddle.float32]:
                attention_op = "flash_npu"
        else:
            if attention_op == "flash" and flash_attn_error is not None:
                raise OSError(flash_attn_error)

        if str2bool(os.getenv("FLAGS_cudnn_deterministic", "no")) or str2bool(
            os.getenv("FLAGS_sdpa_select_math", "no")
        ):
            if attention_op == "flash":
                if paddle.nn.functional.flash_attention._select_sdp(query.shape[3]) == "mem_efficient":
                    attention_op = "math"
            else:
                attention_op = "math"

        if attention_op == "math":
            if scale is None:
                scale = 1 / math.sqrt(query.shape[-1])
            qt = paddle.transpose(query, [0, 2, 1, 3])
            kt = paddle.transpose(key, [0, 2, 1, 3])
            vt = paddle.transpose(value, [0, 2, 1, 3])
            s = paddle.matmul(qt * scale, kt, transpose_y=True)
            if is_causal:
                p = paddle.incubate.softmax_mask_fuse_upper_triangle(s)
            else:
                if attn_mask is not None:
                    s = s + attn_mask.cast(s.dtype)
                p = paddle.nn.functional.softmax(s, axis=-1)
            if dropout_p > 0.0:
                p = paddle.nn.functional.dropout(p, dropout_p, training=training, mode="upscale_in_train")
            o = paddle.matmul(p, vt)
            return paddle.transpose(o, [0, 2, 1, 3])
        elif attention_op in ["cutlass", "memory_efficient"]:
            if scale is None:
                scale = 1 / math.sqrt(query.shape[-1])
            # (1) attn_mask is not None, use cutlass v2
            # (2) FLAG_USE_CUTLASS_V2 in yes, y, true, t, 1, use cutlass v2
            use_cutlass_v2 = attn_mask is not None or str2bool(os.getenv("FLAG_USE_CUTLASS_V2", "no"))
            if not use_cutlass_v2:
                with requires_grad_and_without_random(query, key, value, stop_gradient=False):
                    output = memory_efficient_attention(
                        query,
                        key,
                        value,
                        None,
                        p=dropout_p if training else 0.0,
                        scale=scale,
                        training=True,
                    )  # make sure we use training=True
            else:
                assert (
                    variable_length_memory_efficient_attention is not None
                ), "Please upgrade your `paddlepaddle>=2.6.0` to support `variable_length_memory_efficient_attention`."
                batch_size, query_seq_len = query.shape[:2]
                kv_seqlen = key.shape[1]
                output = variable_length_memory_efficient_attention(
                    query.transpose([0, 2, 1, 3]),
                    key.transpose([0, 2, 1, 3]),
                    value.transpose([0, 2, 1, 3]),
                    seq_lens=paddle.to_tensor(
                        [query_seq_len] * batch_size,
                        dtype="int32",
                    ),
                    kv_seq_lens=paddle.to_tensor(
                        [kv_seqlen] * batch_size,
                        dtype="int32",
                    ),
                    mask=None if is_causal else attn_mask,
                    scale=scale,
                    causal=bool(is_causal),
                    pre_cache_length=0,
                ).transpose([0, 2, 1, 3])
        elif attention_op == "flash":
            with requires_grad_and_without_random(query, key, value, stop_gradient=False):
                output = paddle.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=None if is_causal else attn_mask,
                    dropout_p=dropout_p if training else 0.0,
                    is_causal=bool(is_causal),
                    training=training,
                )
        elif attention_op == "flash_npu":
            output = paddle.nn.functional.scaled_dot_product_attention_npu(
                query,
                key,
                value,
                attn_mask=None if is_causal else attn_mask,
                dropout_p=dropout_p if training else 0.0,
                is_causal=bool(is_causal),
                training=training,
            )
        else:
            raise ValueError(
                "ppxformers's attention_op should be in ['auto', 'math', 'cutlass', `memory_efficient`, 'flash']."
            )
        return output

    paddle.nn.functional.scaled_dot_product_attention_ = scaled_dot_product_attention_


if not hasattr(nn, "ParameterDict"):

    def typename(o):
        if isinstance(o, paddle.Tensor):
            dtype = str(o.dtype).replace("paddle.", "")
            return f"paddle.Tensor {dtype}"

        module = ""
        class_name = ""
        if (
            hasattr(o, "__module__")
            and o.__module__ != "builtins"
            and o.__module__ != "__builtin__"
            and o.__module__ is not None
        ):
            module = o.__module__ + "."

        if hasattr(o, "__qualname__"):
            class_name = o.__qualname__
        elif hasattr(o, "__name__"):
            class_name = o.__name__
        else:
            class_name = o.__class__.__name__

        return module + class_name

    class ParameterDict(nn.Layer):
        r"""Holds parameters in a dictionary.

        ParameterDict can be indexed like a regular Python dictionary, but Parameters it
        contains are properly registered, and will be visible by all Module methods.
        Other objects are treated as would be done by a regular Python dictionary

        :class:`~paddle.nn.ParameterDict` is an **ordered** dictionary.
        :meth:`~paddle.nn.ParameterDict.update` with other unordered mapping
        types (e.g., Python's plain ``dict``) does not preserve the order of the
        merged mapping. On the other hand, ``OrderedDict`` or another :class:`~paddle.nn.ParameterDict`
        will preserve their ordering.

        Note that the constructor, assigning an element of the dictionary and the
        :meth:`~paddle.nn.ParameterDict.update` method will convert any :class:`~paddle.Tensor` into
        :class:`~paddle.nn.Parameter`.

        Args:
            values (iterable, optional): a mapping (dictionary) of
                (string : Any) or an iterable of key-value pairs
                of type (string, Any)

        Example::

            class MyModule(nn.Layer):
                def __init__(self):
                    super().__init__()
                    self.params = nn.ParameterDict({
                            'left': nn.Parameter(paddle.randn([5, 10])),
                            'right': nn.Parameter(paddle.randn([5, 10]))
                    })

                def forward(self, x, choice):
                    x = self.params[choice].mm(x)
                    return x
        """

        def __init__(self, parameters: Any = None) -> None:
            super().__init__()
            self._keys: Dict[str, None] = {}
            if parameters is not None:
                self.update(parameters)

        def _key_to_attr(self, key: str) -> str:
            if not isinstance(key, str):
                raise TypeError(
                    "Index given to ParameterDict cannot be used as a key as it is "
                    f"not a string (type is '{type(key).__name__}'). Open an issue on "
                    "github if you need non-string keys."
                )
            else:
                # Use the key as-is so that `.named_parameters()` returns the right thing
                return key

        def __getitem__(self, key: str) -> Any:
            with param_guard(self._parameters):
                attr = self._key_to_attr(key)
                return getattr(self, attr)

        def __setitem__(self, key: str, value: Any) -> None:
            # Note that all other function that add an entry to the dictionary part of
            # the ParameterDict end up here. So this is the only place where we need
            # to wrap things into Parameter if needed.
            # Objects added via setattr() are not in the dictionary part and thus won't
            # call into this function.
            self._keys[key] = None
            attr = self._key_to_attr(key)
            if isinstance(value, paddle.Tensor) and not isinstance(value, ParameterBase):
                value = paddle.create_parameter(
                    value.shape, dtype=value.dtype, default_initializer=nn.initializer.Assign(value)
                )
            setattr(self, attr, value)

        def __delitem__(self, key: str) -> None:
            del self._keys[key]
            attr = self._key_to_attr(key)
            delattr(self, attr)

        def __len__(self) -> int:
            return len(self._keys)

        def __iter__(self) -> Iterator[str]:
            with param_guard(self._parameters):
                return iter(self._keys)

        def __reversed__(self) -> Iterator[str]:
            return reversed(list(self._keys))

        def copy(self) -> "ParameterDict":
            """Returns a copy of this :class:`~paddle.nn.ParameterDict` instance."""
            # We have to use an OrderedDict because the ParameterDict constructor
            # behaves differently on plain dict vs OrderedDict
            return ParameterDict(OrderedDict((k, self[k]) for k in self._keys))

        def __contains__(self, key: str) -> bool:
            return key in self._keys

        def setdefault(self, key: str, default: Optional[Any] = None) -> Any:
            """If key is in the ParameterDict, return its value.
            If not, insert `key` with a parameter `default` and return `default`.
            `default` defaults to `None`.

            Args:
                key (str): key to set default for
                default (Any): the parameter set to the key
            """

            if key not in self:
                self[key] = default
            return self[key]

        def clear(self) -> None:
            """Remove all items from the ParameterDict."""
            for k in self._keys.copy():
                del self[k]

        def pop(self, key: str) -> Any:
            r"""Remove key from the ParameterDict and return its parameter.

            Args:
                key (str): key to pop from the ParameterDict
            """
            v = self[key]
            del self[key]
            return v

        def popitem(self) -> Tuple[str, Any]:
            """Remove and return the last inserted `(key, parameter)` pair
            from the ParameterDict
            """
            k, _ = self._keys.popitem()
            # We need the key in the _keys to be able to access/del
            self._keys[k] = None
            val = self[k]
            del self[k]
            return k, val

        def get(self, key: str, default: Optional[Any] = None) -> Any:
            r"""Return the parameter associated with key if present.
            Otherwise return default if provided, None if not.

            Args:
                key (str): key to get from the ParameterDict
                default (Parameter, optional): value to return if key not present
            """
            return self[key] if key in self else default

        def fromkeys(self, keys: Iterable[str], default: Optional[Any] = None) -> "ParameterDict":
            r"""Return a new ParameterDict with the keys provided

            Args:
                keys (iterable, string): keys to make the new ParameterDict from
                default (Parameter, optional): value to set for all keys
            """
            return ParameterDict(((k, default) for k in keys))

        def keys(self) -> Iterable[str]:
            r"""Return an iterable of the ParameterDict keys."""
            return self._keys.keys()

        def items(self) -> Iterable[Tuple[str, Any]]:
            r"""Return an iterable of the ParameterDict key/value pairs."""
            return ((k, self[k]) for k in self._keys)

        def values(self) -> Iterable[Any]:
            r"""Return an iterable of the ParameterDict values."""
            return (self[k] for k in self._keys)

        def update(self, parameters: Union[Mapping[str, Any], "ParameterDict"]) -> None:
            r"""Update the :class:`~paddle.nn.ParameterDict` with the key-value pairs from a
            mapping or an iterable, overwriting existing keys.

            .. note::
                If :attr:`parameters` is an ``OrderedDict``, a :class:`~paddle.nn.ParameterDict`, or
                an iterable of key-value pairs, the order of new elements in it is preserved.

            Args:
                parameters (iterable): a mapping (dictionary) from string to
                    :class:`~paddle.nn.Parameter`, or an iterable of
                    key-value pairs of type (string, :class:`~paddle.nn.Parameter`)
            """
            if not isinstance(parameters, container_abcs.Iterable):
                raise TypeError(
                    "ParametersDict.update should be called with an "
                    "iterable of key/value pairs, but got " + type(parameters).__name__
                )

            if isinstance(parameters, (OrderedDict, ParameterDict)):
                for key, parameter in parameters.items():
                    self[key] = parameter
            elif isinstance(parameters, container_abcs.Mapping):
                for key, parameter in sorted(parameters.items()):
                    self[key] = parameter
            else:
                for j, p in enumerate(parameters):
                    if not isinstance(p, container_abcs.Iterable):
                        raise TypeError(
                            "ParameterDict update sequence element "
                            "#" + str(j) + " should be Iterable; is" + type(p).__name__
                        )
                    if not len(p) == 2:
                        raise ValueError(
                            "ParameterDict update sequence element "
                            "#" + str(j) + " has length " + str(len(p)) + "; 2 is required"
                        )
                    # parameters as length-2 list too cumbersome to type, see ModuleDict.update comment
                    self[p[0]] = p[1]  # type: ignore[assignment]

        def extra_repr(self) -> str:
            child_lines = []
            for k, p in self.items():
                if isinstance(p, paddle.Tensor):
                    size_str = "x".join(str(size) for size in p.shape)
                    if size_str == "":
                        size_str = "0"
                    device_str = "" if not p.place.is_gpu_place() else " (gpu:{})".format(p.place.gpu_device_id())
                    parastr = "{} containing: [{} of shape {}{}]".format(
                        "Parameter" if isinstance(p, ParameterBase) else "Tensor", typename(p), size_str, device_str
                    )
                    child_lines.append("(" + str(k) + "): " + parastr)
                else:
                    child_lines.append("(" + str(k) + "): Object of type: " + type(p).__name__)
            tmpstr = "\n".join(child_lines)
            return tmpstr

        def __call__(self, input):
            raise RuntimeError("ParameterDict should not be called.")

        def __or__(self, other: "ParameterDict") -> "ParameterDict":
            copy = self.copy()
            copy.update(other)
            return copy

        def __ror__(self, other: "ParameterDict") -> "ParameterDict":
            copy = other.copy()
            copy.update(self)
            return copy

        def __ior__(self, other: "ParameterDict") -> "ParameterDict":
            self.update(other)
            return self

    nn.ParameterDict = ParameterDict


def copy_func(f):
    "Copy a non-builtin function (NB `copy.copy` does not work for this)"
    if not isinstance(f, FunctionType):
        return copy.copy(f)
    fn = FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
    fn.__kwdefaults__ = f.__kwdefaults__
    fn.__dict__.update(f.__dict__)
    fn.__annotations__.update(f.__annotations__)
    fn.__qualname__ = f.__qualname__
    return fn


class _clsmethod:
    def __init__(self, f):
        self.f = f

    def __get__(self, _, f_cls):
        return MethodType(self.f, f_cls)


# copied from https://github.com/fastai/fastcore/blob/c9b4c088d3706569c076e7c197c724730be190ab/fastcore/basics.py#L938-L954
def patch_to(cls, as_prop=False, cls_method=False):
    "Decorator: add `f` to `cls`"
    if not isinstance(cls, (tuple, list)):
        cls = (cls,)

    def _inner(f):
        for c_ in cls:
            nf = copy_func(f)
            nm = f.__name__
            # `functools.update_wrapper` when passing patched function to `Pipeline`, so we do it manually
            for o in functools.WRAPPER_ASSIGNMENTS:
                setattr(nf, o, getattr(f, o))
            nf.__qualname__ = f"{c_.__name__}.{nm}"
            if cls_method:
                # fix https://github.com/fastai/fastcore/issues/510
                setattr(c_, nm, _clsmethod(nf))
            else:
                setattr(c_, nm, property(nf) if as_prop else nf)
        # Avoid clobbering existing functions
        return globals().get(nm, builtins.__dict__.get(nm, None))

    return _inner
