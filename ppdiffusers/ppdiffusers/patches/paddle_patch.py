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
import math
import os

import numpy as np
import paddle
import paddle.nn as nn


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

    paddle.nn.Layer.requires_grad_ = requires_grad_

if not hasattr(paddle.Tensor, "requires_grad_"):

    def requires_grad_(self, value=True):
        self.stop_gradient = not value

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
    bfp16 = False
    if x.dtype == paddle.bfloat16:
        x = x.cast(paddle.float16)
        bfp16 = True

    out = raw_gather_nd(x, index=index, name=name)

    if bfp16:
        out = out.cast(paddle.bfloat16)
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

import contextlib


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
def requires_grad_and_without_random(*tensors, seed=0, stop_gradient=False):
    raw_rng_state = paddle.get_cuda_rng_state()
    paddle.seed(seed)
    raw_stop_gradient = [each_tensor.stop_gradient for each_tensor in tensors]
    need_switch_stop_gradient = False in raw_stop_gradient
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

from ..utils.import_utils import is_ppxformers_available

if is_ppxformers_available():
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
        else:
            if attention_op == "flash" and flash_attn_error is not None:
                raise OSError(flash_attn_error)

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
                with requires_grad_and_without_random(query, key, value):
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
            with requires_grad_and_without_random(query, key, value):
                output = paddle.nn.functional.scaled_dot_product_attention(
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
                "ppxformers's attention_op shoulde be in ['auto', 'math', 'cutlass', `memory_efficient`, 'flash']."
            )
        return output

    paddle.nn.functional.scaled_dot_product_attention_ = scaled_dot_product_attention_
