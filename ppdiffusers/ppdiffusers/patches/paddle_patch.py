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

import numpy as np
import paddle
import paddle.nn as nn

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
    from paddle.nn.functional.flash_attention import flash_attention

    try:
        sdp_kernel = paddle.nn.functional.flash_attention._select_sdp_cuda(128 + 64)
        if sdp_kernel == "mem_efficient":
            flash_attn_version = 1
        else:
            flash_attn_version = 2
    except Exception:
        flash_attn_version = 1

    is_support_flash_attention = True
    flash_attn_error = None
    try:
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
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        training=True,
        attention_op=None,
    ):

        if attention_op in [None, "auto"]:
            head_dim = query.shape[-1]
            attention_op = "cutlass"
            if is_support_flash_attention and query.dtype not in [paddle.float32]:
                if flash_attn_version == 1:
                    if head_dim <= 128:
                        attention_op = "flash"
                else:
                    if head_dim <= 256:
                        attention_op = "flash"
        else:
            if attention_op == "flash" and flash_attn_error is not None:
                raise OSError(flash_attn_error)

        if attention_op == "math" or attn_mask is not None:
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
                    attn_mask = paddle.transpose(attn_mask, [0, 2, 1, 3])
                    if attn_mask.cast("float32").min() == 0 and attn_mask.cast("float32").max() == 1:
                        attn_mask = (attn_mask.cast(s.dtype) - 1) * 10000.0
                    s = s + attn_mask
                p = paddle.nn.functional.softmax(s, axis=-1)
            if dropout_p > 0.0:
                p = paddle.nn.functional.dropout(p, dropout_p, training=training, mode="upscale_in_train")
            o = paddle.matmul(p, vt)
            return paddle.transpose(o, [0, 2, 1, 3])
        elif attention_op == "cutlass":
            if scale is None:
                scale = 1 / math.sqrt(query.shape[-1])
            # support fp32, fp16, bfp16
            query.stop_gradient = False
            key.stop_gradient = False
            value.stop_gradient = False
            output = memory_efficient_attention(
                query,
                key,
                value,
                None,
                p=dropout_p if training else 0.0,
                scale=scale,
                training=True,
            )  # make sure we use training=True
        elif attention_op == "flash":
            output = flash_attention(query, key, value, dropout=dropout_p, causal=is_causal, return_softmax=False)[0]
        else:
            raise ValueError("ppxformers's attention_op shoulde be in ['cutlass', 'flash', 'math']")
        return output

    paddle.nn.functional.scaled_dot_product_attention_ = scaled_dot_product_attention_
