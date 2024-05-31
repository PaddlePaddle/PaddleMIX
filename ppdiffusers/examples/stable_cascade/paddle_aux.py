# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


def repeat(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.tile(self, args[0])
        else:
            return paddle.tile(self, list(args))
    elif kwargs:
        assert "repeats" in kwargs
        return paddle.tile(self, repeat_times=kwargs["repeats"])


setattr(paddle.Tensor, "repeat", repeat)


def to(self, *args, **kwargs):
    args_list = ["x", "y", "non_blocking", "copy", "memory_format"]
    new_kwargs = {}
    for i, node in enumerate(args):
        k = args_list[i]
        new_kwargs[k] = node
    for node in kwargs:
        v = kwargs[node]
        new_kwargs[node] = v
    kwargs = new_kwargs
    if not kwargs:
        return self
    elif "tensor" in kwargs:
        return paddle.cast(self, "{}.dtype".format(kwargs["tensor"]))
    elif "dtype" in kwargs:
        return paddle.cast(self, "{}".format(kwargs["dtype"]))
    elif "device" in kwargs and "dtype" not in kwargs:
        return self
    elif kwargs:
        if "y" not in kwargs and "x" in kwargs:
            if isinstance(kwargs["x"], paddle.dtype):
                dtype = kwargs["x"]
            elif isinstance(kwargs["x"], str) and kwargs["x"] not in [
                "cpu",
                "cuda",
                "ipu",
                "xpu",
            ]:
                dtype = kwargs["x"]
            elif isinstance(kwargs["x"], paddle.Tensor):
                dtype = kwargs["x"].dtype
            else:
                dtype = self.dtype
            return paddle.cast(self, dtype)

        elif "y" in kwargs and "x" in kwargs:
            if isinstance(kwargs["x"], paddle.dtype):
                dtype = kwargs["x"]
            elif isinstance(kwargs["x"], str):
                x = kwargs["x"]
                if x not in ["cpu", "cuda", "ipu", "xpu"]:
                    dtype = kwargs["x"]
                else:
                    dtype = kwargs["y"] if isinstance(kwargs["y"], str) else self.dtype
            else:
                dtype = kwargs["x"]
            return paddle.cast(self, dtype)
        else:
            return self


setattr(paddle.Tensor, "to", to)


def mul(self, *args, **kwargs):
    if "other" in kwargs:
        y = kwargs["other"]
    elif "y" in kwargs:
        y = kwargs["y"]
    else:
        y = args[0]

    if not isinstance(y, paddle.Tensor):
        y = paddle.to_tensor(y)

    return paddle.multiply(self, y)


setattr(paddle.Tensor, "mul", mul)
setattr(paddle.Tensor, "multiply", mul)


def div(self, *args, **kwargs):
    if "other" in kwargs:
        y = kwargs["other"]
    elif "y" in kwargs:
        y = kwargs["y"]
    else:
        y = args[0]

    if not isinstance(y, paddle.Tensor):
        y = paddle.to_tensor(y)

    res = paddle.divide(self, y)

    if "rounding_mode" in kwargs:
        rounding_mode = kwargs["rounding_mode"]
        if rounding_mode == "trunc":
            res = paddle.trunc(res)
        elif rounding_mode == "floor":
            res = paddle.floor(res)

    return res


setattr(paddle.Tensor, "div", div)
setattr(paddle.Tensor, "divide", div)


def reshape(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return paddle.reshape(self, args[0])
        else:
            return paddle.reshape(self, list(args))
    elif kwargs:
        assert "shape" in kwargs
        return paddle.reshape(self, shape=kwargs["shape"])


setattr(paddle.Tensor, "reshape", reshape)


def add(self, *args, **kwargs):
    if "other" in kwargs:
        y = kwargs["other"]
    elif "y" in kwargs:
        y = kwargs["y"]
    else:
        y = args[0]

    if "alpha" in kwargs:
        alpha = kwargs["alpha"]
        if alpha != 1:
            if not isinstance(y, paddle.Tensor):
                y = paddle.to_tensor(alpha * y)
            else:
                y = alpha * y
    else:
        if not isinstance(y, paddle.Tensor):
            y = paddle.to_tensor(y)

    return paddle.add(self, y)


setattr(paddle.Tensor, "add", add)
