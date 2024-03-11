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


def _FUNCTIONAL_PAD(x, pad, mode="constant", value=0.0, data_format="NCHW"):
    if len(x.shape) * 2 == len(pad) and mode == "constant":
        pad = paddle.to_tensor(pad, dtype="int32").reshape((-1, 2)).flip([0]).flatten().tolist()
    return paddle.nn.functional.pad(x, pad, mode, value, data_format)


def _STR_2_PADDLE_DTYPE(type):
    type_map = {
        "uint8": paddle.uint8,
        "int8": paddle.int8,
        "int16": paddle.int16,
        "int32": paddle.int32,
        "int64": paddle.int64,
        "float16": paddle.float16,
        "float32": paddle.float32,
        "float64": paddle.float64,
        "bfloat16": paddle.bfloat16,
    }
    return type_map.get(type)
