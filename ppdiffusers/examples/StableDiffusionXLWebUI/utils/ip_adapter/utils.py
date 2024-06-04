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
import paddle.nn.functional as F


def is_paddle_available():
    return hasattr(F, "scaled_dot_product_attention")


def swich_state(model_weight, dtype="float16", keys=False):
    state_dict = {}
    for k in list(model_weight.keys()):
        value = paddle.to_tensor(model_weight.pop(k)).cast(dtype)
        print(k, value)
        ndim = len(value.shape)
        if ndim == 2 and not ("position_embedding" in k and "token_embedding" in k):
            state_dict[k] = value.t()
            continue
        state_dict[k] = value
    return state_dict
