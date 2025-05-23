# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Optional

import numpy as np
import paddle


def scatter(
    tensor: paddle.Tensor, index: paddle.Tensor, src: paddle.Tensor, dim: int = 0, reduce: Optional[str] = None
) -> paddle.Tensor:

    dim = dim if dim >= 0 else tensor.ndim + dim
    max_idx = tensor.shape[dim] - 1
    index = paddle.clip(index, 0, max_idx)

    for d in range(tensor.ndim):
        if d == dim:

            target_size = index.shape[d]
            if src.shape[d] < target_size:
                src = dynamic_pad(src, target_size, d)
            elif src.shape[d] > target_size:
                src = src.slice([d], 0, target_size)
        else:
            target_size = max(index.shape[d], src.shape[d])
            if index.shape[d] < target_size:
                index = dynamic_pad(index, target_size, d)
            elif index.shape[d] > target_size:
                index = index.slice([d], 0, target_size)
            if src.shape[d] < target_size:
                src = dynamic_pad(src, target_size, d)
            elif src.shape[d] > target_size:
                src = src.slice([d], 0, target_size)

    output = tensor.clone()
    grid = [paddle.arange(s) for s in src.shape]
    indices = paddle.meshgrid(*grid, indexing="ij")

    for coord in np.ndindex(tuple(src.shape)):
        idx_tuple = tuple(indices[i][coord] for i in range(len(coord)))
        target_idx = list(idx_tuple)
        target_idx[dim] = index[coord]
        target_idx = tuple(target_idx)

        current_val = src[coord].item()
        if reduce == "sum":
            output[target_idx] += current_val
        elif reduce == "multiply":
            output[target_idx] *= current_val
        else:
            output[target_idx] = current_val
    return output


def dynamic_pad(tensor: paddle.Tensor, target_size: int, dim: int, pad_value=0) -> paddle.Tensor:
    pad_size = target_size - tensor.shape[dim]
    if pad_size <= 0:
        return tensor
    pad_shape = [(0, 0)] * tensor.ndim
    pad_shape[dim] = (0, pad_size)
    return paddle.nn.functional.pad(tensor, pad=pad_shape, mode="constant", value=pad_value)
