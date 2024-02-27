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

from typing import List

import paddle
from utils import _FUNCTIONAL_PAD


def _compute_padding(kernel_size: List[int]) -> List[int]:
    """Compute padding tuple."""
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [(k - 1) for k in kernel_size]
    out_padding = 2 * len(kernel_size) * [0]
    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]
        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front
        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear
    return out_padding


def normalize_kernel2d(input: paddle.Tensor) -> paddle.Tensor:
    norm = input.abs().sum(axis=-1).sum(axis=-1)
    return input / norm[..., None, None]


def filter3d(
    input: paddle.Tensor, kernel: paddle.Tensor, border_type: str = "replicate", normalized: bool = False
) -> paddle.Tensor:

    b, c, d, h, w = tuple(input.shape)
    tmp_kernel = kernel[:, (None), (...)].to(device=input.place, dtype=input.dtype)
    if normalized:
        bk, dk, hk, wk = tuple(kernel.shape)
        tmp_kernel = normalize_kernel2d(tmp_kernel.view(bk, dk, hk * wk)).view_as(other=tmp_kernel)
    tmp_kernel = tmp_kernel.expand(shape=[-1, c, -1, -1, -1])
    depth, height, width = tuple(tmp_kernel.shape)[-3:]
    padding_shape: list[int] = _compute_padding([depth, height, width])
    input_pad = _FUNCTIONAL_PAD(pad=padding_shape, mode=border_type, x=input)
    tmp_kernel = tmp_kernel.reshape(-1, 1, depth, height, width)
    input_pad = input_pad.view(-1, tmp_kernel.shape[0], input_pad.shape[-3], input_pad.shape[-2], input_pad.shape[-1])
    output = paddle.nn.functional.conv3d(
        x=input_pad, weight=tmp_kernel, groups=tmp_kernel.shape[0], padding=0, stride=1
    )
    return output.view(b, c, d, h, w)
