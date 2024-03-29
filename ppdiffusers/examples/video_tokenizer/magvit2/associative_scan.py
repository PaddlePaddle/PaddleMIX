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

from typing import Callable, Tuple

import paddle
from utils import _FUNCTIONAL_PAD


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = -dim - 1 if dim < 0 else t.ndim - dim - 1
    zeros = (0, 0) * dims_from_right
    return _FUNCTIONAL_PAD(pad=(*zeros, *pad), value=value, x=t)


def associative_scan(operator: Callable, elems: Tuple[paddle.Tensor, paddle.Tensor]):
    num_elems = int(tuple(elems[0].shape)[1])
    if not all(int(tuple(elem.shape)[1]) == num_elems for elem in elems[1:]):
        raise ValueError(
            "Array inputs to associative_scan must have the same first dimension. (saw: {})".format(
                [tuple(elem.shape) for elem in elems]
            )
        )

    def _scan(elems):
        """Perform scan on `elems`."""
        num_elems = tuple(elems[0].shape)[1]
        if num_elems < 2:
            return elems
        reduced_elems = operator([elem[:, :-1:2] for elem in elems], [elem[:, 1::2] for elem in elems])
        odd_elems = _scan(reduced_elems)
        if num_elems % 2 == 0:
            even_elems = operator([e[:, :-1] for e in odd_elems], [e[:, 2::2] for e in elems])
        else:
            even_elems = operator(odd_elems, [e[:, 2::2] for e in elems])
        even_elems = [paddle.concat(x=[elem[:, :1], result], axis=1) for elem, result in zip(elems, even_elems)]
        return list(map(_interleave, even_elems, odd_elems))

    return _scan(elems)


def _interleave(a, b):
    a_axis_len, b_axis_len = tuple(a.shape)[1], tuple(b.shape)[1]
    output_axis_len = a_axis_len + b_axis_len
    if a_axis_len == b_axis_len + 1:
        b = pad_at_dim(b, (0, 1), dim=1)
    stacked = paddle.stack(x=[a, b], axis=2)
    interleaved = paddle.flatten(x=stacked, start_axis=1, stop_axis=2)
    return interleaved[:, :output_axis_len]
