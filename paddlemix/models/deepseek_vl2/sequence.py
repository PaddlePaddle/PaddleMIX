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

import paddle


def pad_sequence(sequences, padding_value=0, fix_len=None):
    """Pad a list of variable length paddle.Tensor with size L x *
    where L is length of the sequence and * is any number of dimensions (including 0)
    """
    max_size = sequences[0].shape
    trailing_dims = max_size[1:]
    max_len = max([s.shape[0] for s in sequences])
    if fix_len is not None:
        assert fix_len >= max_len, "fix_len is too small."
        max_len = fix_len
    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = paddle.full(out_dims, padding_value, dtype=sequences[0].dtype)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        out_tensor[i, :length, ...] = tensor
    return out_tensor
