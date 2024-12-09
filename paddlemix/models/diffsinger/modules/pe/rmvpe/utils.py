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

import sys

import librosa
import numpy as np
import paddle
import paddle_aux

from .constants import *


def to_local_average_f0(hidden, center=None, thred=0.03):
    idx = paddle.arange(end=N_CLASS)[None, None, :]
    idx_cents = idx * 20 + CONST
    if center is None:
        center = paddle.argmax(x=hidden, axis=2, keepdim=True)
    start = paddle.clip(x=center - 4, min=0)
    end = paddle.clip(x=center + 5, max=N_CLASS)
    idx_mask = (idx >= start) & (idx < end)
    weights = hidden * idx_mask
    product_sum = paddle.sum(x=weights * idx_cents, axis=2)
    weight_sum = paddle.sum(x=weights, axis=2)
    cents = product_sum / (weight_sum + (weight_sum == 0))
    f0 = 10 * 2 ** (cents / 1200)
    uv = hidden.max(dim=2)[0] < thred
    f0 = f0 * ~uv
    return f0.squeeze(axis=0).cpu().numpy()


def to_viterbi_f0(hidden, thred=0.03):
    if not hasattr(to_viterbi_f0, "transition"):
        xx, yy = np.meshgrid(range(N_CLASS), range(N_CLASS))
        transition = np.maximum(30 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        to_viterbi_f0.transition = transition
    prob = hidden.squeeze(axis=0).cpu().numpy()
    prob = prob.T
    prob = prob / prob.sum(axis=0)
    path = librosa.sequence.viterbi(prob, to_viterbi_f0.transition).astype(np.int64)
    center = paddle.to_tensor(data=path).unsqueeze(axis=0).unsqueeze(axis=-1).to(hidden.place)
    return to_local_average_f0(hidden, center=center, thred=thred)
