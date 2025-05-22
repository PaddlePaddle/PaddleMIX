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

import paddle
import paddle_aux
import torchmetrics
from modules.fastspeech.tts_modules import RhythmRegulator


def linguistic_checks(pred, target, ph2word, mask=None):
    if mask is None:
        assert (
            tuple(pred.shape) == tuple(target.shape) == tuple(ph2word.shape)
        ), f"shapes of pred, target and ph2word mismatch: {tuple(pred.shape)}, {tuple(target.shape)}, {tuple(ph2word.shape)}"
    else:
        assert (
            tuple(pred.shape) == tuple(target.shape) == tuple(ph2word.shape) == tuple(mask.shape)
        ), f"shapes of pred, target and mask mismatch: {tuple(pred.shape)}, {tuple(target.shape)}, {tuple(ph2word.shape)}, {tuple(mask.shape)}"
    assert pred.ndim == 2, f"all inputs should be 2D, but got {tuple(pred.shape)}"
    assert paddle.any(x=ph2word > 0), "empty word sequence"
    assert paddle.all(x=ph2word >= 0), "unexpected negative word index"
    assert ph2word.max() <= tuple(pred.shape)[1], f"word index out of range: {ph2word.max()} > {tuple(pred.shape)[1]}"
    assert paddle.all(x=pred >= 0.0), f"unexpected negative ph_dur prediction"
    assert paddle.all(x=target >= 0.0), f"unexpected negative ph_dur target"


class RhythmCorrectness(torchmetrics.Metric):
    def __init__(self, *, tolerance, **kwargs):
        super().__init__(**kwargs)
        assert 0.0 < tolerance < 1.0, "tolerance should be within (0, 1)"
        self.tolerance = tolerance
        self.add_state("correct", default=paddle.to_tensor(data=0, dtype="int32"), dist_reduce_fx="sum")
        self.add_state("total", default=paddle.to_tensor(data=0, dtype="int32"), dist_reduce_fx="sum")

    def update(self, pdur_pred: paddle.Tensor, pdur_target: paddle.Tensor, ph2word: paddle.Tensor, mask=None) -> None:
        """

        :param pdur_pred: predicted ph_dur
        :param pdur_target: reference ph_dur
        :param ph2word: word division sequence
        :param mask: valid or non-padding mask
        """
        linguistic_checks(pdur_pred, pdur_target, ph2word, mask=mask)
        shape = tuple(pdur_pred.shape)[0], ph2word.max() + 1
        wdur_pred = paddle.zeros(shape=shape, dtype=pdur_pred.dtype).put_along_axis(
            axis=1, indices=ph2word, values=pdur_pred, reduce="add"
        )[:, 1:]
        wdur_target = paddle.zeros(shape=shape, dtype=pdur_target.dtype).put_along_axis(
            axis=1, indices=ph2word, values=pdur_target, reduce="add"
        )[:, 1:]
        if mask is None:
            wdur_mask = paddle.ones_like(x=wdur_pred, dtype="bool")
        else:
            wdur_mask = (
                paddle.zeros(shape=shape, dtype=mask.dtype)
                .put_along_axis(axis=1, indices=ph2word, values=mask, reduce="add")[:, 1:]
                .astype(dtype="bool")
            )
        correct = paddle.abs(x=wdur_pred - wdur_target) <= wdur_target * self.tolerance
        correct &= wdur_mask
        self.correct += correct.sum()
        self.total += wdur_mask.sum()

    def compute(self) -> paddle.Tensor:
        return self.correct / self.total


class PhonemeDurationAccuracy(torchmetrics.Metric):
    def __init__(self, *, tolerance, **kwargs):
        super().__init__(**kwargs)
        self.tolerance = tolerance
        self.rr = RhythmRegulator()
        self.add_state("accurate", default=paddle.to_tensor(data=0, dtype="int32"), dist_reduce_fx="sum")
        self.add_state("total", default=paddle.to_tensor(data=0, dtype="int32"), dist_reduce_fx="sum")

    def update(self, pdur_pred: paddle.Tensor, pdur_target: paddle.Tensor, ph2word: paddle.Tensor, mask=None) -> None:
        """

        :param pdur_pred: predicted ph_dur
        :param pdur_target: reference ph_dur
        :param ph2word: word division sequence
        :param mask: valid or non-padding mask
        """
        linguistic_checks(pdur_pred, pdur_target, ph2word, mask=mask)
        shape = tuple(pdur_pred.shape)[0], ph2word.max() + 1
        wdur_target = paddle.zeros(shape=shape, dtype=pdur_target.dtype).put_along_axis(
            axis=1, indices=ph2word, values=pdur_target, reduce="add"
        )[:, 1:]
        pdur_align = self.rr(pdur_pred, ph2word=ph2word, word_dur=wdur_target)
        accurate = paddle.abs(x=pdur_align - pdur_target) <= pdur_target * self.tolerance
        if mask is not None:
            accurate &= mask
        self.accurate += accurate.sum()
        self.total += pdur_pred.size if mask is None else mask.sum()

    def compute(self) -> paddle.Tensor:
        return self.accurate / self.total
