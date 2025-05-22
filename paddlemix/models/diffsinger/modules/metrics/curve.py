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
import torchmetrics


class RawCurveAccuracy(torchmetrics.Metric):
    def __init__(self, *, tolerance, **kwargs):
        super().__init__(**kwargs)
        self.tolerance = tolerance
        self.add_state("close", default=paddle.to_tensor(data=0, dtype="int32"), dist_reduce_fx="sum")
        self.add_state("total", default=paddle.to_tensor(data=0, dtype="int32"), dist_reduce_fx="sum")

    def update(self, pred: paddle.Tensor, target: paddle.Tensor, mask=None) -> None:
        """

        :param pred: predicted curve
        :param target: reference curve
        :param mask: valid or non-padding mask
        """
        if mask is None:
            assert tuple(pred.shape) == tuple(
                target.shape
            ), f"shapes of pred and target mismatch: {tuple(pred.shape)}, {tuple(target.shape)}"
        else:
            assert (
                tuple(pred.shape) == tuple(target.shape) == tuple(mask.shape)
            ), f"shapes of pred, target and mask mismatch: {tuple(pred.shape)}, {tuple(target.shape)}, {tuple(mask.shape)}"
        close = paddle.abs(x=pred - target) <= self.tolerance
        if mask is not None:
            close &= mask
        self.close += close.sum()
        self.total += pred.size if mask is None else mask.sum()

    def compute(self) -> paddle.Tensor:
        return self.close / self.total
