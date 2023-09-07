# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class CosineSimilarityLoss(paddle.nn.Layer):
    def __init__(self, data_type=paddle.float32):
        super(CosineSimilarityLoss, self).__init__()
        self.data_type = data_type
        self.loss_func = paddle.nn.CosineSimilarity(axis=-1)

    def forward(self, output, label):
        loss = self.loss_func(output.cast(self.data_type), label.cast(self.data_type))
        return -loss.mean()


class LabelSmoothingCrossEntropy(paddle.nn.Layer):
    """NLL loss with label smoothing."""

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, axis=-1)
        nll_loss = -logprobs.take_along_axis(indices=target.unsqueeze(1), axis=-1)
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(axis=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
