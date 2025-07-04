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


class BCELoss(paddle.nn.Layer):
    def forward(self, prediction, target):
        loss = paddle.nn.functional.binary_cross_entropy_with_logits(logit=prediction, label=target)
        return loss, {}


class BCELossWithQuant(paddle.nn.Layer):
    def __init__(self, codebook_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight

    def forward(self, qloss, target, prediction, split):
        bce_loss = paddle.nn.functional.binary_cross_entropy_with_logits(logit=prediction, label=target)
        loss = bce_loss + self.codebook_weight * qloss
        return loss, {
            "{}/total_loss".format(split): loss.clone().detach().mean(),
            "{}/bce_loss".format(split): bce_loss.detach().mean(),
            "{}/quant_loss".format(split): qloss.detach().mean(),
        }
