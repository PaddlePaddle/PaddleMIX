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
import paddle.nn as nn

from .efficientnet_v2_s import efficientnet_v2_s


class BatchNorm2D(nn.Layer):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm2D, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = self.create_parameter(
                shape=[num_features], default_initializer=paddle.nn.initializer.Constant(value=1.0)
            )
            self.bias = self.create_parameter(
                shape=[num_features], default_initializer=paddle.nn.initializer.Constant(value=0.0)
            )
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self._mean = self.create_parameter(
                shape=[num_features], default_initializer=paddle.nn.initializer.Constant(value=0.0), is_bias=False
            )
            self._variance = self.create_parameter(
                shape=[num_features], default_initializer=paddle.nn.initializer.Constant(value=1.0), is_bias=False
            )
            self._mean.stop_gradient = True
            self._variance.stop_gradient = True
        else:
            self._mean = None
            self._variance = None

    def forward(self, input):
        mean = self._mean
        variance = self._variance

        output = (input - paddle.unsqueeze(mean, axis=[0, 2, 3])) / paddle.unsqueeze(
            paddle.sqrt(variance + self.eps), axis=[0, 2, 3]
        )
        if self.affine:
            output = output * paddle.unsqueeze(self.weight, axis=[0, 2, 3]) + paddle.unsqueeze(
                self.bias, axis=[0, 2, 3]
            )
        return output


class EfficientNetEncoder(nn.Layer):
    def __init__(self, c_latent=16):
        super().__init__()
        self.backbone = efficientnet_v2_s().features
        self.backbone.eval()
        self.mapper = nn.Sequential(
            nn.Conv2D(1280, c_latent, kernel_size=1, bias_attr=False),
            BatchNorm2D(c_latent, affine=False),
        )
        self.mapper.eval()

    def forward(self, x):

        x = self.backbone(x)
        x = self.mapper(x)
        return x
