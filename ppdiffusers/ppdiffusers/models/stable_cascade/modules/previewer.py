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


class Previewer(paddle.nn.Layer):
    def __init__(self, c_in=16, c_hidden=512, c_out=3):
        super().__init__()
        self.blocks = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=c_in, out_channels=c_hidden, kernel_size=1),
            paddle.nn.GELU(),
            paddle.nn.BatchNorm2D(num_features=c_hidden),
            paddle.nn.Conv2D(in_channels=c_hidden, out_channels=c_hidden, kernel_size=3, padding=1),
            paddle.nn.GELU(),
            paddle.nn.BatchNorm2D(num_features=c_hidden),
            paddle.nn.Conv2DTranspose(
                in_channels=c_hidden,
                out_channels=c_hidden // 2,
                kernel_size=2,
                stride=2,
            ),
            paddle.nn.GELU(),
            paddle.nn.BatchNorm2D(num_features=c_hidden // 2),
            paddle.nn.Conv2D(
                in_channels=c_hidden // 2,
                out_channels=c_hidden // 2,
                kernel_size=3,
                padding=1,
            ),
            paddle.nn.GELU(),
            paddle.nn.BatchNorm2D(num_features=c_hidden // 2),
            paddle.nn.Conv2DTranspose(
                in_channels=c_hidden // 2,
                out_channels=c_hidden // 4,
                kernel_size=2,
                stride=2,
            ),
            paddle.nn.GELU(),
            paddle.nn.BatchNorm2D(num_features=c_hidden // 4),
            paddle.nn.Conv2D(
                in_channels=c_hidden // 4,
                out_channels=c_hidden // 4,
                kernel_size=3,
                padding=1,
            ),
            paddle.nn.GELU(),
            paddle.nn.BatchNorm2D(num_features=c_hidden // 4),
            paddle.nn.Conv2DTranspose(
                in_channels=c_hidden // 4,
                out_channels=c_hidden // 4,
                kernel_size=2,
                stride=2,
            ),
            paddle.nn.GELU(),
            paddle.nn.BatchNorm2D(num_features=c_hidden // 4),
            paddle.nn.Conv2D(
                in_channels=c_hidden // 4,
                out_channels=c_hidden // 4,
                kernel_size=3,
                padding=1,
            ),
            paddle.nn.GELU(),
            paddle.nn.BatchNorm2D(num_features=c_hidden // 4),
            paddle.nn.Conv2D(in_channels=c_hidden // 4, out_channels=c_out, kernel_size=1),
        )

    def forward(self, x):
        return self.blocks(x)
