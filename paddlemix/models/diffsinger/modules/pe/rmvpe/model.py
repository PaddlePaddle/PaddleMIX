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

from .constants import *
from .deepunet import DeepUnet0
from .seq import BiGRU


class E2E0(paddle.nn.Layer):
    def __init__(
        self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16
    ):
        super(E2E0, self).__init__()
        self.unet = DeepUnet0(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = paddle.nn.Conv2D(in_channels=en_out_channels, out_channels=3, kernel_size=(3, 3), padding=(1, 1))
        if n_gru:
            self.fc = paddle.nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru),
                paddle.nn.Linear(in_features=512, out_features=N_CLASS),
                paddle.nn.Dropout(p=0.25),
                paddle.nn.Sigmoid(),
            )
        else:
            self.fc = paddle.nn.Sequential(
                paddle.nn.Linear(in_features=3 * N_MELS, out_features=N_CLASS),
                paddle.nn.Dropout(p=0.25),
                paddle.nn.Sigmoid(),
            )

    def forward(self, mel):
        mel = mel.transpose(perm=paddle_aux.transpose_aux_func(mel.ndim, -1, -2)).unsqueeze(axis=1)
        x = (
            self.cnn(self.unet(mel))
            .transpose(perm=paddle_aux.transpose_aux_func(self.cnn(self.unet(mel)).ndim, 1, 2))
            .flatten(start_axis=-2)
        )
        x = self.fc(x)
        return x
