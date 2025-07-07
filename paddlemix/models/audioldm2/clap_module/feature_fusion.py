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


class DAF(nn.Layer):
    """
    直接相加 DirectAddFuse
    """

    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual


class iAFF(nn.Layer):
    """
    多特征融合 iAFF
    """

    def __init__(self, channels=64, r=4, type="2D"):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        if type == "1D":
            # 本地注意力
            self.local_att = nn.Sequential(
                nn.Conv1D(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1D(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1D(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1D(channels),
            )

            # 全局注意力
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool1D(1),
                nn.Conv1D(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1D(inter_channels),
                nn.ReLU(),
                nn.Conv1D(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1D(channels),
            )

            # 第二次本地注意力
            self.local_att2 = nn.Sequential(
                nn.Conv1D(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1D(inter_channels),
                nn.ReLU(),
                nn.Conv1D(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1D(channels),
            )
            # 第二次全局注意力
            self.global_att2 = nn.Sequential(
                nn.AdaptiveAvgPool1D(1),
                nn.Conv1D(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1D(inter_channels),
                nn.ReLU(),
                nn.Conv1D(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1D(channels),
            )
        elif type == "2D":
            # 本地注意力
            self.local_att = nn.Sequential(
                nn.Conv2D(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2D(inter_channels),
                nn.ReLU(),
                nn.Conv2D(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2D(channels),
            )

            # 全局注意力
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2D(1),
                nn.Conv2D(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2D(inter_channels),
                nn.ReLU(),
                nn.Conv2D(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2D(channels),
            )

            # 第二次本地注意力
            self.local_att2 = nn.Sequential(
                nn.Conv2D(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2D(inter_channels),
                nn.ReLU(),
                nn.Conv2D(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2D(channels),
            )
            # 第二次全局注意力
            self.global_att2 = nn.Sequential(
                nn.AdaptiveAvgPool2D(1),
                nn.Conv2D(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2D(inter_channels),
                nn.ReLU(),
                nn.Conv2D(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2D(channels),
            )
        else:
            raise "the type is not supported"

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        flag = False
        xa = x + residual
        if xa.size(0) == 1:
            xa = paddle.concat([xa, xa], axis=0)
            flag = True
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        if flag:
            xo = xo[0].unsqueeze(0)
        return xo


class AFF(nn.Layer):
    """
    多特征融合 AFF
    """

    def __init__(self, channels=64, r=4, type="2D"):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        if type == "1D":
            self.local_att = nn.Sequential(
                nn.Conv1D(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1D(inter_channels),
                nn.ReLU(),
                nn.Conv1D(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1D(channels),
            )
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool1D(1),
                nn.Conv1D(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1D(inter_channels),
                nn.ReLU(),
                nn.Conv1D(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1D(channels),
            )
        elif type == "2D":
            self.local_att = nn.Sequential(
                nn.Conv2D(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2D(inter_channels),
                nn.ReLU(),
                nn.Conv2D(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2D(channels),
            )
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2D(1),
                nn.Conv2D(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2D(inter_channels),
                nn.ReLU(),
                nn.Conv2D(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2D(channels),
            )
        else:
            raise "the type is not supported."

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        flag = False
        xa = x + residual
        if xa.size(0) == 1:
            xa = paddle.concat([xa, xa], axis=0)
            flag = True
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        if flag:
            xo = xo[0].unsqueeze(0)
        return xo
