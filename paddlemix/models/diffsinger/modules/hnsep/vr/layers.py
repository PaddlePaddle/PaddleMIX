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


def crop_center(h1, h2):
    h1_shape = tuple(h1.shape)
    h2_shape = tuple(h2.shape)
    if h1_shape[3] == h2_shape[3]:
        return h1
    elif h1_shape[3] < h2_shape[3]:
        raise ValueError("h1_shape[3] must be greater than h2_shape[3]")
    s_time = (h1_shape[3] - h2_shape[3]) // 2
    e_time = s_time + h2_shape[3]
    h1 = h1[:, :, :, s_time:e_time]
    return h1


class Conv2DBNActiv(paddle.nn.Layer):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, dilation=1, activ=paddle.nn.ReLU):
        super(Conv2DBNActiv, self).__init__()
        self.conv = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=nin,
                out_channels=nout,
                kernel_size=ksize,
                stride=stride,
                padding=pad,
                dilation=dilation,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=nout),
            activ(),
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(paddle.nn.Layer):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=paddle.nn.LeakyReLU):
        super(Encoder, self).__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, stride, pad, activ=activ)
        self.conv2 = Conv2DBNActiv(nout, nout, ksize, 1, pad, activ=activ)

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        return h


class Decoder(paddle.nn.Layer):
    def __init__(self, nin, nout, ksize=3, stride=1, pad=1, activ=paddle.nn.ReLU, dropout=False):
        super(Decoder, self).__init__()
        self.conv1 = Conv2DBNActiv(nin, nout, ksize, 1, pad, activ=activ)
        self.dropout = paddle.nn.Dropout2D(p=0.1) if dropout else None

    def forward(self, x, skip=None):
        x = paddle.nn.functional.interpolate(x=x, scale_factor=2, mode="bilinear", align_corners=True)
        if skip is not None:
            skip = crop_center(skip, x)
            x = paddle.concat(x=[x, skip], axis=1)
        h = self.conv1(x)
        if self.dropout is not None:
            h = self.dropout(h)
        return h


class Mean(paddle.nn.Layer):
    def __init__(self, dim, keepdims=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keepdims = keepdims

    def forward(self, x):
        return x.mean(self.dim, keepdims=self.keepdims)


class ASPPModule(paddle.nn.Layer):
    def __init__(self, nin, nout, dilations=(4, 8, 12), activ=paddle.nn.ReLU, dropout=False):
        super(ASPPModule, self).__init__()
        self.conv1 = paddle.nn.Sequential(Mean(dim=-2, keepdims=True), Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ))
        self.conv2 = Conv2DBNActiv(nin, nout, 1, 1, 0, activ=activ)
        self.conv3 = Conv2DBNActiv(nin, nout, 3, 1, dilations[0], dilations[0], activ=activ)
        self.conv4 = Conv2DBNActiv(nin, nout, 3, 1, dilations[1], dilations[1], activ=activ)
        self.conv5 = Conv2DBNActiv(nin, nout, 3, 1, dilations[2], dilations[2], activ=activ)
        self.bottleneck = Conv2DBNActiv(nout * 5, nout, 1, 1, 0, activ=activ)
        self.dropout = paddle.nn.Dropout2D(p=0.1) if dropout else None

    def forward(self, x):
        _, _, h, w = tuple(x.shape)
        feat1 = self.conv1(x).tile(repeat_times=[1, 1, h, 1])
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = paddle.concat(x=(feat1, feat2, feat3, feat4, feat5), axis=1)
        out = self.bottleneck(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class LSTMModule(paddle.nn.Layer):
    def __init__(self, nin_conv, nin_lstm, nout_lstm):
        super(LSTMModule, self).__init__()
        self.conv = Conv2DBNActiv(nin_conv, 1, 1, 1, 0)
        self.lstm = paddle.nn.LSTM(
            input_size=nin_lstm, hidden_size=nout_lstm // 2, time_major=not False, direction="bidirect"
        )
        self.dense = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=nout_lstm, out_features=nin_lstm),
            paddle.nn.BatchNorm1D(num_features=nin_lstm),
            paddle.nn.ReLU(),
        )

    def forward(self, x):
        N, _, nbins, nframes = tuple(x.shape)
        h = self.conv(x)[:, 0]
        h = h.transpose(perm=[2, 0, 1])
        h, _ = self.lstm(h)
        h = self.dense(h.reshape(-1, tuple(h.shape)[-1]))
        h = h.reshape(nframes, N, 1, nbins)
        h = h.transpose(perm=[1, 2, 3, 0])
        return h
