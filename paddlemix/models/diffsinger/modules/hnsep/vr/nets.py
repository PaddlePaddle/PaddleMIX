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

from . import layers


class BaseNet(paddle.nn.Layer):
    def __init__(self, nin, nout, nin_lstm, nout_lstm, dilations=((4, 2), (8, 4), (12, 6))):
        super(BaseNet, self).__init__()
        self.enc1 = layers.Conv2DBNActiv(nin, nout, 3, 1, 1)
        self.enc2 = layers.Encoder(nout, nout * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(nout * 2, nout * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(nout * 4, nout * 6, 3, 2, 1)
        self.enc5 = layers.Encoder(nout * 6, nout * 8, 3, 2, 1)
        self.aspp = layers.ASPPModule(nout * 8, nout * 8, dilations, dropout=True)
        self.dec4 = layers.Decoder(nout * (6 + 8), nout * 6, 3, 1, 1)
        self.dec3 = layers.Decoder(nout * (4 + 6), nout * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(nout * (2 + 4), nout * 2, 3, 1, 1)
        self.lstm_dec2 = layers.LSTMModule(nout * 2, nin_lstm, nout_lstm)
        self.dec1 = layers.Decoder(nout * (1 + 2) + 1, nout * 1, 3, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        h = self.aspp(e5)
        h = self.dec4(h, e4)
        h = self.dec3(h, e3)
        h = self.dec2(h, e2)
        h = paddle.concat(x=[h, self.lstm_dec2(h)], axis=1)
        h = self.dec1(h, e1)
        return h


class CascadedNet(paddle.nn.Layer):
    def __init__(self, n_fft, hop_length, nout=32, nout_lstm=128, is_complex=False, is_mono=False):
        super(CascadedNet, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.is_complex = is_complex
        self.is_mono = is_mono
        self.register_buffer(
            name="window",
            tensor=paddle.audio.functional.get_window("hann", n_fft).astype("float32"),
            persistable=False,
        )
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.nin_lstm = self.max_bin // 2
        self.offset = 64
        nin = 4 if is_complex else 2
        if is_mono:
            nin = nin // 2
        self.stg1_low_band_net = paddle.nn.Sequential(
            BaseNet(nin, nout // 2, self.nin_lstm // 2, nout_lstm), layers.Conv2DBNActiv(nout // 2, nout // 4, 1, 1, 0)
        )
        self.stg1_high_band_net = BaseNet(nin, nout // 4, self.nin_lstm // 2, nout_lstm // 2)
        self.stg2_low_band_net = paddle.nn.Sequential(
            BaseNet(nout // 4 + nin, nout, self.nin_lstm // 2, nout_lstm),
            layers.Conv2DBNActiv(nout, nout // 2, 1, 1, 0),
        )
        self.stg2_high_band_net = BaseNet(nout // 4 + nin, nout // 2, self.nin_lstm // 2, nout_lstm // 2)
        self.stg3_full_band_net = BaseNet(3 * nout // 4 + nin, nout, self.nin_lstm, nout_lstm)
        self.out = paddle.nn.Conv2D(in_channels=nout, out_channels=nin, kernel_size=1, bias_attr=False)
        self.aux_out = paddle.nn.Conv2D(in_channels=3 * nout // 4, out_channels=nin, kernel_size=1, bias_attr=False)

    def forward(self, x):
        if self.is_complex:
            x = paddle.concat(x=[x.real(), x.imag()], axis=1)
        x = x[:, :, : self.max_bin]
        bandw = tuple(x.shape)[2] // 2
        l1_in = x[:, :, :bandw]
        h1_in = x[:, :, bandw:]
        l1 = self.stg1_low_band_net(l1_in)
        h1 = self.stg1_high_band_net(h1_in)
        aux1 = paddle.concat(x=[l1, h1], axis=2)
        l2_in = paddle.concat(x=[l1_in, l1], axis=1)
        h2_in = paddle.concat(x=[h1_in, h1], axis=1)
        l2 = self.stg2_low_band_net(l2_in)
        h2 = self.stg2_high_band_net(h2_in)
        aux2 = paddle.concat(x=[l2, h2], axis=2)
        f3_in = paddle.concat(x=[x, aux1, aux2], axis=1)
        f3 = self.stg3_full_band_net(f3_in)
        if self.is_complex:
            mask = self.out(f3)
            if self.is_mono:
                mask = paddle.complex(real=mask[:, :1], imag=mask[:, 1:])
            else:
                mask = paddle.complex(real=mask[:, :2], imag=mask[:, 2:])
            mask = self.bounded_mask(mask)
        else:
            mask = paddle.nn.functional.sigmoid(x=self.out(f3))
        mask = paddle.nn.functional.pad(
            x=mask, pad=(0, 0, 0, self.output_bin - tuple(mask.shape)[2]), mode="replicate", pad_from_left_axis=False
        )
        return mask

    def bounded_mask(self, mask, eps=1e-08):
        mask_mag = paddle.abs(x=mask)
        mask = paddle.nn.functional.tanh(x=mask_mag) * mask / (mask_mag + eps)
        return mask

    def predict_mask(self, x):
        mask = self.forward(x)
        if self.offset > 0:
            mask = mask[:, :, :, self.offset : -self.offset]
            assert tuple(mask.shape)[3] > 0
        return mask

    def predict(self, x):
        mask = self.forward(x)
        pred = x * mask
        if self.offset > 0:
            pred = pred[:, :, :, self.offset : -self.offset]
            assert tuple(pred.shape)[3] > 0
        return pred

    def audio2spec(self, x, use_pad=False):
        B, C, T = tuple(x.shape)
        x = x.reshape(B * C, T)
        if use_pad:
            n_frames = T // self.hop_length + 1
            T_pad = (32 * ((n_frames - 1) // 32 + 1) - 1) * self.hop_length - T
            nl_pad = T_pad // 2 // self.hop_length
            Tl_pad = nl_pad * self.hop_length
            x = paddle.nn.functional.pad(x=x, pad=(Tl_pad, T_pad - Tl_pad), pad_from_left_axis=False)
        spec = paddle.signal.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
            window=self.window,
            pad_mode="constant",
        )
        spec = spec.reshape(B, C, tuple(spec.shape)[-2], tuple(spec.shape)[-1])
        return spec

    def spec2audio(self, x):
        B, C, N, T = tuple(x.shape)
        x = x.reshape(-1, N, T)
        x = paddle.signal.istft(x=x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window)
        x = x.reshape(B, C, -1)
        return x

    def predict_from_audio(self, x):
        B, C, T = tuple(x.shape)
        x = x.reshape(B * C, T)
        n_frames = T // self.hop_length + 1
        T_pad = (32 * (n_frames // 32 + 1) - 1) * self.hop_length - T
        nl_pad = T_pad // 2 // self.hop_length
        Tl_pad = nl_pad * self.hop_length
        x = paddle.nn.functional.pad(x=x, pad=(Tl_pad, T_pad - Tl_pad), pad_from_left_axis=False)
        spec = paddle.signal.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
            window=self.window,
            pad_mode="constant",
        )
        spec = spec.reshape(B, C, tuple(spec.shape)[-2], tuple(spec.shape)[-1])
        mask = self.forward(spec)
        spec_pred = spec * mask
        spec_pred = spec_pred.reshape(B * C, tuple(spec.shape)[-2], tuple(spec.shape)[-1])
        x_pred = paddle.signal.istft(x=spec_pred, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window)
        x_pred = x_pred[:, Tl_pad : Tl_pad + T]
        x_pred = x_pred.reshape(B, C, T)
        return x_pred
