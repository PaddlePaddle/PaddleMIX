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

import os

import paddle

os.environ["LRU_CACHE_CAPACITY"] = "3"
import numpy as np
from librosa.filters import mel as librosa_mel_fn


def dynamic_range_compression(x, C=1, clip_val=1e-05):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-05):
    return paddle.log(x=paddle.clip(x=x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return paddle.exp(x=x) / C


class STFT:
    def __init__(
        self,
        sr=22050,
        n_mels=80,
        n_fft=1024,
        win_size=1024,
        hop_length=256,
        fmin=20,
        fmax=11025,
        clip_val=1e-05,
        device=None,
    ):
        self.target_sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.clip_val = clip_val
        if device is None:
            device = str("cuda" if paddle.device.cuda.device_count() >= 1 else "cpu").replace("cuda", "gpu")
        self.device = device
        mel_basis = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        self.mel_basis = paddle.to_tensor(data=mel_basis).astype(dtype="float32").to(device)

    def get_mel(self, y, keyshift=0, speed=1, center=False):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_size_new = int(np.round(self.win_size * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        if paddle.min(x=y) < -1.0:
            print("min value is ", paddle.min(x=y))
        if paddle.max(x=y) > 1.0:
            print("max value is ", paddle.max(x=y))
        window = paddle.audio.functional.get_window("hann", win_size_new).astype("float32").to(self.device)
        y = paddle.nn.functional.pad(
            x=y.unsqueeze(axis=1),
            pad=((win_size_new - hop_length_new) // 2, (win_size_new - hop_length_new + 1) // 2),
            mode="reflect",
            pad_from_left_axis=False,
        )
        y = y.squeeze(axis=1)
        spec = paddle.signal.stft(
            y,
            n_fft_new,
            hop_length=hop_length_new,
            win_length=win_size_new,
            window=window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
        ).abs()

        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = spec.shape[1]
            if resize < size:
                spec = paddle.nn.functional.pad(x=spec, pad=(0, 0, 0, size - resize), pad_from_left_axis=False)
            spec = spec[:, :size, :] * self.win_size / win_size_new
        spec = paddle.matmul(x=self.mel_basis, y=spec)
        spec = dynamic_range_compression_torch(spec, clip_val=self.clip_val)
        return spec
