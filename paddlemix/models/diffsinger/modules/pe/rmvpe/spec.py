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

import numpy as np
import paddle
from librosa.filters import mel


class MelSpectrogram(paddle.nn.Layer):
    def __init__(
        self, n_mel_channels, sampling_rate, win_length, hop_length, n_fft=None, mel_fmin=0, mel_fmax=None, clamp=1e-05
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax, htk=True)
        mel_basis = paddle.to_tensor(data=mel_basis).astype(dtype="float32")
        self.register_buffer(name="mel_basis", tensor=mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        keyshift_key = str(keyshift) + "_" + str(audio.place)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = paddle.audio.functional.get_window("hann", win_length_new).to(audio.place)
        fft = paddle.signal.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=center,
            return_complex=True,
        )
        magnitude = fft.abs()
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.shape[1]
            if resize < size:
                magnitude = paddle.nn.functional.pad(
                    x=magnitude, pad=(0, 0, 0, size - resize), pad_from_left_axis=False
                )
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
        mel_output = paddle.matmul(x=self.mel_basis, y=magnitude)
        log_mel_spec = paddle.log(x=paddle.clip(x=mel_output, min=self.clamp))
        return log_mel_spec
