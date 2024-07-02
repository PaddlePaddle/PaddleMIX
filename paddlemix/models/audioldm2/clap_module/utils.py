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

import librosa
import numpy as np
import paddle
import paddle.nn as nn


def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat_interleave(ratio, 2)
    upsampled = upsampled.reshape([batch_size, time_steps * ratio, classes_num])
    return upsampled


def do_mixup(x, mixup_lambda):
    """
    Args:
      x: (batch_size , ...)
      mixup_lambda: (batch_size,)
    Returns:
      out: (batch_size, ...)
    """
    perm_shape = list(range(x.dim()))
    new_perm_shape = perm_shape
    new_perm_shape[0], new_perm_shape[-1] = perm_shape[-1], perm_shape[0]
    out = (
        x.transpose(new_perm_shape) * mixup_lambda
        + paddle.flip(x, axis=[0]).transpose(new_perm_shape) * (1 - mixup_lambda)
    ).transpose(new_perm_shape)
    return out


class DFTBase(nn.Layer):
    def __init__(self):
        r"""Base class for DFT and IDFT matrix."""
        super(DFTBase, self).__init__()

    def dft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2 * np.pi * 1j / n)
        W = np.power(omega, x * y)  # shape: (n, n)
        return W

    def idft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(2 * np.pi * 1j / n)
        W = np.power(omega, x * y)  # shape: (n, n)
        return W


class STFT(DFTBase):
    def __init__(
        self,
        n_fft=2048,
        hop_length=None,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        freeze_parameters=True,
    ):
        r"""Paddle implementation of STFT with Conv1d. The function has the
        same output as librosa.stft.

        Args:
            n_fft: int, fft window size, e.g., 2048
            hop_length: int, hop length samples, e.g., 441
            win_length: int, window length e.g., 2048
            window: str, window function name, e.g., 'hann'
            center: bool
            pad_mode: str, e.g., 'reflect'
            freeze_parameters: bool, set to True to freeze all parameters. Set
                to False to finetune all parameters.
        """
        super(STFT, self).__init__()

        assert pad_mode in ["constant", "reflect"]

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame.
        if self.win_length is None:
            self.win_length = n_fft

        # Set the default hop, if it's not already specified.
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        fft_window = librosa.filters.get_window(window, self.win_length, fftbins=True)

        # Pad the window out to n_fft size.
        fft_window = librosa.util.pad_center(data=fft_window, size=n_fft)

        # DFT & IDFT matrix.
        self.W = self.dft_matrix(n_fft)

        out_channels = n_fft // 2 + 1

        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Assign(
                paddle.to_tensor(np.real(self.W[:, 0:out_channels] * fft_window[:, None]).T)[:, None, :]
            )
        )
        self.conv_real = nn.Conv1D(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=n_fft,
            stride=self.hop_length,
            padding=0,
            dilation=1,
            groups=1,
            weight_attr=weight_attr,
            bias_attr=False,
        )

        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Assign(
                paddle.to_tensor(np.imag(self.W[:, 0:out_channels] * fft_window[:, None]).T)[:, None, :]
            )
        )
        self.conv_imag = nn.Conv1D(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=n_fft,
            stride=self.hop_length,
            padding=0,
            dilation=1,
            groups=1,
            weight_attr=weight_attr,
            bias_attr=False,
        )

        if freeze_parameters:
            for param in self.parameters():
                param.stop_gradient = True

    def forward(self, input):
        r"""Calculate STFT of batch of signals.

        Args:
            input: (batch_size, data_length), input signals.

        Returns:
            real: (batch_size, 1, time_steps, n_fft // 2 + 1)
            imag: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """

        x = input[:, None, :]  # (batch_size, channels_num, data_length)

        if self.center:
            x = nn.functional.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode, data_format="NCL")

        real = self.conv_real(x)
        imag = self.conv_imag(x)
        # (batch_size, n_fft // 2 + 1, time_steps)

        real = real[:, None, :, :].transpose([0, 1, 3, 2])
        imag = imag[:, None, :, :].transpose([0, 1, 3, 2])
        # (batch_size, 1, time_steps, n_fft // 2 + 1)

        return real, imag


class Spectrogram(nn.Layer):
    def __init__(
        self,
        n_fft=2048,
        hop_length=None,
        win_length=None,
        window="hann",
        center=True,
        pad_mode="reflect",
        power=2.0,
        freeze_parameters=True,
    ):
        r"""Calculate spectrogram using paddle. The STFT is implemented with
        Conv1d. The function has the same output of librosa.stft
        """
        super(Spectrogram, self).__init__()

        self.power = power

        self.stft = STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

    def forward(self, input):
        r"""Calculate spectrogram of input signals.
        Args:
            input: (batch_size, data_length)

        Returns:
            spectrogram: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """

        (real, imag) = self.stft.forward(input)
        # (batch_size, n_fft // 2 + 1, time_steps)

        spectrogram = real**2 + imag**2

        if self.power == 2.0:
            pass
        else:
            spectrogram = spectrogram ** (self.power / 2.0)

        return spectrogram


class LogmelFilterBank(nn.Layer):
    def __init__(
        self,
        sr=22050,
        n_fft=2048,
        n_mels=64,
        fmin=0.0,
        fmax=None,
        is_log=True,
        ref=1.0,
        amin=1e-10,
        top_db=80.0,
        freeze_parameters=True,
    ):
        r"""Calculate logmel spectrogram using paddle. The mel filter bank is
        the paddle implementation of as librosa.filters.mel
        """
        super(LogmelFilterBank, self).__init__()

        self.is_log = is_log
        self.ref = paddle.to_tensor(ref, dtype="float32")
        self.amin = paddle.to_tensor(amin, dtype="float32")
        self.top_db = top_db
        if fmax is None:
            fmax = sr // 2

        self.melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).T
        # (n_fft // 2 + 1, mel_bins)

        self.melW = paddle.to_tensor(self.melW)
        self.melW = paddle.create_parameter(
            self.melW.shape, str(self.melW.numpy().dtype), default_initializer=nn.initializer.Assign(self.melW)
        )

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        r"""Calculate (log) mel spectrogram from spectrogram.

        Args:
            input: (*, n_fft), spectrogram

        Returns:
            output: (*, mel_bins), (log) mel spectrogram
        """

        # Mel spectrogram
        mel_spectrogram = paddle.matmul(input, self.melW)
        # (*, mel_bins)

        # Logmel spectrogram
        if self.is_log:
            output = self.power_to_db(mel_spectrogram)
        else:
            output = mel_spectrogram

        return output

    def power_to_db(self, input):
        r"""Power to db, this function is the paddle implementation of
        librosa.power_to_lb
        """
        ref_value = self.ref
        log_spec = 10.0 * paddle.log10(paddle.clip(input, min=self.amin, max=None))
        log_spec -= 10.0 * paddle.log10(paddle.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise librosa.util.exceptions.ParameterError("top_db must be non-negative")
            log_spec = paddle.clip(log_spec, min=log_spec.max().item() - self.top_db, max=None)

        return log_spec


class DropStripes(nn.Layer):
    def __init__(self, dim, drop_width, stripes_num):
        """Drop stripes.

        Args:
          dim: int, dimension along which to drop
          drop_width: int, maximum width of stripes to drop
          stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [2, 3]  # dim 2: time; dim 3: frequency

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, input):
        """input: (batch_size, channels, time_steps, freq_bins)"""

        assert input.ndim == 4

        if self.training is False:
            return input

        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]

            for n in range(batch_size):
                self.transform_slice(input[n], total_width)

            return input

    def transform_slice(self, e, total_width):
        """e: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = paddle.randint(low=0, high=self.drop_width, shape=(1,))[0]
            bgn = paddle.randint(low=0, high=total_width - distance, shape=(1,))[0]

            if self.dim == 2:
                e[:, bgn : bgn + distance, :] = 0
            elif self.dim == 3:
                e[:, :, bgn : bgn + distance] = 0


class SpecAugmentation(nn.Layer):
    def __init__(self, time_drop_width, time_stripes_num, freq_drop_width, freq_stripes_num):
        """Spec augmentation.
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D.
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.

        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
        """

        super(SpecAugmentation, self).__init__()

        self.time_dropper = DropStripes(dim=2, drop_width=time_drop_width, stripes_num=time_stripes_num)

        self.freq_dropper = DropStripes(dim=3, drop_width=freq_drop_width, stripes_num=freq_stripes_num)

    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x
