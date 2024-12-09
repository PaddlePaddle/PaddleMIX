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
from typing import Union

import librosa
import numpy as np
import paddle
import parselmouth
from modules.nsf_hifigan.nvSTFT import STFT
from utils.decomposed_waveform import DecomposedWaveform
from utils.pitch_utils import interp_f0


def get_mel_torch(
    waveform,
    samplerate,
    *,
    num_mel_bins=128,
    hop_size=512,
    win_size=2048,
    fft_size=2048,
    fmin=40,
    fmax=16000,
    keyshift=0,
    speed=1,
    device=None
):
    if device is None:
        device = "cuda" if paddle.device.cuda.device_count() >= 1 else "cpu"
    stft = STFT(samplerate, num_mel_bins, fft_size, win_size, hop_size, fmin, fmax, device=device)
    with paddle.no_grad():
        wav_torch = paddle.to_tensor(data=waveform).to(device)
        mel_torch = stft.get_mel(wav_torch.unsqueeze(axis=0), keyshift=keyshift, speed=speed).squeeze(axis=0).T
        return mel_torch.cpu().numpy()


@paddle.no_grad()
def get_mel2ph_torch(lr, durs, length, timestep, device="cpu"):
    ph_acc = paddle.round(paddle.cumsum(x=durs.to(device), axis=0) / timestep + 0.5).astype(dtype="int64")
    ph_dur = paddle.diff(x=ph_acc, axis=0, prepend=paddle.to_tensor(data=[0], dtype="int64").to(device))
    mel2ph = lr(ph_dur[None])[0]
    num_frames = tuple(mel2ph.shape)[0]
    if num_frames < length:
        mel2ph = paddle.concat(x=(mel2ph, paddle.full(shape=(length - num_frames,), fill_value=mel2ph[-1])), axis=0)
    elif num_frames > length:
        mel2ph = mel2ph[:length]
    return mel2ph


def get_pitch_parselmouth(waveform, samplerate, length, *, hop_size, f0_min=65, f0_max=1100, speed=1, interp_uv=False):
    """

    :param waveform: [T]
    :param samplerate: sampling rate
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param f0_min: Minimum f0 in Hz
    :param f0_max: Maximum f0 in Hz
    :param speed: Change the speed
    :param interp_uv: Interpolate unvoiced parts
    :return: f0, uv
    """
    hop_size = int(np.round(hop_size * speed))
    time_step = hop_size / samplerate
    l_pad = int(np.ceil(1.5 / f0_min * samplerate))
    r_pad = hop_size * ((len(waveform) - 1) // hop_size + 1) - len(waveform) + l_pad + 1
    waveform = np.pad(waveform, (l_pad, r_pad))
    s = parselmouth.Sound(waveform, sampling_frequency=samplerate).to_pitch_ac(
        time_step=time_step, voicing_threshold=0.6, pitch_floor=f0_min, pitch_ceiling=f0_max
    )
    assert np.abs(s.t1 - 1.5 / f0_min) < 0.001
    f0 = s.selected_array["frequency"].astype(np.float32)
    if len(f0) < length:
        f0 = np.pad(f0, (0, length - len(f0)))
    f0 = f0[:length]
    uv = f0 == 0
    if interp_uv:
        f0, uv = interp_f0(f0, uv)
    return f0, uv


def get_energy_librosa(waveform, length, *, hop_size, win_size, domain="db"):
    """
    Definition of energy: RMS of the waveform, in dB representation
    :param waveform: [T]
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param win_size: Window size, in number of samples
    :param domain: db or amplitude
    :return: energy
    """
    energy = librosa.feature.rms(y=waveform, frame_length=win_size, hop_length=hop_size)[0]
    if len(energy) < length:
        energy = np.pad(energy, (0, length - len(energy)))
    energy = energy[:length]
    if domain == "db":
        energy = librosa.amplitude_to_db(energy)
    elif domain == "amplitude":
        pass
    else:
        raise ValueError(f"Invalid domain: {domain}")
    return energy


def get_breathiness(
    waveform: Union[np.ndarray, DecomposedWaveform],
    samplerate,
    f0,
    length,
    *,
    hop_size=None,
    fft_size=None,
    win_size=None
):
    """
    Definition of breathiness: RMS of the aperiodic part, in dB representation
    :param waveform: All other analysis parameters will not take effect if a DeconstructedWaveform is given
    :param samplerate: sampling rate
    :param f0: reference f0
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param fft_size: Number of fft bins
    :param win_size: Window size, in number of samples
    :return: breathiness
    """
    if not isinstance(waveform, DecomposedWaveform):
        waveform = DecomposedWaveform(
            waveform=waveform, samplerate=samplerate, f0=f0, hop_size=hop_size, fft_size=fft_size, win_size=win_size
        )
    waveform_ap = waveform.aperiodic()
    breathiness = get_energy_librosa(
        waveform_ap, length=length, hop_size=waveform.hop_size, win_size=waveform.win_size
    )
    return breathiness


def get_voicing(
    waveform: Union[np.ndarray, DecomposedWaveform],
    samplerate,
    f0,
    length,
    *,
    hop_size=None,
    fft_size=None,
    win_size=None
):
    """
    Definition of voicing: RMS of the harmonic part, in dB representation
    :param waveform: All other analysis parameters will not take effect if a DeconstructedWaveform is given
    :param samplerate: sampling rate
    :param f0: reference f0
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param fft_size: Number of fft bins
    :param win_size: Window size, in number of samples
    :return: voicing
    """
    if not isinstance(waveform, DecomposedWaveform):
        waveform = DecomposedWaveform(
            waveform=waveform, samplerate=samplerate, f0=f0, hop_size=hop_size, fft_size=fft_size, win_size=win_size
        )
    waveform_sp = waveform.harmonic()
    voicing = get_energy_librosa(waveform_sp, length=length, hop_size=waveform.hop_size, win_size=waveform.win_size)
    return voicing


def get_tension_base_harmonic(
    waveform: Union[np.ndarray, DecomposedWaveform],
    samplerate,
    f0,
    length,
    *,
    hop_size=None,
    fft_size=None,
    win_size=None,
    domain="logit"
):
    """
    Definition of tension: radio of the real harmonic part (harmonic part except the base harmonic)
    to the full harmonic part.
    :param waveform: All other analysis parameters will not take effect if a DeconstructedWaveform is given
    :param samplerate: sampling rate
    :param f0: reference f0
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param fft_size: Number of fft bins
    :param win_size: Window size, in number of samples
    :param domain: The domain of the final ratio representation.
     Can be 'ratio' (the raw ratio), 'db' (log decibel) or 'logit' (the reverse function of sigmoid)
    :return: tension
    """
    if not isinstance(waveform, DecomposedWaveform):
        waveform = DecomposedWaveform(
            waveform=waveform, samplerate=samplerate, f0=f0, hop_size=hop_size, fft_size=fft_size, win_size=win_size
        )
    waveform_h = waveform.harmonic()
    waveform_base_h = waveform.harmonic(0)
    energy_base_h = get_energy_librosa(
        waveform_base_h, length, hop_size=waveform.hop_size, win_size=waveform.win_size, domain="amplitude"
    )
    energy_h = get_energy_librosa(
        waveform_h, length, hop_size=waveform.hop_size, win_size=waveform.win_size, domain="amplitude"
    )
    tension = np.sqrt(np.clip(energy_h**2 - energy_base_h**2, a_min=0, a_max=None)) / (energy_h + 1e-05)
    if domain == "ratio":
        tension = np.clip(tension, a_min=0, a_max=1)
    elif domain == "db":
        tension = np.clip(tension, a_min=1e-05, a_max=1)
        tension = librosa.amplitude_to_db(tension)
    elif domain == "logit":
        tension = np.clip(tension, a_min=0.0001, a_max=1 - 0.0001)
        tension = np.log(tension / (1 - tension))
    return tension


class SinusoidalSmoothingConv1d(paddle.nn.Conv1D):
    def __init__(self, kernel_size):
        super().__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            bias=False,
            padding="same",
            padding_mode="replicate",
        )
        smooth_kernel = paddle.sin(x=paddle.to_tensor(data=np.linspace(0, 1, kernel_size).astype(np.float32) * np.pi))
        smooth_kernel /= smooth_kernel.sum()
        self.weight.data = smooth_kernel[None, None]
