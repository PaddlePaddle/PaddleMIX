import os
import math
from functools import lru_cache
from typing import Union

import librosa
import numpy as np
import paddle
import torch
import torch.nn.functional as F

from .utils import exact_div
# def exact_div(x, y):
#     assert x % y == 0
#     return x // y

from librosa.filters import mel as librosa_mel_fn

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000: number of samples in a chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000: number of frames in a mel spectrogram input


def load_audio(file: str, sr: int = SAMPLE_RATE):
    x, sr = librosa.load(file, sr=sr)
    return x


def pad_or_trim(array, length_max: int = N_SAMPLES, length_min: int = N_SAMPLES // 2, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length_max:
            array = array.index_select(dim=axis, index=torch.arange(length_max, device=array.device))

        if array.shape[axis] < length_min:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length_min - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length_max:
            array = array.take(indices=range(length_max), axis=axis)

        if array.shape[axis] < length_min:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length_min - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = N_MELS) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    return torch.from_numpy(librosa_mel_fn(sr=SAMPLE_RATE,n_fft=N_FFT,n_mels=n_mels)).to(device)



@lru_cache(maxsize=None)
def mel_filters_paddle(device, n_mels: int = N_MELS) -> paddle.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels == 80, f"Unsupported n_mels: {n_mels}"
    return paddle.to_tensor(librosa_mel_fn(sr=SAMPLE_RATE,n_fft=N_FFT,n_mels=n_mels))



def custom_hann_window_paddle(window_length, periodic=True, dtype=None,):
    if dtype is None:
        dtype = 'float32'
    if periodic:
        window_length += 1
    n = paddle.arange(dtype=dtype, end=window_length)
    window = 0.5 - 0.5 * paddle.cos(x=2 * math.pi * n / (window_length - 1))
    if periodic:
        window = window[:-1]
    return window


def log_mel_spectrogram_torch(audio: Union[str, np.ndarray, torch.Tensor], n_mels: int = N_MELS):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def log_mel_spectrogram(audio: Union[str, np.ndarray, torch.Tensor], n_mels: int = N_MELS):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not paddle.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = paddle.to_tensor(audio)

    window = custom_hann_window_paddle(N_FFT)
    stft = paddle.signal.stft(audio, 
                N_FFT, 
                HOP_LENGTH, 
                window=window
                )
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters_paddle(None, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = paddle.clip(x=mel_spec, min=1e-10).log10()
    log_spec = paddle.maximum(x=log_spec, y=log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec


if __name__ == "__main__":

    x = np.random.rand(480000).astype("float32")

    y_pd = log_mel_spectrogram(x).detach().cpu().numpy()
    y_tc = log_mel_spectrogram_torch(x).detach().cpu().numpy()

    print(
        abs(y_pd - y_tc).max()
    )