# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import math
from typing import Optional

import paddle


def _get_sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interpolation",
    beta: Optional[float] = None,
    device: str = str("cpu").replace("cuda", "gpu"),
    dtype: Optional[paddle.dtype] = None,
):
    if not (int(orig_freq) == orig_freq and int(new_freq) == new_freq):
        raise Exception(
            "Frequencies must be of integer type to ensure quality resampling computation. To work around this, manually convert both frequencies to integer values that maintain their resampling rate ratio before passing them into the function. Example: To downsample a 44100 hz waveform by a factor of 8, use `orig_freq=8` and `new_freq=1` instead of `orig_freq=44100` and `new_freq=5512.5`."
        )
    if resampling_method not in ["sinc_interpolation", "kaiser_window"]:
        raise ValueError("Invalid resampling method: {}".format(resampling_method))
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd
    assert lowpass_filter_width > 0
    kernels = []
    base_freq = min(orig_freq, new_freq)
    base_freq *= rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    idx_dtype = dtype if dtype is not None else "float64"
    idx = paddle.arange(start=-width, end=width + orig_freq).astype(idx_dtype)
    for i in range(new_freq):
        t = (-i / new_freq + idx / orig_freq) * base_freq
        t = t.clip_(min=-lowpass_filter_width, max=lowpass_filter_width)
        if resampling_method == "sinc_interpolation":
            window = paddle.cos(x=t * math.pi / lowpass_filter_width / 2) ** 2
        else:
            if beta is None:
                beta = 14.769656459379492
            beta_tensor = paddle.to_tensor(data=float(beta))
            window = paddle.i0(beta_tensor * paddle.sqrt(x=1 - (t / lowpass_filter_width) ** 2)) / paddle.i0(
                beta_tensor
            )
        t *= math.pi
        # breakpoint()
        kernel = paddle.where(condition=t == 0, x=paddle.to_tensor(data=1.0), y=paddle.sin(x=t) / t)
        paddle.assign(paddle.multiply(kernel, window), kernel)
        #  kernel.scale_(scale=window)
        kernels.append(kernel)
    scale = base_freq / orig_freq

    kernels = paddle.stack(x=kernels).reshape((new_freq, 1, -1)).scale_(scale=scale)
    if dtype is None:
        kernels = kernels.to(dtype="float32")
    return kernels, width


def _apply_sinc_resample_kernel(waveform, orig_freq: int, new_freq: int, gcd: int, kernel, width: int):
    if not waveform.is_floating_point():
        raise TypeError(f"Expected floating point type for waveform tensor, but received {waveform.dtype}.")
    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd
    shape = waveform.shape

    waveform = waveform.reshape((-1, shape[-1]))
    num_wavs, length = waveform.shape
    waveform = paddle.nn.functional.pad(waveform.unsqueeze(1), (width, width + orig_freq), data_format="NCL").squeeze()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    resampled = paddle.nn.functional.conv1d(x=waveform[:, (None)], weight=kernel, stride=orig_freq)
    x = resampled
    perm_0 = list(range(x.ndim))
    perm_0[1] = 2
    perm_0[2] = 1
    resampled = x.transpose(perm=perm_0).reshape(num_wavs, -1)
    target_length = int(math.ceil(new_freq * length / orig_freq))
    resampled = resampled[(...), :target_length]

    resampled = resampled.reshape((shape[:-1] + resampled.shape[-1:]))
    return resampled


def resample(
    waveform,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interpolation",
    beta: Optional[float] = None,
):
    """Resamples the waveform at the new frequency using bandlimited interpolation. [:footcite:`RESAMPLE`].

    .. devices:: CPU CUDA


    Note:
        ``transforms.Resample`` precomputes and reuses the resampling kernel, so using it will result in
        more efficient computation if resampling multiple waveforms with the same resampling parameters.

    Args:
        waveform (Tensor): The input signal of dimension `(..., time)`
        orig_freq (int): The original frequency of the signal
        new_freq (int): The desired frequency
        lowpass_filter_width (int, optional): Controls the sharpness of the filter, more == sharper
            but less efficient. (Default: ``6``)
        rolloff (float, optional): The roll-off frequency of the filter, as a fraction of the Nyquist.
            Lower values reduce anti-aliasing, but also reduce some of the highest frequencies. (Default: ``0.99``)
        resampling_method (str, optional): The resampling method to use.
            Options: [``sinc_interpolation``, ``kaiser_window``] (Default: ``'sinc_interpolation'``)
        beta (float or None, optional): The shape parameter used for kaiser window.

    Returns:
        Tensor: The waveform at the new frequency of dimension `(..., time).`
    """
    assert orig_freq > 0.0 and new_freq > 0.0
    if orig_freq == new_freq:
        return waveform
    gcd = math.gcd(int(orig_freq), int(new_freq))

    kernel, width = _get_sinc_resample_kernel(
        orig_freq,
        new_freq,
        gcd,
        lowpass_filter_width,
        rolloff,
        resampling_method,
        beta,
        waveform.place,
        waveform.dtype,
    )

    resampled = _apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd, kernel, width)
    return resampled
