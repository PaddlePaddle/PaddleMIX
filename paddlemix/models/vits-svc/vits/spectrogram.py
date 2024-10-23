import torch
# import torch.utils.data
import paddle
import math
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


# def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
#     """
#     PARAMS
#     ------
#     C: compression factor
#     """
#     return torch.log(torch.clamp(x, min=clip_val) * C)


# def dynamic_range_decompression_torch(x, C=1):
#     """
#     PARAMS
#     ------
#     C: compression factor used to compress
#     """
#     return torch.exp(x) / C


# def spectral_normalize_torch(magnitudes):
#     output = dynamic_range_compression_torch(magnitudes)
#     return output


# def spectral_de_normalize_torch(magnitudes):
#     output = dynamic_range_decompression_torch(magnitudes)
#     return output


mel_basis = {}
hann_window = {}




def custom_hann_window_torch(window_length, periodic=True, dtype=None, device=None):
    if dtype is None:
        dtype = torch.float32
    
    if device is None:
        device = torch.device('cpu')
    
    if periodic:
        window_length += 1
    
    n = torch.arange(window_length, dtype=dtype, device=device)
    window = 0.5 - 0.5 * torch.cos(2 * math.pi * n / (window_length - 1))
    
    if periodic:
        window = window[:-1]
    
    return window


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


# if __name__ == "__main__":

#     win_size = 100
#     y_custom = custom_hann_window_torch(win_size)
#     y = torch.hann_window(win_size)
#     print(abs((y - y_custom).numpy()).max())


#     y_custom = custom_hann_window_torch(win_size, periodic=False)
#     y = torch.hann_window(win_size, periodic=False)
#     print(abs((y - y_custom).numpy()).max())


#     win_size = 997
#     y_paddle = custom_hann_window_paddle(win_size).numpy()
#     y = torch.hann_window(win_size).numpy()
#     print( abs(y - y_paddle).max() )





def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec




def spectrogram_paddle(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if paddle.min(x=y) < -1.0:
        print('min value is ', paddle.min(x=y))
    if paddle.max(x=y) > 1.0:
        print('max value is ', paddle.max(x=y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.place)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = custom_hann_window_paddle(win_size, dtype=y.dtype)

    y = paddle.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
        data_format="NCL"
    )
    y = y.squeeze(1)

    spec = paddle.signal.stft(
                y, 
                n_fft, 
                hop_length=hop_size, 
                win_length=win_size, 
                window=hann_window[wnsize_dtype_device], 
                center=center, 
                pad_mode='reflect', 
                normalized=False, 
                onesided=True)

    # spec = spec.real()
    spec = paddle.stack( [spec.real(), spec.imag()], axis=-1 )
    spec = paddle.sqrt(x=spec.pow(y=2).sum(axis=-1) + 1e-06)
    
    return spec




if __name__ == "__main__":



    import numpy as np

    y = np.random.normal(loc=0.1406, scale=0.2395, size=[1, 1112384]).astype("float32")
    y_pd = paddle.to_tensor(y)
    y_tc = torch.from_numpy(y).cpu()

    hop_size = 320
    n_fft = 1024
    hop_length = 320
    win_length = 1024

    window_pd = custom_hann_window_paddle(win_length)
    window_tc = torch.hann_window(win_length)

    center = False
    pad_mode = "reflect"
    normalized = False
    onesided = True
    return_complex = False

    spec_tc = torch.stft(
        y_tc,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window_tc,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    ).cpu().numpy()


    spec_pd = paddle.signal.stft(
                y_pd, 
                n_fft, 
                hop_length=hop_length, 
                win_length=win_length, 
                window=window_pd, 
                center=center, 
                pad_mode='reflect', 
                normalized=False, 
                onesided=True)
    # spec_pd = paddle.stack( [spec_pd.real(), spec_pd.imag()], axis=-1 ).cpu().numpy()
    spec_pd = paddle.as_real(spec_pd).cpu().numpy()

    print(
        abs(spec_tc - spec_pd).max().item()
    )
    # print(spec_tc.mean().item(), spec_pd.mean().item())

    # print(
    #     abs( spec_tc.mean().numpy() - spec_pd.mean().numpy() )
    # )


# def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
#     global mel_basis
#     dtype_device = str(spec.dtype) + "_" + str(spec.device)
#     fmax_dtype_device = str(fmax) + "_" + dtype_device
#     if fmax_dtype_device not in mel_basis:
#         mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
#         mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
#             dtype=spec.dtype, device=spec.device
#         )
#     spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
#     spec = spectral_normalize_torch(spec)
#     return spec


# def mel_spectrogram_torch(
#     y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
# ):
#     if torch.min(y) < -1.0:
#         print("min value is ", torch.min(y))
#     if torch.max(y) > 1.0:
#         print("max value is ", torch.max(y))

#     global mel_basis, hann_window
#     dtype_device = str(y.dtype) + "_" + str(y.device)
#     fmax_dtype_device = str(fmax) + "_" + dtype_device
#     wnsize_dtype_device = str(win_size) + "_" + dtype_device
#     if fmax_dtype_device not in mel_basis:
#         mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
#         mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
#             dtype=y.dtype, device=y.device
#         )
#     if wnsize_dtype_device not in hann_window:
#         hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
#             dtype=y.dtype, device=y.device
#         )

#     y = torch.nn.functional.pad(
#         y.unsqueeze(1),
#         (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
#         mode="reflect",
#     )
#     y = y.squeeze(1)

#     spec = torch.stft(
#         y,
#         n_fft,
#         hop_length=hop_size,
#         win_length=win_size,
#         window=hann_window[wnsize_dtype_device],
#         center=center,
#         pad_mode="reflect",
#         normalized=False,
#         onesided=True,
#         return_complex=False,
#     )

#     spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

#     spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
#     spec = spectral_normalize_torch(spec)

#     return spec
