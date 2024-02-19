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

import math
import warnings
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.audio.features import MelSpectrogram
from paddlenlp.transformers.roberta.tokenizer import RobertaTokenizer

from ..clap_module.clap import create_clap_model


def get_audio_features(audio_data, mel, max_len, data_truncating, data_filling, audio_cfg):
    """
    Calculate and add audio features to sample.
    Sample: a dict containing all the data of current sample.
    audio_data: a tensor of shape (T) containing audio data.
    max_len: the maximum length of audio data.
    data_truncating: the method of truncating data.
    data_filling: the method of filling data.
    audio_cfg: a dict containing audio configuration. Comes from model_cfg['audio_cfg'].
    """
    sample = {}

    # assert audio_data.size(-1) <= max_len, str(audio_data.size())

    # split to three parts
    chunk_frames = max_len // audio_cfg["hop_size"] + 1  # the +1 related to how the spectrogram is computed
    mel = mel[:chunk_frames]

    audio_data = audio_data[..., :max_len]
    sample["mel_fusion"] = mel
    longer = paddle.to_tensor([True], dtype="bool")

    sample["longer"] = longer
    sample["waveform"] = audio_data

    return sample


def _get_sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: Optional[float] = None,
    dtype: Optional[paddle.dtype] = None,
):
    if not (int(orig_freq) == orig_freq and int(new_freq) == new_freq):
        raise Exception("Frequencies must be of integer type to ensure quality resampling computation. ")

    if resampling_method in ["sinc_interpolation", "kaiser_window"]:
        method_map = {
            "sinc_interpolation": "sinc_interp_hann",
            "kaiser_window": "sinc_interp_kaiser",
        }
        warnings.warn(
            f'"{resampling_method}" resampling method name is being deprecated and replaced by '
            f'"{method_map[resampling_method]}" in the next release. '
            "The default behavior remains unchanged.",
            stacklevel=3,
        )
    elif resampling_method not in ["sinc_interp_hann", "sinc_interp_kaiser"]:
        raise ValueError("Invalid resampling method: {}".format(resampling_method))

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    if lowpass_filter_width <= 0:
        raise ValueError("Low pass filter width should be positive.")
    base_freq = min(orig_freq, new_freq)
    # This will perform antialiasing filtering by removing the highest frequencies.
    base_freq *= rolloff

    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    # If orig_freq is still big after GCD reduction, most filters will be very unbalanced, i.e.,
    # they will have a lot of almost zero values to the left or to the right...
    # There is probably a way to evaluate those filters more efficiently, but this is kept for
    # future work.
    idx_dtype = dtype if dtype is not None else paddle.float64

    idx = paddle.arange(-width, width + orig_freq, dtype=idx_dtype)[None, None] / orig_freq

    t = paddle.arange(0, -new_freq, -1, dtype=dtype)[:, None, None] / new_freq + idx
    t *= base_freq
    t = t.clip_(-lowpass_filter_width, lowpass_filter_width)

    if resampling_method == "sinc_interp_hann":
        window = paddle.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    else:
        # sinc_interp_kaiser
        if beta is None:
            beta = 14.769656459379492
        beta_tensor = paddle.to_tensor(float(beta))
        window = paddle.i0(beta_tensor * paddle.sqrt(1 - (t / lowpass_filter_width) ** 2)) / paddle.i0(beta_tensor)

    t *= math.pi

    scale = base_freq / orig_freq
    kernels = paddle.where(t == 0, paddle.to_tensor(1.0, dtype=t.dtype), t.sin() / t)
    kernels *= window * scale

    if dtype is None:
        kernels = paddle.cast(kernels, dtype=paddle.float32)

    return kernels, width


def _apply_sinc_resample_kernel(
    waveform: paddle.Tensor,
    orig_freq: int,
    new_freq: int,
    gcd: int,
    kernel: paddle.Tensor,
    width: int,
):
    if "float" not in str(waveform.dtype):
        raise TypeError(f"Expected floating point type for waveform tensor, but received {waveform.dtype}.")

    orig_freq = int(orig_freq) // gcd
    new_freq = int(new_freq) // gcd

    # pack batch
    shape = waveform.shape
    waveform = waveform.reshape([-1, shape[-1]])

    num_wavs, length = waveform.shape
    waveform = nn.functional.pad(waveform.unsqueeze(0), (width, width + orig_freq), data_format="NCL").squeeze(0)
    resampled = nn.functional.conv1d(waveform[:, None], kernel, stride=orig_freq)
    perm_shape = list(range(resampled.dim()))
    new_perm_shape = perm_shape
    new_perm_shape[1], new_perm_shape[2] = perm_shape[2], perm_shape[1]
    resampled = resampled.transpose(new_perm_shape).reshape([num_wavs, -1])
    target_length = paddle.cast(paddle.ceil(paddle.to_tensor(new_freq * length / orig_freq)), dtype="int64")
    resampled = resampled[..., :target_length]

    # unpack batch
    resampled = resampled.reshape(shape[:-1] + resampled.shape[-1:])
    return resampled


def resample(
    waveform: paddle.Tensor,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: Optional[float] = None,
) -> paddle.Tensor:
    r"""Resamples the waveform at the new frequency using bandlimited interpolation. :cite:`RESAMPLE`.

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
            Options: [``"sinc_interp_hann"``, ``"sinc_interp_kaiser"``] (Default: ``"sinc_interp_hann"``)
        beta (float or None, optional): The shape parameter used for kaiser window.

    Returns:
        Tensor: The waveform at the new frequency of dimension `(..., time).`
    """

    if orig_freq <= 0.0 or new_freq <= 0.0:
        raise ValueError("Original frequency and desired frequency should be positive")

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
        waveform.dtype,
    )
    resampled = _apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd, kernel, width)
    return resampled


class CLAPAudioEmbeddingClassifierFreev2(nn.Layer):
    def __init__(
        self,
        pretrained_path="",
        enable_cuda=False,
        sampling_rate=16000,
        embed_mode="audio",
        amodel="HTSAT-base",
        unconditional_prob=0.1,
        random_mute=False,
        max_random_mute_portion=0.5,
        training_mode=True,
    ):
        super().__init__()
        self.device = "cpu"  # The model itself is on cpu
        self.cuda = enable_cuda
        self.precision = "fp32"
        self.amodel = amodel  # or 'PANN-14'
        self.tmodel = "roberta"  # the best text encoder in our training
        self.enable_fusion = False  # False if you do not want to use the fusion model
        self.fusion_type = "aff_2d"
        self.pretrained = pretrained_path
        self.embed_mode = embed_mode
        self.embed_mode_orig = embed_mode
        self.sampling_rate = sampling_rate
        self.unconditional_prob = unconditional_prob
        self.random_mute = random_mute
        self.tokenize = RobertaTokenizer.from_pretrained("roberta-base")
        self.max_random_mute_portion = max_random_mute_portion
        self.training_mode = training_mode
        self.model, self.model_cfg = create_clap_model(
            self.amodel,
            self.tmodel,
            self.pretrained,
            precision=self.precision,
            enable_fusion=self.enable_fusion,
            fusion_type=self.fusion_type,
        )
        self.model = self.model.to(self.device)
        audio_cfg = self.model_cfg["audio_cfg"]
        self.mel_transform = MelSpectrogram(
            sr=audio_cfg["sample_rate"],
            n_fft=audio_cfg["window_size"],
            hop_length=audio_cfg["hop_size"],
            win_length=audio_cfg["window_size"],
            power=2.0,
            center=True,
            pad_mode="reflect",
            # onesided=True,
            n_mels=64,
            f_min=audio_cfg["fmin"],
            f_max=audio_cfg["fmax"],
            norm=None,
        )
        for p in self.model.parameters():
            # p.requires_grad = False
            p.stop_gradient = True
        self.unconditional_token = None
        self.model.eval()

    def get_unconditional_condition(self, batchsize):
        self.unconditional_token = self.model.get_text_embedding(self.tokenizer(["", ""]))[0:1]
        return paddle.concat([self.unconditional_token.unsqueeze(0)] * batchsize, axis=0)

    def batch_to_list(self, batch):
        ret = []
        for i in range(batch.size(0)):
            ret.append(batch[i])
        return ret

    def make_decision(self, probability):
        if float(paddle.rand([])) < probability:
            return True
        else:
            return False

    def random_uniform(self, start, end):
        val = paddle.rand([]).item()
        return start + (end - start) * val

    def _random_mute(self, waveform):
        # waveform: [bs, t-steps]
        t_steps = waveform.shape[-1]
        for i in range(waveform.shape[0]):
            mute_size = int(self.random_uniform(0, end=int(t_steps * self.max_random_mute_portion)))
            mute_start = int(self.random_uniform(0, t_steps - mute_size))
            waveform[i, mute_start : mute_start + mute_size] = 0
        return waveform

    def cos_similarity(self, waveform, text):
        # waveform: [bs, t_steps]
        original_embed_mode = self.embed_mode
        with paddle.no_grad():
            self.embed_mode = "audio"
            audio_emb = self(waveform)
            self.embed_mode = "text"
            text_emb = self(text)
            similarity = F.cosine_similarity(audio_emb, text_emb, axis=2)
        self.embed_mode = original_embed_mode
        return similarity.squeeze()

    def build_unconditional_emb(self):
        self.unconditional_token = self.model.get_text_embedding(self.tokenizer(["", ""]))[0:1]

    def forward(self, batch):
        # If you want this conditioner to be unconditional, set self.unconditional_prob = 1.0
        # If you want this conditioner to be fully conditional, set self.unconditional_prob = 0.0
        if self.model.training is True and not self.training_mode:
            print(
                "The pretrained CLAP model should always be in eval mode. Reloading model just in case you change the parameters."
            )
            self.model, self.model_cfg = create_clap_model(
                self.amodel,
                self.tmodel,
                self.pretrained,
                precision=self.precision,
                device="cuda" if self.cuda else "cpu",
                enable_fusion=self.enable_fusion,
                fusion_type=self.fusion_type,
            )
            for p in self.model.parameters():
                # p.requires_grad = False
                p.stop_gradient = True
            self.model.eval()

        if self.unconditional_token is None:
            self.build_unconditional_emb()

        # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
        if self.embed_mode == "audio":
            if not self.training:
                print("INFO: clap model calculate the audio embedding as condition")
            with paddle.no_grad():
                if self.sampling_rate != 48000:
                    batch = resample(batch, orig_freq=self.sampling_rate, new_freq=48000)
                audio_data = batch.squeeze(1)
                mel = self.mel_transform(audio_data)
                audio_dict = get_audio_features(
                    audio_data,
                    mel,
                    480000,
                    data_truncating="fusion",
                    data_filling="repeatpad",
                    audio_cfg=self.model_cfg["audio_cfg"],
                )
                # [bs, 512]
                embed = self.model.get_audio_embedding(audio_dict)
        elif self.embed_mode == "text":
            with paddle.no_grad():
                # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
                text_data = self.tokenizer(batch)

                if isinstance(batch, str) or (isinstance(batch, list) and len(batch) == 1):
                    for key in text_data.keys():
                        text_data[key] = text_data[key].unsqueeze(0)

                embed = self.model.get_text_embedding(text_data)

        embed = embed.unsqueeze(1)
        for i in range(embed.shape[0]):
            if self.make_decision(self.unconditional_prob):
                embed[i] = self.unconditional_token
        return embed.detach()

    def tokenizer(self, text):
        result = self.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pd",
            return_attention_mask=True,
        )
        return {k: v.squeeze(0) for k, v in result.items()}
