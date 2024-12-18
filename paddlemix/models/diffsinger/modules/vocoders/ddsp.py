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

import pathlib
import sys
import numpy as np
import paddle
import yaml

from librosa.filters import mel as librosa_mel_fn

from paddlemix.models.diffsinger.basics.base_vocoder import BaseVocoder
from paddlemix.models.diffsinger.modules.vocoders.registry import register_vocoder
from paddlemix.models.diffsinger.utils.hparams import hparams
from paddlemix.models.diffsinger.utils import paddle_aux

class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_model(model_path: pathlib.Path, device="cpu"):
    config_file = model_path.with_name("config.yaml")
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    print(" [Loading] " + str(model_path))
    model = paddle.jit.load(model_path)
    model.eval()
    return model, args


@register_vocoder
class DDSP(BaseVocoder):
    def __init__(self, device="cpu"):
        self.device = device
        model_path = pathlib.Path(hparams["vocoder_ckpt"])
        assert model_path.exists(), "DDSP model file is not found!"
        self.model, self.args = load_model(model_path, device=self.device)

    def to_device(self, device):
        pass

    def get_device(self):
        return self.device

    def spec2wav_torch(self, mel, f0):
        if self.args.data.sampling_rate != hparams["audio_sample_rate"]:
            print(
                "Mismatch parameters: hparams['audio_sample_rate']=",
                hparams["audio_sample_rate"],
                "!=",
                self.args.data.sampling_rate,
                "(vocoder)",
            )
        if self.args.data.n_mels != hparams["audio_num_mel_bins"]:
            print(
                "Mismatch parameters: hparams['audio_num_mel_bins']=",
                hparams["audio_num_mel_bins"],
                "!=",
                self.args.data.n_mels,
                "(vocoder)",
            )
        if self.args.data.n_fft != hparams["fft_size"]:
            print(
                "Mismatch parameters: hparams['fft_size']=",
                hparams["fft_size"],
                "!=",
                self.args.data.n_fft,
                "(vocoder)",
            )
        if self.args.data.win_length != hparams["win_size"]:
            print(
                "Mismatch parameters: hparams['win_size']=",
                hparams["win_size"],
                "!=",
                self.args.data.win_length,
                "(vocoder)",
            )
        if self.args.data.block_size != hparams["hop_size"]:
            print(
                "Mismatch parameters: hparams['hop_size']=",
                hparams["hop_size"],
                "!=",
                self.args.data.block_size,
                "(vocoder)",
            )
        if self.args.data.mel_fmin != hparams["fmin"]:
            print("Mismatch parameters: hparams['fmin']=", hparams["fmin"], "!=", self.args.data.mel_fmin, "(vocoder)")
        if self.args.data.mel_fmax != hparams["fmax"]:
            print("Mismatch parameters: hparams['fmax']=", hparams["fmax"], "!=", self.args.data.mel_fmax, "(vocoder)")
        with paddle.no_grad():
            mel = mel.to(self.device)
            mel_base = hparams.get("mel_base", 10)
            if mel_base != "e":
                assert mel_base in [10, "10"], "mel_base must be 'e', '10' or 10."
            else:
                mel = 0.434294 * mel
            f0 = f0.unsqueeze(axis=-1).to(self.device)
            signal, _, (s_h, s_n) = self.model(mel, f0)
            signal = signal.view(-1)
        return signal

    def spec2wav(self, mel, f0):
        if self.args.data.sampling_rate != hparams["audio_sample_rate"]:
            print(
                "Mismatch parameters: hparams['audio_sample_rate']=",
                hparams["audio_sample_rate"],
                "!=",
                self.args.data.sampling_rate,
                "(vocoder)",
            )
        if self.args.data.n_mels != hparams["audio_num_mel_bins"]:
            print(
                "Mismatch parameters: hparams['audio_num_mel_bins']=",
                hparams["audio_num_mel_bins"],
                "!=",
                self.args.data.n_mels,
                "(vocoder)",
            )
        if self.args.data.n_fft != hparams["fft_size"]:
            print(
                "Mismatch parameters: hparams['fft_size']=",
                hparams["fft_size"],
                "!=",
                self.args.data.n_fft,
                "(vocoder)",
            )
        if self.args.data.win_length != hparams["win_size"]:
            print(
                "Mismatch parameters: hparams['win_size']=",
                hparams["win_size"],
                "!=",
                self.args.data.win_length,
                "(vocoder)",
            )
        if self.args.data.block_size != hparams["hop_size"]:
            print(
                "Mismatch parameters: hparams['hop_size']=",
                hparams["hop_size"],
                "!=",
                self.args.data.block_size,
                "(vocoder)",
            )
        if self.args.data.mel_fmin != hparams["fmin"]:
            print("Mismatch parameters: hparams['fmin']=", hparams["fmin"], "!=", self.args.data.mel_fmin, "(vocoder)")
        if self.args.data.mel_fmax != hparams["fmax"]:
            print("Mismatch parameters: hparams['fmax']=", hparams["fmax"], "!=", self.args.data.mel_fmax, "(vocoder)")
        with paddle.no_grad():
            mel = paddle.to_tensor(data=mel, dtype="float32").unsqueeze(axis=0).to(self.device)
            mel_base = hparams.get("mel_base", 10)
            if mel_base != "e":
                assert mel_base in [10, "10"], "mel_base must be 'e', '10' or 10."
            else:
                mel = 0.434294 * mel
            f0 = paddle.to_tensor(data=f0, dtype="float32").unsqueeze(axis=0).unsqueeze(axis=-1).to(self.device)
            signal, _, (s_h, s_n) = self.model(mel, f0)
            signal = signal.view(-1)
        wav_out = signal.cpu().numpy()
        return wav_out
