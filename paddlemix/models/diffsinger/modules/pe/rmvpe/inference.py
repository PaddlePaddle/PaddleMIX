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
from basics.base_pe import BasePE
from torchaudio.transforms import Resample
from utils.infer_utils import resample_align_curve
from utils.pitch_utils import interp_f0

from .constants import *
from .model import E2E0
from .spec import MelSpectrogram
from .utils import to_local_average_f0, to_viterbi_f0


class RMVPE(BasePE):
    def __init__(self, model_path, hop_length=160):
        self.resample_kernel = {}
        self.device = "cuda" if paddle.device.cuda.device_count() >= 1 else "cpu"
        self.model = E2E0(4, 1, (2, 2)).eval().to(self.device)
        ckpt = paddle.load(path=str(model_path))
        self.model.set_state_dict(state_dict=ckpt["model"])
        self.mel_extractor = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX
        ).to(self.device)

    @paddle.no_grad()
    def mel2hidden(self, mel):
        n_frames = tuple(mel.shape)[-1]
        mel = paddle.nn.functional.pad(
            x=mel, pad=(0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="constant", pad_from_left_axis=False
        )
        hidden = self.model(mel)
        return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03, use_viterbi=False):
        if use_viterbi:
            f0 = to_viterbi_f0(hidden, thred=thred)
        else:
            f0 = to_local_average_f0(hidden, thred=thred)
        return f0

    def infer_from_audio(self, audio, sample_rate=16000, thred=0.03, use_viterbi=False):
        audio = paddle.to_tensor(data=audio).astype(dtype="float32").unsqueeze(axis=0).to(self.device)
        if sample_rate == 16000:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, 16000, lowpass_filter_width=128)
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
        mel = self.mel_extractor(audio_res, center=True)
        hidden = self.mel2hidden(mel)
        f0 = self.decode(hidden, thred=thred, use_viterbi=use_viterbi)
        return f0

    def get_pitch(self, waveform, samplerate, length, *, hop_size, f0_min=65, f0_max=1100, speed=1, interp_uv=False):
        f0 = self.infer_from_audio(waveform, sample_rate=samplerate)
        uv = f0 == 0
        f0, uv = interp_f0(f0, uv)
        hop_size = int(np.round(hop_size * speed))
        time_step = hop_size / samplerate
        f0_res = resample_align_curve(f0, 0.01, time_step, length)
        uv_res = resample_align_curve(uv.astype(np.float32), 0.01, time_step, length) > 0.5
        if not interp_uv:
            f0_res[uv_res] = 0
        return f0_res, uv_res
