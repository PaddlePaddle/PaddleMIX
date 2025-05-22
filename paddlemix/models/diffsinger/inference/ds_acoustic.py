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

import json
import pathlib
import sys
from collections import OrderedDict
from typing import Dict

import numpy as np

import paddle
import tqdm

from paddlemix.models.diffsinger.basics.base_svs_infer import BaseSVSInfer
from paddlemix.models.diffsinger.modules.fastspeech.param_adaptor import (
    VARIANCE_CHECKLIST,
)
from paddlemix.models.diffsinger.modules.fastspeech.tts_modules import LengthRegulator
from paddlemix.models.diffsinger.modules.toplevel import (
    DiffSingerAcoustic,
    ShallowDiffusionOutput,
)
from paddlemix.models.diffsinger.modules.vocoders.registry import VOCODERS
from paddlemix.models.diffsinger.utils import load_ckpt
from paddlemix.models.diffsinger.utils.hparams import hparams
from paddlemix.models.diffsinger.utils.infer_utils import (
    cross_fade,
    resample_align_curve,
    save_wav,
)
from paddlemix.models.diffsinger.utils.phoneme_utils import build_phoneme_list
from paddlemix.models.diffsinger.utils.text_encoder import TokenTextEncoder


class DiffSingerAcousticInfer(BaseSVSInfer):
    def __init__(self, device=None, load_model=True, load_vocoder=True, ckpt_steps=None):
        super().__init__(device=device)
        if load_model:
            self.variance_checklist = []
            self.variances_to_embed = set()
            if hparams.get("use_energy_embed", False):
                self.variances_to_embed.add("energy")
            if hparams.get("use_breathiness_embed", False):
                self.variances_to_embed.add("breathiness")
            if hparams.get("use_voicing_embed", False):
                self.variances_to_embed.add("voicing")
            if hparams.get("use_tension_embed", False):
                self.variances_to_embed.add("tension")
            self.ph_encoder = TokenTextEncoder(vocab_list=build_phoneme_list())
            if hparams["use_spk_id"]:
                with open(pathlib.Path(hparams["work_dir"]) / "spk_map.json", "r", encoding="utf8") as f:
                    self.spk_map = json.load(f)
                assert isinstance(self.spk_map, dict) and len(self.spk_map) > 0, "Invalid or empty speaker map!"
                assert len(self.spk_map) == len(set(self.spk_map.values())), "Duplicate speaker id in speaker map!"
            self.model = self.build_model(ckpt_steps=ckpt_steps)
            self.lr = LengthRegulator().to(self.device)
        if load_vocoder:
            self.vocoder = self.build_vocoder()

    def build_model(self, ckpt_steps=None):
        model = DiffSingerAcoustic(vocab_size=len(self.ph_encoder), out_dims=hparams["audio_num_mel_bins"])
        model.eval()
        model = model.to(self.device)
        load_ckpt(
            model, hparams["work_dir"], ckpt_steps=ckpt_steps, prefix_in_ckpt="model", strict=True, device=self.device
        )
        return model

    def build_vocoder(self):
        if hparams["vocoder"] in VOCODERS:
            vocoder = VOCODERS[hparams["vocoder"]]()
        else:
            vocoder = VOCODERS[hparams["vocoder"].split(".")[-1]]()
        vocoder.to_device(self.device)
        return vocoder

    def preprocess_input(self, param, idx=0):
        """
        :param param: one segment in the .ds file
        :param idx: index of the segment
        :return: batch of the model inputs
        """
        batch = {}
        summary = OrderedDict()
        txt_tokens = paddle.to_tensor(data=[self.ph_encoder.encode(param["ph_seq"])], dtype="int64").to(self.device)
        batch["tokens"] = txt_tokens
        ph_dur = paddle.to_tensor(data=np.array(param["ph_dur"].split(), np.float32)).to(self.device)
        ph_acc = paddle.round(paddle.cumsum(x=ph_dur, axis=0) / self.timestep + 0.5).astype(dtype="int64")
        durations = paddle.diff(x=ph_acc, axis=0, prepend=paddle.to_tensor(data=[0], dtype="int64").to(self.device))[
            None
        ]
        mel2ph = self.lr(durations, txt_tokens == 0)
        batch["mel2ph"] = mel2ph
        length = mel2ph.shape[1]
        summary["tokens"] = txt_tokens.shape[1]
        summary["frames"] = length
        summary["seconds"] = "%.2f" % (length * self.timestep)
        if hparams["use_spk_id"]:
            spk_mix_id, spk_mix_value = self.load_speaker_mix(
                param_src=param, summary_dst=summary, mix_mode="frame", mix_length=length
            )
            batch["spk_mix_id"] = spk_mix_id
            batch["spk_mix_value"] = spk_mix_value
        batch["f0"] = paddle.to_tensor(
            data=resample_align_curve(
                np.array(param["f0_seq"].split(), np.float32),
                original_timestep=float(param["f0_timestep"]),
                target_timestep=self.timestep,
                align_length=length,
            )
        ).to(self.device)[None]
        for v_name in VARIANCE_CHECKLIST:
            if v_name in self.variances_to_embed:
                batch[v_name] = paddle.to_tensor(
                    data=resample_align_curve(
                        np.array(param[v_name].split(), np.float32),
                        original_timestep=float(param[f"{v_name}_timestep"]),
                        target_timestep=self.timestep,
                        align_length=length,
                    )
                ).to(self.device)[None]
                summary[v_name] = "manual"
        if hparams["use_key_shift_embed"]:
            shift_min, shift_max = hparams["augmentation_args"]["random_pitch_shifting"]["range"]
            gender = param.get("gender")
            if gender is None:
                gender = 0.0
            if isinstance(gender, (int, float, bool)):
                summary["gender"] = f"static({gender:.3f})"
                key_shift_value = gender * shift_max if gender >= 0 else gender * abs(shift_min)
                batch["key_shift"] = paddle.to_tensor(data=[key_shift_value], dtype="float32").to(self.device)[:, None]
            else:
                summary["gender"] = "dynamic"
                gender_seq = resample_align_curve(
                    np.array(gender.split(), np.float32),
                    original_timestep=float(param["gender_timestep"]),
                    target_timestep=self.timestep,
                    align_length=length,
                )
                gender_mask = gender_seq >= 0
                key_shift_seq = gender_seq * (gender_mask * shift_max + (1 - gender_mask) * abs(shift_min))
                batch["key_shift"] = paddle.clip(
                    x=paddle.to_tensor(data=key_shift_seq.astype(np.float32)).to(self.device)[None],
                    min=shift_min,
                    max=shift_max,
                )
        if hparams["use_speed_embed"]:
            if param.get("velocity") is None:
                summary["velocity"] = "default"
                batch["speed"] = paddle.to_tensor(data=[1.0], dtype="float32").to(self.device)[:, None]
            else:
                summary["velocity"] = "manual"
                speed_min, speed_max = hparams["augmentation_args"]["random_time_stretching"]["range"]
                speed_seq = resample_align_curve(
                    np.array(param["velocity"].split(), np.float32),
                    original_timestep=float(param["velocity_timestep"]),
                    target_timestep=self.timestep,
                    align_length=length,
                )
                batch["speed"] = paddle.clip(
                    x=paddle.to_tensor(data=speed_seq.astype(np.float32)).to(self.device)[None],
                    min=speed_min,
                    max=speed_max,
                )
        print(f"[{idx}]\t" + ", ".join(f"{k}: {v}" for k, v in summary.items()))
        return batch

    @paddle.no_grad()
    def forward_model(self, sample):
        txt_tokens = sample["tokens"]
        variances = {v_name: sample.get(v_name) for v_name in self.variances_to_embed}
        if hparams["use_spk_id"]:
            spk_mix_id = sample["spk_mix_id"]
            spk_mix_value = sample["spk_mix_value"]
            spk_mix_embed = paddle.sum(
                x=self.model.fs2.spk_embed(spk_mix_id) * spk_mix_value.unsqueeze(axis=3), axis=2, keepdim=False
            )
        else:
            spk_mix_embed = None
        mel_pred: ShallowDiffusionOutput = self.model(
            txt_tokens,
            mel2ph=sample["mel2ph"],
            f0=sample["f0"],
            **variances,
            key_shift=sample.get("key_shift"),
            speed=sample.get("speed"),
            spk_mix_embed=spk_mix_embed,
            infer=True,
        )
        return mel_pred.diff_out

    @paddle.no_grad()
    def run_vocoder(self, spec, **kwargs):
        y = self.vocoder.spec2wav_torch(spec, **kwargs)
        return y[None]

    def run_inference(
        self,
        params,
        out_dir: pathlib.Path = None,
        title: str = None,
        num_runs: int = 1,
        spk_mix: Dict[str, float] = None,
        seed: int = -1,
        save_mel: bool = False,
    ):
        batches = [self.preprocess_input(param, idx=i) for i, param in enumerate(params)]
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".wav" if not save_mel else ".mel.pt"
        for i in range(num_runs):
            if save_mel:
                result = []
            else:
                result = np.zeros(0)
            current_length = 0
            for param, batch in tqdm.tqdm(zip(params, batches), desc="infer segments", total=len(params)):
                if "seed" in param:
                    paddle.seed(seed=param["seed"] & 4294967295)
                elif seed >= 0:
                    paddle.seed(seed=seed & 4294967295)
                mel_pred = self.forward_model(batch)
                if save_mel:
                    result.append({"offset": param.get("offset", 0.0), "mel": mel_pred.cpu(), "f0": batch["f0"].cpu()})
                else:
                    waveform_pred = self.run_vocoder(mel_pred, f0=batch["f0"])[0].cpu().numpy()
                    silent_length = round(param.get("offset", 0) * hparams["audio_sample_rate"]) - current_length
                    if silent_length >= 0:
                        result = np.append(result, np.zeros(silent_length))
                        result = np.append(result, waveform_pred)
                    else:
                        result = cross_fade(result, waveform_pred, current_length + silent_length)
                    current_length = current_length + silent_length + tuple(waveform_pred.shape)[0]
            if num_runs > 1:
                filename = f"{title}-{str(i).zfill(3)}{suffix}"
            else:
                filename = title + suffix
            save_path = out_dir / filename
            if save_mel:
                print(f"| save mel: {save_path}")
                paddle.save(obj=result, path=save_path)
            else:
                print(f"| save audio: {save_path}")
                save_wav(result, save_path, hparams["audio_sample_rate"])
