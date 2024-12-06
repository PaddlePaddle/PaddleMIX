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

from typing import Dict, Tuple

import numpy as np
import paddle

from paddlemix.models.diffsinger.utils import hparams
from paddlemix.models.diffsinger.utils.infer_utils import resample_align_curve


class BaseSVSInfer:
    """
    Base class for SVS inference models.
    Subclasses should define:
    1. *build_model*:
        how to build the model;
    2. *run_model*:
        how to run the model (typically, generate a mel-spectrogram and
        pass it to the pre-built vocoder);
    3. *preprocess_input*:
        how to preprocess user input.
    4. *infer_once*
        infer from raw inputs to the final outputs
    """

    def __init__(self, device=None):
        if device is None:
            device = "gpu" if paddle.device.cuda.device_count() >= 1 else "cpu"
        self.device = device
        self.timestep = hparams["hop_size"] / hparams["audio_sample_rate"]
        self.spk_map = {}
        self.model: paddle.nn.Layer = None

    def build_model(self, ckpt_steps=None) -> paddle.nn.Layer:
        raise NotImplementedError()

    def load_speaker_mix(
        self, param_src: dict, summary_dst: dict, mix_mode: str = "frame", mix_length: int = None
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """

        :param param_src: param dict
        :param summary_dst: summary dict
        :param mix_mode: 'token' or 'frame'
        :param mix_length: total tokens or frames to mix
        :return: spk_mix_id [B=1, 1, N], spk_mix_value [B=1, T, N]
        """
        assert mix_mode == "token" or mix_mode == "frame"
        param_key = "spk_mix" if mix_mode == "frame" else "ph_spk_mix"
        summary_solo_key = "spk" if mix_mode == "frame" else "ph_spk"
        spk_mix_map = param_src.get(param_key)
        dynamic = False
        if spk_mix_map is None:
            for name in self.spk_map.keys():
                spk_mix_map = {name: 1.0}
                break
        else:
            for name in spk_mix_map:
                assert name in self.spk_map, f"Speaker '{name}' not found."
        if len(spk_mix_map) == 1:
            summary_dst[summary_solo_key] = list(spk_mix_map.keys())[0]
        elif any([isinstance(val, str) for val in spk_mix_map.values()]):
            print_mix = "|".join(spk_mix_map.keys())
            summary_dst[param_key] = f"dynamic({print_mix})"
            dynamic = True
        else:
            print_mix = "|".join([f"{n}:{'%.3f' % spk_mix_map[n]}" for n in spk_mix_map])
            summary_dst[param_key] = f"static({print_mix})"
        spk_mix_id_list = []
        spk_mix_value_list = []
        if dynamic:
            for name, values in spk_mix_map.items():
                spk_mix_id_list.append(self.spk_map[name])
                if isinstance(values, str):
                    if mix_mode == "token":
                        cur_spk_mix_value = values.split()
                        assert (
                            len(cur_spk_mix_value) == mix_length
                        ), "Speaker mix checks failed. In dynamic token-level mix, number of proportion values must equal number of tokens."
                        cur_spk_mix_value = paddle.to_tensor(data=np.array(cur_spk_mix_value, "float32")).to(
                            self.device
                        )[None]
                    else:
                        cur_spk_mix_value = paddle.to_tensor(
                            data=resample_align_curve(
                                np.array(values.split(), "float32"),
                                original_timestep=float(param_src["spk_mix_timestep"]),
                                target_timestep=self.timestep,
                                align_length=mix_length,
                            )
                        ).to(self.device)[None]
                    assert paddle.all(
                        x=cur_spk_mix_value >= 0.0
                    ), f"""Speaker mix checks failed.
Proportions of speaker '{name}' on some {mix_mode}s are negative."""
                else:
                    assert (
                        values >= 0.0
                    ), f"""Speaker mix checks failed.
Proportion of speaker '{name}' is negative."""
                    cur_spk_mix_value = paddle.full(shape=(1, mix_length), fill_value=values, dtype="float32")
                spk_mix_value_list.append(cur_spk_mix_value)
            spk_mix_id = paddle.to_tensor(data=spk_mix_id_list, dtype="int64").to(self.device)[None, None]
            spk_mix_value = paddle.stack(x=spk_mix_value_list, axis=2)
            spk_mix_value_sum = paddle.sum(x=spk_mix_value, axis=2, keepdim=True)
            assert paddle.all(
                x=spk_mix_value_sum > 0.0
            ), f"""Speaker mix checks failed.
Proportions of speaker mix on some frames sum to zero."""
            spk_mix_value /= spk_mix_value_sum
        else:
            for name, value in spk_mix_map.items():
                spk_mix_id_list.append(self.spk_map[name])
                assert (
                    value >= 0.0
                ), f"""Speaker mix checks failed.
Proportion of speaker '{name}' is negative."""
                spk_mix_value_list.append(value)
            spk_mix_id = paddle.to_tensor(data=spk_mix_id_list, dtype="int64").to(self.device)[None, None]
            spk_mix_value = paddle.to_tensor(data=spk_mix_value_list, dtype="float32").to(self.device)[None, None]
            spk_mix_value_sum = spk_mix_value.sum()
            assert (
                spk_mix_value_sum > 0.0
            ), f"""Speaker mix checks failed.
Proportions of speaker mix sum to zero."""
            spk_mix_value /= spk_mix_value_sum
        return spk_mix_id, spk_mix_value

    def preprocess_input(self, param: dict, idx=0) -> Dict[str, paddle.Tensor]:
        raise NotImplementedError()

    def forward_model(self, sample: Dict[str, paddle.Tensor]):
        raise NotImplementedError()

    def run_inference(self, params, **kwargs):
        raise NotImplementedError()
