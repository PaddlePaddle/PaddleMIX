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

import paddle
import yaml

from .nets import CascadedNet


class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_sep_model(model_path, device="cpu"):
    model_path = pathlib.Path(model_path)
    config_file = model_path.with_name("config.yaml")
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    model = CascadedNet(args.n_fft, args.hop_length, args.n_out, args.n_out_lstm, True, is_mono=args.is_mono)
    model.to(device)
    model.set_state_dict(state_dict=paddle.load(path=str(model_path)))
    model.eval()
    return model
