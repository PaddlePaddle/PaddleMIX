# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os

import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    return parser


def combine_npz(logdir):
    save_npzs = [npz for npz in os.listdir(logdir)]
    npzs = []
    for save_npz in save_npzs:
        tem_npz = np.load(os.path.join(logdir, save_npz))
        data = tem_npz["arr_0"]
        npzs.append(data)
    save_npz = np.vstack(npzs)
    np.random.shuffle(save_npz)
    np.savez(os.path.join(logdir, "sample.npz"), save_npz)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    combine_npz(args.logdir)
