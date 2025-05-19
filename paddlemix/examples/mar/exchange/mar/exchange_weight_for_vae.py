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


# vae中无2维矩阵，无需转置，直接遍历存储后，转换格式即可
import os

parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 5)))

import sys

sys.path.append(parent_path)
import argparse

import paddle
import torch
from models.mar.vae import AutoencoderKL

parser = argparse.ArgumentParser()
parser.add_argument("--torch_path", type=str, default="kl16.ckpt", help="path of the torch_model")
parser.add_argument(
    "--paddle_path", type=str, default="exchange/mar/VAE_kl16_ckpt.pdparams", help="path of the paddle_model"
)

args = parser.parse_args()

torch_path = args.torch_path
paddle_path = args.paddle_path


def convert_torch_to_paddle(torch_path, paddle_path):
    # 加载Torch权重到CPU
    loaded = torch.load(torch_path, map_location="cpu")
    paddle_model = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4))

    torch_weights = loaded["model"]
    paddle_dict = {}
    for key, value in torch_weights.items():
        print(key)
        paddle_dict[key] = value.cpu().detach().numpy()
    paddle_model.set_state_dict(paddle_dict)
    # 保存为Paddle权重
    paddle.save(paddle_model.state_dict(), paddle_path)
    print(f"转换完成，权重已保存至 {paddle_path}")


# 替换路径
convert_torch_to_paddle(torch_path, paddle_path)
