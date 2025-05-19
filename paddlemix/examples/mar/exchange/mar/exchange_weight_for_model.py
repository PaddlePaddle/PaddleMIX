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


# 本模型自己实现了注意力机制，因此qkv集中存放，与Torch版本框架结构完全相同，如调用Paddle的多头注意力API，则qkv需分开处理
# 二维线性层需要转置（class_emb不需要）

# 下载safetensors格式的权重

import os

from safetensors import safe_open

parent_path = os.path.abspath(os.path.join(__file__, *([".."] * 5)))

import sys

sys.path.append(parent_path)
import argparse

import paddle

parser = argparse.ArgumentParser()
parser.add_argument("--torch_path", type=str, default="mar-huge.safetensors", help="path of the torch_model")
parser.add_argument(
    "--paddle_path", type=str, default="exchange/mar/paddle_mar_huge.pdparams", help="path of the paddle_model"
)

args = parser.parse_args()

torch_path = args.torch_path
paddle_path = args.paddle_path


def convert_torch_to_paddle(torch_path, paddle_path):

    torch_weights = {}
    with safe_open(torch_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            torch_weights[key] = f.get_tensor(key)

    paddle_weights = {}
    for key, tensor in torch_weights.items():
        # 处理注意力qkv权重拆分
        # if 'attn.qkv.weight' in key:
        #     qkv = tensor
        #     split_size = qkv.size(0) // 3
        #     q, k, v = qkv.chunk(3, dim=0)

        #     base = key.replace('qkv.weight', 'q_proj.weight')
        #     paddle_weights[base] = q.T.numpy()

        #     base = key.replace('qkv.weight', 'k_proj.weight')
        #     paddle_weights[base] = k.T.numpy()

        #     base = key.replace('qkv.weight', 'v_proj.weight')
        #     paddle_weights[base] = v.T.numpy()
        #     continue

        # # 处理注意力qkv偏置拆分
        # if 'attn.qkv.bias' in key:
        #     qkv_bias = tensor
        #     split_size = qkv_bias.size(0) // 3
        #     q, k, v = qkv_bias.chunk(3, dim=0)

        #     base = key.replace('qkv.bias', 'q_proj.bias')
        #     paddle_weights[base] = q.numpy()

        #     base = key.replace('qkv.bias', 'k_proj.bias')
        #     paddle_weights[base] = k.numpy()

        #     base = key.replace('qkv.bias', 'v_proj.bias')
        #     paddle_weights[base] = v.numpy()
        #     continue

        # # 处理MLP层转换
        # if 'mlp.fc1.weight' in key:
        #     new_key = key.replace('fc1.weight', '0.weight')
        #     paddle_weights[new_key] = tensor.T.numpy()
        #     continue

        # if 'mlp.fc1.bias' in key:
        #     new_key = key.replace('fc1.bias', '0.bias')
        #     paddle_weights[new_key] = tensor.numpy()
        #     continue

        # if 'mlp.fc2.weight' in key:
        #     new_key = key.replace('fc2.weight', '2.weight')
        #     paddle_weights[new_key] = tensor.T.numpy()
        #     continue

        # if 'mlp.fc2.bias' in key:
        #     new_key = key.replace('fc2.bias', '2.bias')
        #     paddle_weights[new_key] = tensor.numpy()
        #     continue

        if key.endswith(".weight") and key != "class_emb.weight" and tensor.ndim == 2:
            paddle_weights[key] = tensor.T.numpy()
        else:
            paddle_weights[key] = tensor.numpy()

    # 保存为Paddle权重
    paddle.save(paddle_weights, paddle_path)
    print(f"Converted weights saved to {paddle_path}")


# 替换路径
convert_torch_to_paddle(torch_path, paddle_path)
