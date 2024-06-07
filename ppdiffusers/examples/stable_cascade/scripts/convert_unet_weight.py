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

import paddle
from safetensors.torch import load_file


def key_torch_to_paddle(key):
    mappings = [
        ("running_mean", "_mean"),
        ("running_var", "_variance"),
    ]
    for pattern, replacement in mappings:
        if pattern in key:
            key = key.replace(pattern, replacement)
    return key


def split_attention_weights(weight=None, bias=None):
    if weight is not None:
        weight_shape = weight.shape
        split_size = weight_shape[0] // 3
        q_weight = weight[:split_size]
        k_weight = weight[split_size : 2 * split_size]
        v_weight = weight[2 * split_size :]
        return q_weight.T, k_weight.T, v_weight.T
    elif bias is not None:
        split_size = bias.shape[0] // 3
        q_bias = bias[:split_size]
        k_bias = bias[split_size : 2 * split_size]
        v_bias = bias[2 * split_size :]
        return q_bias, k_bias, v_bias


def convert_weights(torch_weight_path, paddle_weight_path):
    # Load PyTorch weights
    torch_weights = load_file(torch_weight_path, "cpu")

    # Initialize PaddlePaddle weights
    paddle_weights = {}

    # Convert weights
    for key in torch_weights.keys():
        weight = torch_weights[key].numpy()

        # Special handling for BatchNorm2d layers
        if "channelwise" in key and ("gamma" in key or "beta" in key):
            weight = weight.reshape((-1, 1, 1, 1))

        t_layers = [
            "fc",
            "channelwise",
            "mapper_crp",
            "mapper_sca",
            ".mapper.",
            "txt_mapper",
            "txt_pooled_mapper",
            "clip_img_mapper",
            "kv_mapper",
            "clip_mapper",
            "out_proj",
            # "in_proj_weight",
        ]
        for t_layer in t_layers:
            if t_layer in key and "bias" not in key:
                weight = weight.transpose()
                break

        # Convert the key to PaddlePaddle format
        paddle_key = key_torch_to_paddle(key)

        # Handle attention weights
        if "attention.attn.in_proj" in key:
            if "weight" in key:
                q_weight, k_weight, v_weight = split_attention_weights(weight=weight)
                paddle_weights[paddle_key.replace("in_proj_weight", "q_proj.weight")] = paddle.to_tensor(q_weight)
                paddle_weights[paddle_key.replace("in_proj_weight", "k_proj.weight")] = paddle.to_tensor(k_weight)
                paddle_weights[paddle_key.replace("in_proj_weight", "v_proj.weight")] = paddle.to_tensor(v_weight)
            elif "bias" in key:
                q_bias, k_bias, v_bias = split_attention_weights(bias=weight)
                paddle_weights[paddle_key.replace("in_proj_bias", "q_proj.bias")] = paddle.to_tensor(q_bias)
                paddle_weights[paddle_key.replace("in_proj_bias", "k_proj.bias")] = paddle.to_tensor(k_bias)
                paddle_weights[paddle_key.replace("in_proj_bias", "v_proj.bias")] = paddle.to_tensor(v_bias)
        else:
            paddle_weights[paddle_key] = paddle.to_tensor(weight)

        paddle_weights[paddle_key] = paddle.to_tensor(weight)
        print("#", key, paddle_key, weight.shape)

    # Save PaddlePaddle weights
    paddle.save(paddle_weights, paddle_weight_path)


# Example usage
convert_weights(
    "./pytorch_model/stage_c_lite.safetensors",
    "./stage_c_lite.pdparams",
)
