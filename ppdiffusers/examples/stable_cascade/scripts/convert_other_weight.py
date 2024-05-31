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


def convert_weights(torch_weight_path, paddle_weight_path):
    # Load PyTorch weights
    # torch_weights = torch.load(torch_weight_path)
    torch_weights = load_file(torch_weight_path, "cpu")

    # Initialize PaddlePaddle weights
    paddle_weights = {}

    for key in torch_weights.keys():
        if "conv" in key:
            # For convolutional layers, transpose the weight tensor
            weight = torch_weights[key].numpy().transpose((2, 3, 1, 0))
        elif "fc" in key:
            # For fully connected layers, transpose the weight tensor
            weight = torch_weights[key].numpy().transpose()
        else:
            weight = torch_weights[key].numpy()

        print(key, weight.shape)
        paddle_weights[key] = weight

    # Save PaddlePaddle weights
    paddle.save(paddle_weights, paddle_weight_path)


# Example usage
convert_weights(
    "./pytorch_model/effnet_encoder.safetensors",
    "./effnet_encoder.pdparams",
)
