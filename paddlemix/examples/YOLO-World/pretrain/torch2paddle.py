# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import paddle
import torch


def torch2paddle(torch_path, paddle_path=None):
    if paddle_path is None:
        paddle_path = os.path.splitext(torch_path)[0] + ".pdparams"

    torch_state_dict = torch.load(torch_path)
    fc_names = ["fc", "mlp", "self_attn", "projection"]
    paddle_state_dict = {}

    for k in torch_state_dict["state_dict"]:
        if "num_batches_tracked" in k:
            continue
        v = torch_state_dict["state_dict"][k].detach().cpu().numpy()
        flag = [i in k for i in fc_names]

        if any(flag) and "weight" in k:
            new_shape = [1, 0] + list(range(2, v.ndim))
            v = v.transpose(new_shape)

        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        # image backbone
        k = k.replace("stage1.0", "layers1.stage1.conv_layer")
        k = k.replace("stage1.1.blocks.0", "layers2.stage1.c2f_layer.bottlenecks.0")
        k = k.replace("stage1.1.main_conv", "layers2.stage1.c2f_layer.conv1")
        k = k.replace("stage1.1.final_conv", "layers2.stage1.c2f_layer.conv2")
        k = k.replace("stage2.0", "layers3.stage2.conv_layer")
        k = k.replace("stage2.1.blocks.0", "layers4.stage2.c2f_layer.bottlenecks.0")
        k = k.replace("stage2.1.blocks.1", "layers4.stage2.c2f_layer.bottlenecks.1")
        k = k.replace("stage2.1.main_conv", "layers4.stage2.c2f_layer.conv1")
        k = k.replace("stage2.1.final_conv", "layers4.stage2.c2f_layer.conv2")
        k = k.replace("stage3.0", "layers5.stage3.conv_layer")
        k = k.replace("stage3.1.blocks.0", "layers6.stage3.c2f_layer.bottlenecks.0")
        k = k.replace("stage3.1.blocks.1", "layers6.stage3.c2f_layer.bottlenecks.1")
        k = k.replace("stage3.1.main_conv", "layers6.stage3.c2f_layer.conv1")
        k = k.replace("stage3.1.final_conv", "layers6.stage3.c2f_layer.conv2")
        k = k.replace("stage4.0", "layers7.stage4.conv_layer")
        k = k.replace("stage4.1.blocks.0", "layers8.stage4.c2f_layer.bottlenecks.0")
        k = k.replace("stage4.1.main_conv", "layers8.stage4.c2f_layer.conv1")
        k = k.replace("stage4.1.final_conv", "layers8.stage4.c2f_layer.conv2")
        k = k.replace("stage4.2.conv1", "layers9.stage4.sppf_layer.conv1")
        k = k.replace("stage4.2.conv2", "layers9.stage4.sppf_layer.conv2")
        k = k.replace(
            "stage1.1.blocks.1.conv1", "layers2.stage1.c2f_layer.bottlenecks.1.conv1"
        )
        k = k.replace(
            "stage1.1.blocks.1.conv2", "layers2.stage1.c2f_layer.bottlenecks.1.conv2"
        )
        k = k.replace(
            "stage1.1.blocks.2.conv1", "layers2.stage1.c2f_layer.bottlenecks.2.conv1"
        )
        k = k.replace(
            "stage1.1.blocks.2.conv2", "layers2.stage1.c2f_layer.bottlenecks.2.conv2"
        )
        k = k.replace(
            "stage2.1.blocks.2.conv1", "layers4.stage2.c2f_layer.bottlenecks.2.conv1"
        )
        k = k.replace(
            "stage2.1.blocks.2.conv2", "layers4.stage2.c2f_layer.bottlenecks.2.conv2"
        )
        k = k.replace(
            "stage2.1.blocks.3.conv1", "layers4.stage2.c2f_layer.bottlenecks.3.conv1"
        )
        k = k.replace(
            "stage2.1.blocks.3.conv2", "layers4.stage2.c2f_layer.bottlenecks.3.conv2"
        )
        k = k.replace(
            "stage2.1.blocks.4.conv1", "layers4.stage2.c2f_layer.bottlenecks.4.conv1"
        )
        k = k.replace(
            "stage2.1.blocks.4.conv2", "layers4.stage2.c2f_layer.bottlenecks.4.conv2"
        )
        k = k.replace(
            "stage2.1.blocks.5.conv1", "layers4.stage2.c2f_layer.bottlenecks.5.conv1"
        )
        k = k.replace(
            "stage2.1.blocks.5.conv2", "layers4.stage2.c2f_layer.bottlenecks.5.conv2"
        )
        k = k.replace(
            "stage3.1.blocks.2.conv1", "layers6.stage3.c2f_layer.bottlenecks.2.conv1"
        )
        k = k.replace(
            "stage3.1.blocks.2.conv2", "layers6.stage3.c2f_layer.bottlenecks.2.conv2"
        )
        k = k.replace(
            "stage3.1.blocks.3.conv1", "layers6.stage3.c2f_layer.bottlenecks.3.conv1"
        )
        k = k.replace(
            "stage3.1.blocks.3.conv2", "layers6.stage3.c2f_layer.bottlenecks.3.conv2"
        )
        k = k.replace(
            "stage3.1.blocks.4.conv1", "layers6.stage3.c2f_layer.bottlenecks.4.conv1"
        )
        k = k.replace(
            "stage3.1.blocks.4.conv2", "layers6.stage3.c2f_layer.bottlenecks.4.conv2"
        )
        k = k.replace(
            "stage3.1.blocks.5.conv1", "layers6.stage3.c2f_layer.bottlenecks.5.conv1"
        )
        k = k.replace(
            "stage3.1.blocks.5.conv2", "layers6.stage3.c2f_layer.bottlenecks.5.conv2"
        )
        k = k.replace(
            "stage4.1.blocks.1.conv1", "layers8.stage4.c2f_layer.bottlenecks.1.conv1"
        )
        k = k.replace(
            "stage4.1.blocks.1.conv2", "layers8.stage4.c2f_layer.bottlenecks.1.conv2"
        )
        k = k.replace(
            "stage4.1.blocks.2.conv1", "layers8.stage4.c2f_layer.bottlenecks.2.conv1"
        )
        k = k.replace(
            "stage4.1.blocks.2.conv2", "layers8.stage4.c2f_layer.bottlenecks.2.conv2"
        )

        # neck
        k = k.replace("top_down_layers.0.main_conv", "top_down_layers_0.conv1")
        k = k.replace("top_down_layers.0.final_conv", "top_down_layers_0.conv2")
        k = k.replace("top_down_layers.0.blocks", "top_down_layers_0.bottlenecks")
        k = k.replace("top_down_layers.0.attn_block", "top_down_layers_0.attn_block")
        k = k.replace("top_down_layers.1.main_conv", "top_down_layers_1.conv1")
        k = k.replace("top_down_layers.1.final_conv", "top_down_layers_1.conv2")
        k = k.replace("top_down_layers.1.blocks", "top_down_layers_1.bottlenecks")
        k = k.replace("top_down_layers.1.attn_block", "top_down_layers_1.attn_block")

        k = k.replace("bottom_up_layers.0.main_conv", "bottom_up_layers_0.conv1")
        k = k.replace("bottom_up_layers.0.final_conv", "bottom_up_layers_0.conv2")
        k = k.replace("bottom_up_layers.0.blocks", "bottom_up_layers_0.bottlenecks")
        k = k.replace("bottom_up_layers.0.attn_block", "bottom_up_layers_0.attn_block")
        k = k.replace("bottom_up_layers.1.main_conv", "bottom_up_layers_1.conv1")
        k = k.replace("bottom_up_layers.1.final_conv", "bottom_up_layers_1.conv2")
        k = k.replace("bottom_up_layers.1.blocks", "bottom_up_layers_1.bottlenecks")
        k = k.replace("bottom_up_layers.1.attn_block", "bottom_up_layers_1.attn_block")
        k = k.replace("downsample_layers.", "downsample_layers_")

        paddle_state_dict[k] = v

    paddle.save(paddle_state_dict, paddle_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Torch model to Paddle model")
    parser.add_argument("torch_path", type=str, help="Path to the Torch model file")
    parser.add_argument(
        "-p", "--paddle_path", type=str, help="Path to save the Paddle model file"
    )

    args = parser.parse_args()
    torch2paddle(args.torch_path, args.paddle_path)
