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

from ppdiffusers.utils import smart_load


def merge_weights(unet, lora_state_dict_or_path, ratio=1.0):
    if isinstance(lora_state_dict_or_path, dict):
        lora_state_dict = lora_state_dict_or_path
    else:
        lora_state_dict = smart_load(lora_state_dict_or_path, map_location=paddle.get_device())
    ckpt_state_dict = unet.state_dict()

    def replace_name(name):
        name = ".".join(name.replace("lora_unet_", "").split(".")[0].split("_"))
        if "input.blocks" in name:
            name = name.replace("input.blocks", "down_blocks")
        else:
            name = name.replace("down.blocks", "down_blocks")
        if "middle.block" in name:
            name = name.replace("middle.block", "mid_block")
        else:
            name = name.replace("mid.block", "mid_block")
        if "output.blocks" in name:
            name = name.replace("output.blocks", "up_blocks")
        else:
            name = name.replace("up.blocks", "up_blocks")
        name = name.replace("emb.layers", "time_emb_proj")
        name = name.replace("transformer.blocks", "transformer_blocks")
        name = name.replace("time.emb.proj", "time_emb_proj")
        name = name.replace("conv.shortcut", "conv_shortcut")
        name = name.replace(".proj.in", ".proj_in")
        name = name.replace(".proj.out", ".proj_out")
        name = name.replace(".to.q", ".to_q")
        name = name.replace(".to.k", ".to_k")
        name = name.replace(".to.v", ".to_v")
        name = name.replace(".to.out", ".to_out")
        return name

    layer_names = sorted(list(set([replace_name(key) for key in lora_state_dict])))
    for l in layer_names:
        weight_name = l + ".weight"
        weight = ckpt_state_dict.get(weight_name, None)
        if weight is None:
            print(f"Skipping {weight_name} because it is not in the checkpoint.")
            continue

        alpha_name = "lora_unet_" + l.replace(".", "_") + ".alpha"
        alpha = lora_state_dict[alpha_name].cast(weight.dtype)

        lora_down_name = "lora_unet_" + l.replace(".", "_") + ".lora_down.weight"
        weight_down = lora_state_dict[lora_down_name].cast(weight.dtype)

        lora_up_name = "lora_unet_" + l.replace(".", "_") + ".lora_up.weight"
        weight_up = lora_state_dict[lora_up_name].cast(weight.dtype)
        rank: float = float(weight_down.shape[0])
        scale: float = alpha / rank

        if len(weight_down.shape) == 4:
            if weight_down.shape[2:4] == [1, 1]:
                # conv2d 1x1
                ckpt_state_dict[weight_name].copy_(
                    weight
                    + ratio * paddle.matmul(weight_up.squeeze(), weight_down.squeeze()).unsqueeze([-1, -2]) * scale,
                    False,
                )
            else:
                # conv2d 3x3
                ckpt_state_dict[weight_name].copy_(
                    weight
                    + ratio
                    * paddle.nn.functional.conv2d(weight_down.transpose([1, 0, 2, 3]), weight_up).transpose(
                        [1, 0, 2, 3]
                    )
                    * scale,
                    False,
                )
        else:
            # linear
            ckpt_state_dict[weight_name].copy_(weight + ratio * paddle.matmul(weight_up, weight_down).T * scale, False)
    del lora_state_dict
