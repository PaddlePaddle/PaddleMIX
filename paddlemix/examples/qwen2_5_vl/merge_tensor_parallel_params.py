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
from collections import OrderedDict

import paddle

from paddlemix.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, required=True, help="The directory of pretrained model.")
    parser.add_argument("--merge_model_path", default=None, help="The directory of merged parameters. Default to None")
    parser.add_argument("--device", type=str, default="gpu", help="Device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="dtype")
    parser.add_argument("--tensor_parallel_degree", type=int, default=2, help="tp_degree")
    return parser.parse_args()


def merge():
    args = parse_arguments()
    paddle.set_device(args.device)
    config = Qwen2_5_VLConfig.from_pretrained(args.model_name_or_path)
    config.tensor_parallel_degree = 1
    # Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, dtype=dtype, attn_implementation="flash_attention_2", config = config)

    # config = Qwen2_5_VLConfig()
    merge_mapping = Qwen2_5_VLForConditionalGeneration._get_tensor_parallel_mappings(config, is_split=False)

    # rootdir = 'work_dirs/baseline_330k_3b_bs32_1e8_debug_parallel_tp2_gpu4'
    state_dicts = []

    for i in range(args.tensor_parallel_degree):
        other_rank_file = os.path.join(args.model_name_or_path, "model_state.tp{:0>2d}.pdparams".format(i))
        state_dicts.append(paddle.load(other_rank_file))

    merged_state_dict = OrderedDict()
    for k, v in state_dicts[0].items():
        map_k = k.replace("model.", "")
        if map_k in merge_mapping:
            v_lst = []
            for j in range(args.tensor_parallel_degree):
                v_lst.append(state_dicts[j][k])
            new_v = merge_mapping[map_k](v_lst)
            print(f"key: {k}, merged weight shape: {new_v.shape}")
        else:
            new_v = v

        merged_state_dict[k] = new_v
    complete_save_file = os.path.join(args.model_name_or_path, "model_state.pdparams")
    paddle.save(merged_state_dict, complete_save_file)


if __name__ == "__main__":
    merge()
