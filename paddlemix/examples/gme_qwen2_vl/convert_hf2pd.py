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


# 将huggingface上的gme-Qwen2-VL-2B-Instruct转换为pd格式
# 使用说明：
# 1.下载模型权重（https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct）
# 2.修改model_path为本地路径（注：2B模型需注释41行，7B模型则不需要）
# 3.运行该脚本


import json
import os
import shutil
import copy
import paddle
import torch
from safetensors.torch import load_file
from safetensors.numpy import save_file
#from paddlenlp.utils.log import logger
import logging as logger

#model_path = "/root/paddlejob/workspace/env_run/nifeng03/pretrain/Qwen2-VL-2B-Instruct"
model_path = "gme-Qwen2-VL-2B-Instruct"
# dst_path = "/root/code/models"
dst_path = model_path + "_pd"

# # 这里不修改，xxxxx代表随机名称，完全不会匹配到对应的key
src_prefix_key = "xxxxx."
dst_prefix_key = "deepseek_v2."

if not os.path.exists(dst_path):
    os.mkdir(dst_path)

need_transpose = {
    "up_proj.weight",
    "gate_proj.weight",
    "down_proj.weight",

    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",

    # "lm_head.weight", # 2b要注释这行，7b不需要注释


    "attn.proj.weight",
    "attn.qkv.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
    #"merger.ln_q.weight",
    "merger.mlp.0.weight",
    "merger.mlp.2.weight",
}


def check_trans(key):
    for x in need_transpose:
        if x in key:
            return True

    return False


def translate_one_safetensors(file_name):
    tensors = load_file(os.path.join(model_path, file_name))
    for key in list(tensors.keys()):
        dst_key = key.replace(src_prefix_key, dst_prefix_key)
        logger.info("{} {}".format(key, tensors[key].shape))
        if check_trans(key):
            t = tensors.pop(key).cuda().t().contiguous()
            capsule = torch.utils.dlpack.to_dlpack(t)
            t = paddle.utils.dlpack.from_dlpack(capsule)
            tensors[dst_key] = t.numpy()
        else:
            t = tensors.pop(key).cuda()
            capsule = torch.utils.dlpack.to_dlpack(t)
            t = paddle.utils.dlpack.from_dlpack(capsule)
            tensors[dst_key] = t.numpy()

            # tensors[dst_key] = paddle.to_tensor(tensors.pop(key).cuda().float().cpu().numpy(), dtype="bfloat16").numpy()
        logger.info("{} {}".format(dst_key, tensors[dst_key].shape))

    save_file(tensors, os.path.join(dst_path, file_name), metadata={"format": "np"})
    # os.remove(os.path.join(model_path, file_name))


def execute_cmd(cmd, file_path):
    cmd = cmd + " " + file_path
    os.system(cmd)


if os.path.exists(os.path.join(model_path, "model.safetensors.index.json")):
    index = json.load(open(os.path.join(model_path, "model.safetensors.index.json")))

    dst_index = copy.deepcopy(index)
    for key in list(dst_index["weight_map"].keys()):
        dst_key = key.replace(src_prefix_key, dst_prefix_key)
        dst_index["weight_map"][dst_key] = dst_index["weight_map"].pop(key)

    files = set(index["weight_map"].values())
    logger.info(files)

    for file_name in sorted(os.listdir(model_path)):
        # skip hidden files
        if file_name.startswith("."):
            continue

        logger.info(file_name)
        if file_name in files:
            # convert safetensors to safetensors(paddle)
            translate_one_safetensors(file_name)
        else:
            # copy config.json and other files
            shutil.copy(os.path.join(model_path, file_name), os.path.join(dst_path, file_name))

    json.dump(dst_index, open(os.path.join(dst_path, "model.safetensors.index.json"), "w"), indent=2)

else:
    for file_name in sorted(os.listdir(model_path)):
        # skip hidden files
        if file_name.startswith("."):
            continue

        logger.info(file_name)
        if file_name == "model.safetensors":
            # convert safetensors to safetensors(paddle)
            translate_one_safetensors(file_name)
        else:
            # copy config.json and other files
            shutil.copy(os.path.join(model_path, file_name), os.path.join(dst_path, file_name))


# execute_cmd(cmd="sed -i -e  's/Qwen2Tokenizer/QWen2Tokenizer/g' ",
#             file_path=os.path.join(dst_path, "tokenizer_config.json"))

# execute_cmd(cmd="sed -i -e  's/Qwen2ForCausalLM/QWen2ForCausalLM/g' ",
#             file_path=os.path.join(dst_path, "config.json"))

execute_cmd(cmd="sed -i -e  's/torch_dtype/dtype/g' ",
            file_path=os.path.join(dst_path, "config.json"))

execute_cmd(cmd="sed -i /transformers_version/d ",
            file_path=os.path.join(dst_path, "config.json"))

execute_cmd(cmd="sed -i /transformers_version/d ",
            file_path=os.path.join(dst_path, "generation_config.json"))
execute_cmd(cmd="sed -i '/max_new_tokens/s/,//g' ",
            file_path=os.path.join(dst_path, "generation_config.json"))


logger.info(model_path)
logger.info(dst_path)