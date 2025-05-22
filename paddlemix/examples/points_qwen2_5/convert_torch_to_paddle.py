# -*- coding: utf-8 -*-

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

# @Time    : 2025/4/25 下午11:33
# @Author  : zhaop-l(zhaop-l@glocon.com)
import argparse
import copy
import json
import os
import shutil

import paddle
import torch
from safetensors.numpy import save_file
from safetensors.torch import load_file

from paddlemix.utils.log import logger

need_transpose = {
    # —— 语言模型部分（CustomLlamaForCausalLM） ——
    "attention.query_dense.weight",
    "attention.key_value_dense.weight",
    "attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_4h_to_h.weight",
    "llm.embed_out.weight",
    # —— 双路视觉编码器部分（general_vit + ocr_vit） ——
    # self_attn
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.q_proj.weight",
    "self_attn.out_proj.weight",
    # mlp
    "mlp.fc1.weight",
    "mlp.fc2.weight",
    # —— vision_projector 重采样 / 映射层 ——
    "vision_projector.0.weight",
    "vision_projector.2.weight",
}

rename_layers = {
    "embeddings.class_embedding": "class_embedding",
    "embeddings.patch_embedding.weight": "conv1.weight",
    "embeddings.position_embedding": "positional_embedding",
    "pre_layrnorm": "ln_pre",
    "vision_model.encoder": "vision_model.transformer",
    "layer_norm1": "norm1",
    "layer_norm2": "norm2",
    "mlp.fc1": "linear1",
    "mlp.fc2": "linear2",
    "post_layernorm": "ln_post",
}


def execute_cmd(cmd, file_path):
    cmd = cmd + " " + file_path
    os.system(cmd)


def check_trans(key, _need_transpose):
    precess_list = []
    for x in _need_transpose:
        if x in key:
            precess_list.append(x)
    if len(precess_list) > 0:
        return True, precess_list
    else:
        return False, None


def translate_one_safetensors(file_name: str, dst_path: str, model_path: str):
    tensors = load_file(os.path.join(model_path, file_name))
    for key in list(tensors.keys()):
        dst_key = key
        shape_ = tensors[key].shape
        rename_flag, rename_key = check_trans(key, rename_layers)
        if rename_flag:
            for _r in rename_key:
                dst_key = dst_key.replace(_r, rename_layers[_r])
        t_flag, _ = check_trans(key, need_transpose)
        if t_flag and len(shape_) == 2:
            t = tensors.pop(key).cuda().t().contiguous()
            capsule = torch.utils.dlpack.to_dlpack(t)
            t = paddle.utils.dlpack.from_dlpack(capsule)
            tensors[dst_key] = t.numpy()
        else:
            t = tensors.pop(key).cuda()
            capsule = torch.utils.dlpack.to_dlpack(t)
            t = paddle.utils.dlpack.from_dlpack(capsule)
            tensors[dst_key] = t.numpy()

    save_file(tensors, os.path.join(dst_path, file_name), metadata={"format": "np"})


def main(args):
    model_path = args.torch_model_path
    if args.paddle_model_path is not None:
        dst_path = args.paddle_model_path
    else:
        dst_path = model_path.rstrip("/") + "_pd"
    os.makedirs(dst_path, exist_ok=True)

    logger.info(f"torch model path: {model_path}, paddle model path: {dst_path}")
    logger.info("start convert torch model to paddle model")

    if os.path.exists(os.path.join(model_path, "model.safetensors.index.json")):
        index = json.load(open(os.path.join(model_path, "model.safetensors.index.json")))
        dst_index = copy.deepcopy(index)
        files = set(index["weight_map"].values())

        for key in list(dst_index["weight_map"].keys()):
            rename_flag, rename_key = check_trans(key, rename_layers)
            dst_key = key
            if rename_flag:
                for _r in rename_key:
                    dst_key = dst_key.replace(_r, rename_layers[_r])
            dst_index["weight_map"][dst_key] = dst_index["weight_map"].pop(key)

        for file_name in sorted(os.listdir(model_path)):
            # skip hidden files
            if file_name.startswith("."):
                continue

            if file_name in files:
                # convert safetensors to safetensors(paddle)
                logger.info(f"start convert {file_name}")
                translate_one_safetensors(file_name, dst_path, model_path)
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
                translate_one_safetensors(file_name, dst_path, model_path)
            else:
                # copy config.json and other files
                shutil.copy(os.path.join(model_path, file_name), os.path.join(dst_path, file_name))

    execute_cmd(cmd="sed -i -e  's/torch_dtype/dtype/g' ", file_path=os.path.join(dst_path, "config.json"))

    execute_cmd(cmd="sed -i /transformers_version/d ", file_path=os.path.join(dst_path, "config.json"))

    logger.info(f"convert torch model to paddle model success, paddle model path: {dst_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_model_path", type=str, default="POINTS-Qwen-2-5-7B-Chat")
    parser.add_argument("--paddle_model_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
