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
import gc

# @Time    : 2025/5/14 下午1:33
# @Author  : Ismoothly(1844252306@qq.com)
import os

import paddle
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Model conversion script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the original PyTorch model directory")
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Output directory for converted PaddlePaddle model"
    )
    return parser.parse_args()


def load_full_model(model_path):
    """加载完整模型到CPU内存"""
    print("Loading full model to CPU...")
    return (
        AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map=None, torch_dtype=torch.float32)
        .float()
        .cpu()
    )


def convert_parameter(key, tensor):
    """带内存优化的参数转换"""
    if tensor.device.type == "meta":
        raise RuntimeError(f"发现meta设备参数：{key}")

    # 使用内存视图减少拷贝
    np_tensor = tensor.detach().cpu().numpy()
    del tensor  # 立即释放PyTorch张量

    # 转置逻辑优化
    transpose_conditions = [
        "weight" in key,
        any(k in key for k in ["linear", "emb", "qkv", "proj", "fc"]),
        len(np_tensor.shape) == 2,
    ]
    if all(transpose_conditions):
        np_tensor = np_tensor.transpose(1, 0)

    # 使用paddle.Tensor.astype优化内存
    return paddle.to_tensor(np_tensor).astype("float32")


def convert_and_save_chunks(torch_state_dict, save_dir, chunk_size=30):
    """分块转换并直接保存到文件"""
    os.makedirs(save_dir, exist_ok=True)
    keys = list(torch_state_dict.keys())

    print(f"开始分块转换（块大小: {chunk_size}）...")
    for i in tqdm(range(0, len(keys), chunk_size)):
        chunk_keys = keys[i : i + chunk_size]

        # 分块转换
        paddle_chunk = {}
        for k in chunk_keys:
            v = torch_state_dict[k]
            try:
                paddle_chunk[k] = convert_parameter(k, v)
                del v  # 立即释放原始张量
            except Exception as e:
                raise RuntimeError(f"转换失败: {k} - {str(e)}")

        # 立即保存当前分块
        chunk_path = os.path.join(save_dir, f"model_part_{i//chunk_size}.pdparams")
        paddle.save(paddle_chunk, chunk_path)

        # 释放内存
        del paddle_chunk
        gc.collect()


def merge_chunks(save_dir):
    """合并分块文件（可选）"""
    chunk_files = sorted(
        [f for f in os.listdir(save_dir) if f.startswith("model_part_")],
        key=lambda x: int(x.split("_")[2].split(".")[0]),
    )

    full_state_dict = {}
    for fname in tqdm(chunk_files, desc="合并分块"):
        chunk = paddle.load(os.path.join(save_dir, fname))
        full_state_dict.update(chunk)
        del chunk
        gc.collect()

    # 保存最终合并文件
    paddle.save(full_state_dict, os.path.join(save_dir, "model_state.pdparams"))

    # 清理分块文件
    for fname in chunk_files:
        os.remove(os.path.join(save_dir, fname))


def main():
    # 解析命令行参数
    args = parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 保存分词器
    print("保存分词器...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.save_pretrained(args.save_dir)

    # 加载PyTorch模型
    torch_model = load_full_model(args.model_path)

    try:
        # 分块转换并保存
        convert_and_save_chunks(torch_model.state_dict(), args.save_dir, chunk_size=30)

        # 可选：合并分块文件（需要足够磁盘空间）
        # merge_chunks(args.save_dir)

        print(f"转换完成！参数保存在：{args.save_dir}")
        print(f"分块数量：{len([f for f in os.listdir(args.save_dir) if f.startswith('model_part_')])}")

    finally:
        # 清理内存
        del torch_model
        gc.collect()


if __name__ == "__main__":
    main()
