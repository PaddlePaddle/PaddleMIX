#!/bin/bash

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

# 设置错误时退出
set -e

echo "开始安装 PaddleMIX 及其依赖..."

# 安装 PaddleMIX
echo "安装 PaddleMIX..."
pip install -e .

# 安装 ppdiffusers
echo "安装 ppdiffusers..."
cd ppdiffusers
pip install -e .
cd ..
#注：ppdiffusers部分模型需要依赖 CUDA 11.2 及以上版本，如果本地机器不符合要求，建议前往 [AI Studio](https://aistudio.baidu.com/index) 进行模型训练、推理任务。
#如果希望使用**bf16**训练推理，请使用支持**bf16**的GPU，如A100。

# 安装依赖包
echo "安装依赖包..."
pip install -r requirements.txt

# 安装自定义算子，非CUDA环境（例如昇腾环境）则跳过
if command -v nvcc > /dev/null 2>&1; then
    echo "安装自定义算子..."
    cd paddlemix/external_ops
    python setup.py install
    cd ../../
else
    echo "未检测到 CUDA。跳过自定义算子安装..."
fi

echo "安装完成!"