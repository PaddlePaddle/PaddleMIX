#!/bin/bash

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

# 安装自定义算子
echo "安装自定义算子..."
cd paddlemix/external_ops
python setup.py install
cd ../../

echo "安装完成!"