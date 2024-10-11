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

# 安装依赖包
echo "安装依赖包..."
pip install -r requirements.txt

# 安装自定义算子
echo "安装自定义算子..."
cd paddlemix/external_ops
python setup.py install
cd ../../

echo "安装完成!"