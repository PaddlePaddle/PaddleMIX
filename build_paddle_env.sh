#!/bin/bash

# 设置错误时退出
set -e

echo "开始安装 paddlepaddle ..."
# 检测 CUDA 版本并安装相应的 paddlepaddle
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "检测到 CUDA 版本: $cuda_version"
    if [[ "$cuda_version" == "11.2" ]]; then
        echo "安装 CUDA 11.2 版本的 paddlepaddle..."
        python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu112/
    elif [[ "$cuda_version" == "11.6" ]]; then
        echo "安装 CUDA 11.6 版本的 paddlepaddle..."
        python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu116/
    elif [[ "$cuda_version" == "11.7" ]]; then
        echo "安装 CUDA 11.7 版本的 paddlepaddle..."
        python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu117/
    elif [[ "$cuda_version" == "11.8" ]]; then
        echo "安装 CUDA 11.8 版本的 paddlepaddle..."
        python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    elif [[ "$cuda_version" == "12.3" ]]; then
        echo "安装 CUDA 12.3 版本的 paddlepaddle..."
        python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
    else
        echo "警告: 不支持的 CUDA 版本。请手动安装适合您系统的 paddlepaddle 版本。"
    fi
else
    echo "未检测到 CUDA。安装 CPU 版本的 paddlepaddle..."
    pip install paddlepaddle-gpu==3.0.0b1
fi

echo "安装完成!"