#!/bin/bash

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

# 设置错误时退出
set -e

# 查找可用的Python解释器
find_python() {
    # 按优先级尝试不同的方式查找Python
    for cmd in python3 python python3.8 python3.9 python3.10; do
        if command -v "$cmd" > /dev/null 2>&1; then
            # 检查Python版本是否满足要求（>=3.7）
            if $cmd -c "import sys; exit(0 if sys.version_info >= (3,7) else 1)" 2>/dev/null; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    
    # 检查conda环境
    if command -v conda > /dev/null 2>&1; then
        echo "检测到conda环境..." >&2
        # 列出所有conda环境中的python
        conda env list | grep -v '^#' | while read -r line; do
            env=$(echo "$line" | awk '{print $1}')
            if [ "$env" != "*" ]; then
                python_path=$(conda run -n "$env" which python 2>/dev/null || true)
                if [ -n "$python_path" ]; then
                    if $python_path -c "import sys; exit(0 if sys.version_info >= (3,7) else 1)" 2>/dev/null; then
                        echo "$python_path"
                        return 0
                    fi
                fi
            fi
        done
    fi
    
    return 1
}

# 查找Python解释器
PYTHON_CMD=$(find_python)

if [ -z "$PYTHON_CMD" ]; then
    echo "错误: 未找到合适的Python环境 (需要Python >= 3.7)"
    echo "请安装Python 3.7或更高版本"
    exit 1
fi

echo "找到Python环境: $($PYTHON_CMD --version)"

echo "开始安装paddlepaddle..."

# 检测CUDA版本并安装相应的paddlepaddle
if command -v nvcc > /dev/null 2>&1; then
    cuda_version=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "检测到CUDA版本: $cuda_version"
    
    case $cuda_version in
        "11.8")
            echo "安装CUDA 11.8版本的paddlepaddle..."
            $PYTHON_CMD -m pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
            ;;
        "12.6")
            echo "安装CUDA 12.3版本的paddlepaddle..."
            $PYTHON_CMD -m pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
            ;;
        "12.9")
            echo "安装CUDA 12.9版本的paddlepaddle..."
            $PYTHON_CMD -m pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
            ;;
        *)
            echo "警告: 不支持的CUDA版本 ($cuda_version)"
            echo "请访问 https://www.paddlepaddle.org.cn/install/quick 选择适合的版本安装"
            exit 1
            ;;
    esac
else
    echo "未检测到CUDA。安装CPU版本的paddlepaddle..."
    $PYTHON_CMD -m pip install paddlepaddle==3.1.0
fi

# 验证安装
echo "验证PaddlePaddle 3.1.0安装..."
if $PYTHON_CMD -c "import paddle; paddle.utils.run_check()"; then
    echo "PaddlePaddle 3.1.0安装成功！"
else
    echo "PaddlePaddle 3.1.0安装验证失败，请检查安装日志"
    exit 1
fi