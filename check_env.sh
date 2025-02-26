#!/bin/bash
# 设置错误时退出
set -e

# 查找可用的Python解释器
find_python() {
    for cmd in python3 python python3.8 python3.9 python3.10; do
        if command -v "$cmd" > /dev/null 2>&1; then
            if $cmd -c "import sys; exit(0 if sys.version_info >= (3,7) else 1)" 2>/dev/null; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

# 查找Python解释器
PYTHON_CMD=$(find_python)

if [ -z "$PYTHON_CMD" ]; then
    echo "错误: 未找到合适的Python环境 (需要Python >= 3.7)"
    exit 1
fi

echo "使用Python环境: $($PYTHON_CMD --version)"
echo "=====================Package Versions====================="

# 检查paddlepaddle版本
echo "检查paddlepaddle版本..."
if $PYTHON_CMD -c "import paddle" 2>/dev/null; then
    paddle_version=$($PYTHON_CMD -c "import paddle; print(paddle.__version__)")
    echo "当前paddlepaddle版本: $paddle_version"
    
    # 检查是否为GPU版本
    if $PYTHON_CMD -c "import paddle; print(paddle.device.is_compiled_with_cuda())" 2>/dev/null | grep -q "True"; then
        echo "paddlepaddle类型: GPU版本"
        cuda_version=$($PYTHON_CMD -c "import paddle; print(paddle.device.get_cudnn_version() / 100)")
        echo "CUDA版本: $cuda_version"
    else
        echo "⚠️ paddlepaddle类型: CPU版本，推荐使用GPU版本"
    fi
    
    if [[ "$paddle_version" == "3.0.0b2" || "$paddle_version" == "3.0.0rc" || "$paddle_version" == *"0.0.0"* ]]; then
        echo "✅ paddlepaddle版本符合要求"
    else
        echo "⚠️ 建议使用paddlepaddle 3.0.0b2或develop版本"
    fi
else
    echo "❌ 未安装paddlepaddle"
fi

# 检查paddlenlp版本
echo -e "\n检查paddlenlp版本..."
if $PYTHON_CMD -c "import paddlenlp" 2>/dev/null; then
    paddlenlp_version=$($PYTHON_CMD -c "import paddlenlp; print(paddlenlp.__version__)")
    echo "当前paddlenlp版本: $paddlenlp_version"
    if [[ "$paddlenlp_version" == "3.0.0b2" ]]; then
        echo "✅ paddlenlp版本符合要求"
    else
        echo "⚠️ 建议使用paddlenlp 3.0.0b2版本"
    fi
else
    echo "❌ 未安装paddlenlp"
fi

# 检查ppdiffusers版本
echo -e "\n检查ppdiffusers版本..."
if $PYTHON_CMD -c "import ppdiffusers" 2>/dev/null; then
    ppdiffusers_version=$($PYTHON_CMD -c "import ppdiffusers; print(ppdiffusers.__version__)")
    echo "当前ppdiffusers版本: $ppdiffusers_version"
    if [[ "$ppdiffusers_version" == "0.29.0" ]]; then
        echo "✅ ppdiffusers版本符合要求"
    else
        echo "⚠️ 建议使用ppdiffusers 0.29.0版本"
    fi
else
    echo "❌ 未安装ppdiffusers"
fi

# 检查huggingface_hub版本
echo -e "\n检查huggingface_hub版本..."
if $PYTHON_CMD -c "import huggingface_hub" 2>/dev/null; then
    hf_version=$($PYTHON_CMD -c "import huggingface_hub; print(huggingface_hub.__version__)")
    echo "当前huggingface_hub版本: $hf_version"
    if [[ "$hf_version" == "0.23.0" ]]; then
        echo "✅ huggingface_hub版本符合要求"
    else
        echo "⚠️ 建议使用huggingface_hub 0.23.0版本"
    fi
else
    echo "❌ 未安装huggingface_hub"
fi

echo -e "\n===================Version Summary===================="
echo "推荐版本:"
echo "- paddlepaddle: 3.0.0b2或develop版本"
echo "- paddlenlp: 3.0.0b2"
echo "- ppdiffusers: 0.29.0"
echo "- huggingface_hub: 0.23.0"
echo "===================================================="