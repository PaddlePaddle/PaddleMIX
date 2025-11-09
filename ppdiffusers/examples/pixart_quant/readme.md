# Pixart Quant

本项目基于 **PaddlePaddle (ppdiffusers)** 和 **QDiff** 实现扩散模型的量化与推理加速。  
整体流程分为三个阶段：**校准数据生成（calib_data）**、**参数调优（ptq）**、**推理（inference）**。

---

## 🧩 1. 环境配置

### 1.1 基础环境

安装示例：
```bash
# 创建虚拟环境
conda create -n qdiff python=3.9
conda activate qdiff

# 安装 PaddlePaddle GPU 版本(以ppdiffusers官方文档为准)
pip install paddlepaddle-gpu==2.6.0.post117 -f https://www.paddlepaddle.org.cn/whl/mkl/stable.html

# 安装 ppdiffusers 及相关库(以ppdiffusers官方文档为准)
pip install ppdiffusers 

# 安装 qdiff (本地版本)
cd ./quant_utils
pip install -e .
```
## 🚀 2. 运行方法

整个流程分为 三个阶段。建议依次运行。

### 2.1 阶段一：生成校准数据（calib_data）

该阶段用于提取模型中关键层的特征分布，用于后续量化参数调优。
```python
CUDA_VISIBLE_DEVICES=$GPU_ID  
python get_calib_data.py \
--quant-config "./configs/${CFG}" \
--log "./logs/${LOG}"  \
--prompt $PROMPT_PATH
```

输出：

./logs/${LOG} 文件夹中保存特征 Tensor。

### 2.2 阶段二：确定量化参数（PTQ, Post-Training Quantization）

该阶段根据校准数据进行量化参数优化、scale 校正。
```python
CUDA_VISIBLE_DEVICES=$GPU_ID 
python ptq.py \
--quant-config "./configs/${CFG}" \
--log "./logs/${LOG}"
```

输出：

./logs/${LOG} 保存优化后的量化参数。

### 2.3 阶段三：量化推理（Inference）

加载量化模型参数并进行推理测试。
```python
CUDA_VISIBLE_DEVICES=$GPU_ID 
python quant_inference.py \
--quant-config "./configs/${CFG}" \
--log "./logs/${LOG}"
```

输出：

生成的图像保存在 ./logs/${LOG}/generated_images 目录。

以上可通过直接调用main.sh实现
```bash
. example/pixart/main.sh
```

## 4.技术文章
This repo's main methods come from our ICLR'25 paper: [ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation](https://arxiv.org/abs/2406.02540).