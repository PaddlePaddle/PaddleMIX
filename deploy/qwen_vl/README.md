# Qwen-VL

## 1. 模型简介
[Qwen-VL](https://arxiv.org/pdf/2308.12966.pdf) 是大规模视觉语言模型。可以以图像、文本、检测框作为输入，并以文本和检测框作为输出。Qwen-VL 系列模型的特点包括：

- **功能强大丰富**：支持多个多模态任务，包括零样本图像描述生成（Zero-shot Image Caption)、视觉问答（VQA）、细粒度视觉定位（Referring Expression Comprehension）等；
- **多语言对话模型**：支持英文、中文等多语言对话，端到端支持图片里中英双语的长文本识别；
- **多图多轮交错对话**：支持多图输入和比较，指定图片问答等；
- **细粒度识别和理解**：细粒度的文字识别、文档问答和检测框标注。

本目录提供paddle版本的Qwen-VL-7b静态图推理部署示例，推荐使用A100进行推理部署。

## 2. 安装依赖

* `paddlenlp_ops`依赖安装

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
pip install -e .
cd csrc
python setup_cuda.py install
```

* `fused_ln`需要安装 /PaddleNLP/model_zoo/gpt-3/external_ops 下的自定义OP, `python setup.py install`

## 3. 示例

```

### 3.1 转出静态图推理所需的视觉模型

* 在`PaddleMIX`目录下，执行转换脚本，得到视觉模型部分静态图

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/PaddleNLP/:/path/to/PaddleMIX

python deploy/qwen_vl/export_image_encoder.py \
    --model_name_or_path "qwen-vl/qwen-vl-7b-static" \
    --save_path "./checkpoints/encode_image/vision"
```

### 3.2 转出静态图推理所需的语言模型

* 在`PaddleNLP/llm`目录下，执行转换脚本，得到语言模型部分静态图

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/PaddleNLP/:/path/to/PaddleMIX:/path/to/PaddleNLP/llm

python ppredict/export_model.py \
    --model_name_or_path "qwen-vl/qwen-vl-7b-static" \
    --output_path ./checkpoints/encode_text/ \
    --dtype float16 \
    --inference_model \
    --model_prefix qwen \
    --model_type qwen-img2txt
```

### 3.3 静态图推理

* 在`PaddleMIX`目录下，运行执行脚本，进行静态图推理

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/PaddleNLP/:/path/to/PaddleMIX

python deploy/qwen_vl/run_static_predict.py \
    --first_model_path "/path/to/checkpoints/encode_image/vision" \
    --second_model_path "/path/to/checkpoints/encode_text/qwen" \
    --model_name_or_path "qwen-vl/qwen-vl-7b-static" \
```
