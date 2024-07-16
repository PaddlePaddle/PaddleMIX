# LLaVA

## 1. 模型介绍

[LLaVA](https://arxiv.org/pdf/2310.03744.pdf) 是基于大规模语言模型 llama 的视觉语言模型。支持多个多模态任务，包括零样本图像描述生成（Zero-shot Image Caption）、视觉问答（VQA）、细粒度视觉定位（Referring Expression Comprehension）等任务。

其性能优于其他模型，在多个任务上取得了更好的效果。

<p align="center">
  <img src="https://github.com/haotian-liu/LLaVA/blob/main/images/llava_v1_5_radar.jpg" align="middle" width = "600" />
</p>

注：图片引用自[LLaVA](https://github.com/haotian-liu/LLaVA).


## 2. 安装依赖

* `paddlenlp_ops`依赖安装

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
pip install -e .
cd csrc
python setup_cuda.py install
```

* `fused_ln`需要安装[此目录](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/gpt-3/external_ops)下的自定义OP, `python setup.py install`

## 3. 示例

### 3.1 转出静态图推理所需的视觉模型和语言模型

* 在`PaddleMIX`目录下，执行转换脚本，得到视觉模型部分静态图

```bash
#!/bin/bash

python deploy/llava/export_model.py \
    --model_name_or_path "paddlemix/llava/llava-v1.5-7b" \
    --save_path "./llava_static" \
    --fp16
```


### 3.2 静态图推理

* 在`PaddleMIX`目录下，运行执行脚本，进行静态图推理

```bash
#!/bin/bash

python deploy/llava/run_static_predict.py --model_name_or_path "paddlemix/llava/llava-v1.5-7b" \
--image_file "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg" \
--first_model_path "llava_static/encode_image/clip"  \
--second_model_path "llava_static/encode_text/llama" \
--fp16

```
