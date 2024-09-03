# Qwen-VL

## 1. 模型介绍

[Qwen-VL](https://arxiv.org/pdf/2308.12966.pdf) 是大规模视觉语言模型。可以以图像、文本、检测框作为输入，并以文本和检测框作为输出。Qwen-VL 系列模型的特点包括：

- **功能强大丰富**：支持多个多模态任务，包括零样本图像描述生成（Zero-shot Image Caption)、视觉问答（VQA）、细粒度视觉定位（Referring Expression Comprehension）等；
- **多语言对话模型**：支持英文、中文等多语言对话，端到端支持图片里中英双语的长文本识别；
- **多图多轮交错对话**：支持多图输入和比较，指定图片问答等；
- **细粒度识别和理解**：细粒度的文字识别、文档问答和检测框标注。

本仓库提供paddle版本的Qwen-VL-7b和Qwen-VL-Chat-7b模型。


## 2 环境准备
- **python >= 3.8**
- tiktoken
> 注：tiktoken 要求python >= 3.8
- paddlepaddle-gpu >= 2.5.1
- paddlenlp >= 2.6.1

> 注：请确保安装了以上依赖，否则无法运行。同时，需要安装 paddlemix/external_ops 下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH

## 3 快速开始
完成环境准备后，我们提供三种使用方式：

## a. 单轮预测
```bash
# qwen-vl
python paddlemix/examples/qwen_vl/run_predict.py \
--model_name_or_path "qwen-vl/qwen-vl-7b" \
--input_image "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg" \
--prompt "Generate the caption in English with grounding:" \
--dtype "bfloat16"
```
可配置参数说明：
  * `model_name_or_path`: 指定qwen_vl系列的模型名字或权重路径，默认 qwen-vl/qwen-vl-7b
  * `seed` :指定随机种子，默认1234。
  * `visual:` :设置是否可视化结果，默认True。
  * `output_dir` :指定可视化图片保存路径。
  * `dtype` :设置数据类型，默认bfloat16,支持float32、bfloat16、float16。
  * `input_image` :输入图片路径或url，默认None。
  * `prompt` :输入prompt。

## b. 多轮对话
```bash
python paddlemix/examples/qwen_vl/chat_demo.py
```

## c. 通过[Appflow](../../../applications/README.md/)调用
> 注：使用Appflow前，需要完成Appflow环境配置，请参考[依赖安装](../../../applications/README.md/#1-appflow-依赖安装)。
```python

import paddle
from paddlemix.appflow import Appflow
paddle.seed(1234)
task = Appflow(app="image2text_generation",
                   models=["qwen-vl/qwen-vl-chat-7b"])
image= "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg"
prompt = "这是什么？"
result = task(image=image,prompt=prompt)

print(result["result"])

prompt2 = "框出图中公交车的位置"
result = task(prompt=prompt2)
print(result["result"])

```

输入图片：<center><img src="https://github.com/LokeZhou/PaddleMIX/assets/13300429/95f73037-097e-4712-95be-17d5ca489f11" /></center>

prompt：“这是什么？”

输出:
```
这是一张红色城市公交车的图片，它正在道路上行驶，穿越城市。该区域似乎是一个住宅区，因为可以在背景中看到一些房屋。除了公交车之外，还有其他车辆，包括一辆汽车和一辆卡车，共同构成了交通场景。此外，图片中还显示了一一个人，他站在路边，可能是在等待公交车或进行其他活动。
```
prompt2：“框出图中公交车的位置”

输出:
```
<ref>公交车</ref><box>(178,280),(803,894)</box>
```
<center><img src="https://github.com/LokeZhou/PaddleMIX/assets/13300429/2ff2ebcf-b7d7-48ed-af42-ead9d2befeb4" /></center>


## 4 模型微调
我们提供 基于 PaddleMIX tool 统一微调工具链，支持全参数、lora微调，数据准备及参数配置等可参考 [tools](../../tools/README.md)
全参数微调需要A100 80G显存，lora微调支持V100 32G显存。

### 参考文献
```BibTeX
@article{Qwen-VL,
  title={Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
  author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.12966},
  year={2023}
}
```
