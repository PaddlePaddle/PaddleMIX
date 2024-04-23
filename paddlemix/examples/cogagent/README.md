# CogAgent

## 1. 模型介绍

该模型是 [CogAgent](https://arxiv.org/abs/2312.08914) 的 paddle 实现。

[CogAgent](https://arxiv.org/abs/2312.08914)是一个基于CogVLM改进的开源视觉语言模型。CogAgent-18B拥有110亿的视觉参数和70亿的语言参数。

CogAgent-18B在9个经典的跨模态基准测试中实现了最先进的全能性能，包括VQAv2、OK-VQ、TextVQA、ST-VQA、ChartQA、infoVQA、DocVQA、MM-Vet和POPE。

除了CogVLM已有的所有功能（视觉多轮对话，视觉定位）之外，CogAgent：

1. 支持更高分辨率的视觉输入和对话式问答。它支持超高分辨率的图像输入，达到1120x1120。

2. 拥有视觉Agent的能力，能够在任何图形用户界面截图上，为任何给定任务返回一个计划，下一步行动，以及带有坐标的特定操作。

3. 增强了与图形用户界面相关的问答能力，使其能够处理关于任何图形用户界面截图的问题，例如网页、PC应用、移动应用等。

4. 通过改进预训练和微调，提高了OCR相关任务的能力。

本仓库提供paddle版本的 cogagent-chat 模型

## 2. 环境准备

1） [安装PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

2）[安装 PaddleMix 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/b4f97ff859e1964c839fc5fab94f7ba63b1e5959?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

## 3. 快速开始
完成环境准备后，我们目前提供多轮对话方式使用：

```bash
python paddlemix/examples/cogagent/chat_demo.py \
--model_name_or_path "THUDM/cogagent-chat"
```

可配置参数说明：
  * `model_name_or_path`: 指定CogAgent的模型名字或权重路径以及tokenizer，默认 THUDM/cogagent-chat
