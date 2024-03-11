# CogVLM

## 1. 模型介绍

该模型是 [CogVLM](https://arxiv.org/abs/2311.03079) 的 paddle 实现。

[CogVLM](https://arxiv.org/abs/2311.03079) 是一个强大的开源视觉语言模型（VLM）。CogVLM-17B拥有100亿的视觉参数和70亿的语言参数。

CogVLM-17B在10个经典的跨模态基准测试中取得了最佳性能，包括 NoCaps, Flicker30k captioning, RefCOCO, RefCOCO+, RefCOCOg, Visual7W, GQA, ScienceQA, VizWiz VQA and TDIUC, 并在 VQAv2, OKVQA, TextVQA, COCO 字幕等方面排名第二., 超越或匹敌 PaLI-X 55B. CogVLM还可以和你聊关于图片的话题。

本仓库提供paddle版本的 cogvlm-chat 模型

## 2. 环境准备

1） [安装PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

2）[安装 PaddleMix 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/b4f97ff859e1964c839fc5fab94f7ba63b1e5959?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

## 3. 快速开始
完成环境准备后，我们目前提供多轮对话方式使用：

```bash
python paddlemix/examples/cogvlm/chat_demo.py \
--from_pretrained "THUDM/cogvlm-chat"
```

可配置参数说明：
  * `from_pretrained`: 指定cogvlm的模型名字或权重路径以及tokenizer，默认 THUDM/cogvlm-chat
