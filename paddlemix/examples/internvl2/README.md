# InternVL2 模型

## 1. 模型介绍

[InternVL2](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/) InternVL 2.0，这是 InternVL 系列多模态大型语言模型的最新成员。InternVL 2.0 包含多种经过指令微调的模型，参数数量从 20 亿到 1080 亿不等。本仓库包含的是经过指令微调的 InternVL2-8B 模型。

与当前最先进的开源多模态大型语言模型相比，InternVL 2.0 超越了大多数开源模型。在多种能力方面，它表现出与专有商业模型相媲美的竞争力，包括文档和图表理解、信息图表问答、场景文本理解和 OCR 任务、科学和数学问题解决以及文化理解和综合多模态能力。


## 2 环境准备

1） [安装PaddleNLP develop分支](https://github.com/PaddlePaddle/PaddleNLP)

2）[安装 PaddleMix 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/b4f97ff859e1964c839fc5fab94f7ba63b1e5959?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

## 3. 快速开始
完成环境准备后，我们目前提供单轮对话方式使用：

## 3.1. 图片预测
```bash
python paddlemix/examples/internvl2/chat_demo.py \
    --model_name_or_path "OpenGVLab/InternVL2-8B" \
    --image_path 'path/to/image.jpg' \
    --text "Please describe this image in detail."
```
可配置参数说明：
  * `model_name_or_path`: 指定 internvl2 的模型名字或权重路径以及tokenizer组件，默认 OpenGVLab/InternVL2-8B
  * `image_path`: 指定图片路径
  * `text`: 用户指令, 例如 "Please describe this image in detail."

## 3.2. 视频预测
```bash
python paddlemix/examples/internvl2/chat_demo_video.py \
    --model_name_or_path "OpenGVLab/InternVL2-8B" \
    --video_path 'path/to/video.mp4' \
    --text "Please describe this video in detail."
```
可配置参数说明：
  * `model_name_or_path`: 指定 internvl2 的模型名字或权重路径以及tokenizer组件，默认 OpenGVLab/InternVL2-8B
  * `video_path`: 指定视频路径
  * `text`: 用户指令, 例如 "Please describe this video in detail."


## 4 模型微调
```bash
# 1B
sh paddlemix/examples/internvl2/shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_full.sh

# 2B
sh paddlemix/examples/internvl2/shell/internvl2.0/2nd_finetune/internvl2_2b_internlm2_1_8b_dynamic_res_2nd_finetune_full.sh

# 8B
sh paddlemix/examples/internvl2/shell/internvl2.0/2nd_finetune/internvl2_8b_internlm2_7b_dynamic_res_2nd_finetune_full.sh
```

## 5 NPU硬件训练
请参照[tools](../../tools/README.md)进行NPU硬件Paddle安装和环境变量设置，配置完成后可直接执行微调命令进行训练或预测。

### 参考文献
```BibTeX
@article{chen2023internvl,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and Li, Bin and Luo, Ping and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2312.14238},
  year={2023}
}

@article{chen2024far,
  title={How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites},
  author={Chen, Zhe and Wang, Weiyun and Tian, Hao and Ye, Shenglong and Gao, Zhangwei and Cui, Erfei and Tong, Wenwen and Hu, Kongzhi and Luo, Jiapeng and Ma, Zheng and others},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
}
```
