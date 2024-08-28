# MiniMonkey 模型

## 1. 模型介绍

[MiniMonkey](https://github.com/Yuliang-Liu/Monkey/blob/main/project/mini_monkey/) 是基于 InternVL2 的专用于OCR文档理解的多模态大模型。


## 2 环境准备

1） [安装PaddleNLP develop分支](https://github.com/PaddlePaddle/PaddleNLP)

2）[安装 PaddleMix 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/b4f97ff859e1964c839fc5fab94f7ba63b1e5959?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

## 3. 快速开始
完成环境准备后，我们目前提供单轮对话方式使用：

## 3.1. 图片预测
```bash
python paddlemix/examples/minimonkey/chat_demo_minimonkey.py \
    --model_name_or_path "HUST-VLRLab/Mini-Monkey" \
    --image_path 'path/to/image.jpg' \
    --text "Read the all text in the image."
```
可配置参数说明：
  * `model_name_or_path`: 指定 minimonkey 的模型名字或权重路径以及tokenizer组件，默认 HUST-VLRLab/Mini-Monkey
  * `image_path`: 指定图片路径
  * `text`: 用户指令, 例如 "Read the all text in the image."

## 4 模型微调
```bash
sh paddlemix/examples/minimonkey/shell/internvl2.0/2nd_finetune/minimonkey_2b_internlm2_1_8b_dynamic_res_2nd_finetune_full.sh
```


### 参考文献
```BibTeX
@article{huang2024mini,
  title={Mini-Monkey: Multi-Scale Adaptive Cropping for Multimodal Large Language Models},
  author={Huang, Mingxin and Liu, Yuliang and Liang, Dingkang and Jin, Lianwen and Bai, Xiang},
  journal={arXiv preprint arXiv:2408.02034},
  year={2024}
}
```
