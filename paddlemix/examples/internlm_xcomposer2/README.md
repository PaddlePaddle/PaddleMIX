# InternLM-XComposer2

## 1. 模型介绍

[InternLM-XComposer2](https://arxiv.org/abs/2401.16420) 是基于 InternLM2-7B 大语言模型研发的突破性的图文多模态大模型，具有非凡的图文写作和图像理解能力，在多种应用场景表现出色：

+ 自由指令输入的图文写作： InternLM-XComposer2 可以理解自由形式的图文指令输入，包括大纲、文章细节要求、参考图片等，为用户打造图文并貌的专属文章。生成的文章文采斐然，图文相得益彰，提供沉浸式的阅读体验。

+ 准确的图文问题解答： InternLM-XComposer2 具有海量图文知识，可以准确的回复各种图文问答难题，在识别、感知、细节描述、视觉推理等能力上表现惊人。

+ 杰出性能： InternLM-XComposer2 在13项多模态评测中大幅领先同量级多模态模型，在其中6项评测中超过 GPT-4V 和 Gemini Pro。

本仓库提供paddle版本的 InternLM-XComposer2-7b 模型。


## 2 环境准备

1） [安装PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

2）[安装 PaddleMix 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/b4f97ff859e1964c839fc5fab94f7ba63b1e5959?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

## 3. 快速开始
完成环境准备后，我们目前提供单轮对话方式使用：


## a. 单轮预测
```bash
python paddlemix/examples/internlm_xcomposer2/chat_demo.py \
--model_name_or_path "internlm/internlm-xcomposer2-7b" \
--image_path "path/to/image.jpg" \
--text "Please describe this image in detail."
```
可配置参数说明：
  * `model_name_or_path`: 指定 internlm_xcomposer2 的模型名字或权重路径以及tokenizer, processor 组件，默认 internlm/internlm-xcomposer2-7b
  * `image_path`: 指定图片路径
  * `text`: 用户指令, 例如 "Please describe this image in detail."



### 参考文献
```BibTeX
@article{internlmxcomposer2,
      title={InternLM-XComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Model},
      author={Xiaoyi Dong and Pan Zhang and Yuhang Zang and Yuhang Cao and Bin Wang and Linke Ouyang and Xilin Wei and Songyang Zhang and Haodong Duan and Maosong Cao and Wenwei Zhang and Yining Li and Hang Yan and Yang Gao and Xinyue Zhang and Wei Li and Jingwen Li and Kai Chen and Conghui He and Xingcheng Zhang and Yu Qiao and Dahua Lin and Jiaqi Wang},
      journal={arXiv preprint arXiv:2401.16420},
      year={2024}
}
```
