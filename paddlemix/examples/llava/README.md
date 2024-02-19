# LLaVA

## 1. 模型介绍

[LLaVA](https://arxiv.org/pdf/2310.03744.pdf) 是基于大规模语言模型 llama 的视觉语言模型。支持多个多模态任务，包括零样本图像描述生成（Zero-shot Image Caption）、视觉问答（VQA）、细粒度视觉定位（Referring Expression Comprehension）等任务。

其性能优于其他模型，在多个任务上取得了更好的效果。

<p align="center">
  <img src="https://github.com/haotian-liu/LLaVA/blob/main/images/llava_v1_5_radar.jpg" align="middle" width = "600" />
</p>

注：图片引用自[LLaVA](https://github.com/haotian-liu/LLaVA).

本仓库提供paddle版本的Llava-7b和Llava-13b模型。


## 2 环境准备
- **python >= 3.8**
- **paddlenlp >= 2.7**

## 3 快速开始
完成环境准备后，我们提供多轮对话示例：

## a. 多轮对话启动
```bash
# llava
python paddlemix/examples/llava/run_predict_multiround.py \
--model-path "paddlemix/llava-v1.5-7b" \
--image-file "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg" \
```
可配置参数说明：
  * `model_name_or_path`: 指定llava系列的模型名字或权重路径
  * `image-flie` :输入图片路径或url，默认None。

```

输入图片：<center><img src="https://github.com/LokeZhou/PaddleMIX/assets/13300429/95f73037-097e-4712-95be-17d5ca489f11" /></center>

```
USER: 描述这张照片
ASSISTANT: 这是一个照片，展示了一辆红色公交车在街道上行驶。车辆正在行驶在一个狭窄的道路上，周围有一些汽车和树木。车辆的前部有一个路灯，并且还有一个路灯在车辆的右侧。
USER: 给出公交车位置的坐标
ASSISTANT: 0.23, 0.33, 0.79, 0.78
```

## 4 模型微调
Llava 基于 PaddleMIX tool 统一微调工具链，支持全参数、lora微调，具体可参考 [tools](../../tools/README.md)

```bash
# llava lora微调
python paddlemix/tools/supervised_finetune.py paddlemix/config/llava/lora_sft_argument.json

# llava full参数微调
python paddlemix/tools/supervised_finetune.py paddlemix/config/llava/sft_argument.json
```

### 参考文献
```BibTeX
@misc{liu2024llavanext,
    title={LLaVA-NeXT: Improved reasoning, OCR, and world knowledge},
    url={https://llava-vl.github.io/blog/2024-01-30-llava-next/},
    author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Li, Bo and Zhang, Yuanhan and Shen, Sheng and Lee, Yong Jae},
    month={January},
    year={2024}
}

@misc{liu2023improvedllava,
      title={Improved Baselines with Visual Instruction Tuning},
      author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Lee, Yong Jae},
      publisher={arXiv:2310.03744},
      year={2023},
}

@misc{liu2023llava,
      title={Visual Instruction Tuning},
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={NeurIPS},
      year={2023},
}
```
