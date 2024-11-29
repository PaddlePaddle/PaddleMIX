# MiniCPM-V-2_6

## 1. 模型介绍

[MiniCPM-V-2_6](https://github.com/OpenBMB/MiniCPM-V) 🔥🔥🔥 MiniCPM-V 系列中最新、最强的模型，共 8B 参数，在单图、多图、视频理解上超越 GPT-4V，在单图理解上超越GPT-4o mini、Gemini 1.5 Pro 和 Claude 3.5 Sonnet。
**本仓库支持的模型权重:**

| Model              |
|--------------------|
| openbmb/MiniCPM-V-2_6  |


## 2 环境准备

1）[安装 PaddleMIX 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/build_env.sh)

2) pip install paddlenlp==3.0.0b2
注意：Python版本最好为3.10及以上版本。

## 3 快速开始

### 推理
```bash
# 单图推理
python paddlemix/examples/minicpm-v-2_6/single_image_infer.py 

# 视频推理
python paddlemix/examples/minicpm-v-2_6/video_infer.py


```

### 参考文献
```BibTeX
@article{yao2024minicpm,
  title={MiniCPM-V: A GPT-4V Level MLLM on Your Phone},
  author={Yao, Yuan and Yu, Tianyu and Zhang, Ao and Wang, Chongyi and Cui, Junbo and Zhu, Hongji and Cai, Tianchi and Li, Haoyu and Zhao, Weilin and He, Zhihui and others},
  journal={arXiv preprint arXiv:2408.01800},
  year={2024}
}
```
