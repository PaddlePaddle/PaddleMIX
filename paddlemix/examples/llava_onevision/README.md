# LLaVA-OneVision

## 1. 模型介绍

[LLaVA-OneVision](https://llava-vl.github.io/blog/2024-08-05-llava-onevision/) 能够处理图像、文本、图像文本交错输入和视频，是首个能够同时突破开放多模态模型在单图像、多图像和视频场景性能瓶颈的单模型。


**本仓库支持的模型权重:**

| Model              |
|--------------------|
| lmms-lab/llava-onevision-qwen2-0.5b-si  |
| lmms-lab/llava-onevision-qwen2-0.5b-ov  |
| lmms-lab/llava-onevision-qwen2-7b-si  |
| lmms-lab/llava-onevision-qwen2-7b-ov  |
| BAAI/Aquila-VL-2B-llava-qwen  |

注意：与huggingface权重同名，但权重为paddle框架的Tensor，使用`xxx.from_pretrained("lmms-lab/llava-onevision-qwen2-0.5b-si")`即可自动下载该权重文件夹到缓存目录。


## 2 环境准备

1）[安装PaddleNLP develop分支](https://github.com/PaddlePaddle/PaddleNLP?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

版本要求：paddlenlp>=3.0.0b2

2）[安装 PaddleMIX 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/b4f97ff859e1964c839fc5fab94f7ba63b1e5959?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

注意：Python版本最好为3.10及以上版本。

## 3 快速开始

### 推理
```bash
python paddlemix/examples/llava_onevision/run_predict.py
```

### 参考文献
```BibTeX
@article{li2024llava,
      title={LLaVA-OneVision: Easy Visual Task Transfer},
      author={Li, Bo and Zhang, Yuanhan and Guo, Dong and Zhang, Renrui and Li, Feng and Zhang, Hao and Zhang, Kaichen and Li, Yanwei and Liu, Ziwei and Li, Chunyuan},
      journal={arXiv preprint arXiv:2408.03326},
      year={2024}
}

@misc{gu2024infinitymmscalingmultimodalperformance,
      title={Infinity-MM: Scaling Multimodal Performance with Large-Scale and High-Quality Instruction Data},
      author={Shuhao Gu and Jialing Zhang and Siyuan Zhou and Kevin Yu and Zhaohu Xing and Liangdong Wang and Zhou Cao and Jintao Jia and Zhuoyi Zhang and Yixuan Wang and Zhenchong Hu and Bo-Wen Zhang and Jijie Li and Dong Liang and Yingli Zhao and Yulong Ao and Yaoqi Liu and Fangxiang Feng and Guang Liu},
      year={2024},
      eprint={2410.18558},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.18558},
}
```
