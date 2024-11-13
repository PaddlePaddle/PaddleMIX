# LLaVA-Critic

## 1. 模型介绍

[LLaVA-Critic](https://llava-vl.github.io/blog/2024-10-03-llava-critic/) 旨在作为通用评估器来评估各种多模态任务的性能。该模型由字节跳动和马里兰大学的研究团队开发，是首个专门用于多任务评测的多模态大模型，通过提供详细评估和自我提升功能，帮助模型在多模态任务中取得更好的表现。

**本仓库支持的模型权重:**

| Model              |
|--------------------|
| lmms-lab/llava-critic-7b  |

注意：与huggingface权重同名，但权重为paddle框架的Tensor，使用`xxx.from_pretrained("lmms-lab/llava-critic-7b")`即可自动下载该权重文件夹到缓存目录。


## 2 环境准备

1）[安装PaddleNLP develop分支](https://github.com/PaddlePaddle/PaddleNLP?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

版本要求：paddlenlp>=3.0.0b2

2）[安装 PaddleMIX 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/b4f97ff859e1964c839fc5fab94f7ba63b1e5959?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

注意：Python版本最好为3.10及以上版本，Python版本最低要求是3.8.


## 3 快速开始

### 推理
```bash
python paddlemix/examples/llava_critic/run_predict.py
```

### 参考文献
```BibTeX
@article{xiong2024llavacritic,
  title={LLaVA-Critic: Learning to Evaluate Multimodal Models},
  author={Xiong, Tianyi and Wang, Xiyao and Guo, Dong and Ye, Qinghao and Fan, Haoqi and Gu, Quanquan and Huang, Heng and Li, Chunyuan},
  year={2024},
  eprint={2410.02712},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2410.02712},
}
```
