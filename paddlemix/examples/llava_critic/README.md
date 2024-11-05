# LLaVA-Critic

## 1. 模型介绍

[LLaVA-Critic](https://llava-vl.github.io/blog/2024-10-03-llava-critic/)

## 2 环境准备
- **python >= 3.10**
- <span style="color:red;">**paddlenlp >= 3.0**</span>
```
cd PaddleMIX/paddlemix/examples/llava_critic
pip install -r requirement.txt

or

pip install paddlenlp==3.0.0b0
```

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
