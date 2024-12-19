# mPLUG-Owl3

## 1. 模型介绍

**本仓库支持的模型权重:**

| Model              |
|--------------------|
| mPLUG/mPLUG-Owl3-7B-241101  |

注意：与huggingface权重同名，但权重为paddle框架的Tensor，使用`xxx.from_pretrained("mPLUG/mPLUG-Owl3-7B-241101")`即可自动下载该权重文件夹到缓存目录。


## 2 环境准备

1）[安装 PaddleMIX 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/develop?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

2）pip install pillow tqdm paddlenlp==3.0.0b2

注意：Python版本最好为3.10及以上版本。

## 3 快速开始

### 推理
```bash
# 图片理解
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/mPLUG_Owl3/run_inference.py \
```


### 参考文献
```BibTeX
@misc{ye2024mplugowl3longimagesequenceunderstanding,
      title={mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models},
      author={Jiabo Ye and Haiyang Xu and Haowei Liu and Anwen Hu and Ming Yan and Qi Qian and Ji Zhang and Fei Huang and Jingren Zhou},
      year={2024},
      eprint={2408.04840},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.04840},
}
```
