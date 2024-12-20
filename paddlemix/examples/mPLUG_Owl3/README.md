# mPLUG-Owl3

## 1. 模型介绍

mPLUG-Owl3 是由阿里巴巴推出的一个通用的多模态大语言模型，旨在有效地处理长图像序列。该模型通过在语言模型中引入创新的超注意力块，实现了高效的视频和图像理解。超注意力块并行地执行交叉注意力和自注意力，并根据文本语义自适应地选择和提取视觉特征。这使得 mPLUG-Owl3 能够在保持高执行效率的同时，处理超长视觉序列输入。


**本仓库支持的模型权重:**

| Model              |
|--------------------|
| mPLUG/mPLUG-Owl3-7B-241101  |

注意：与huggingface权重同名，但权重为paddle框架的Tensor，使用`xxx.from_pretrained("mPLUG/mPLUG-Owl3-7B-241101")`即可自动下载该权重文件夹到缓存目录。


## 2 环境准备

- **python >= 3.10**
- **paddlepaddle-gpu 要求3.0.0b2版本或develop版本**
```
# 安装示例
python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

- **paddlenlp == 3.0.0b3**
```
# 安装示例
python -m pip install paddlenlp==3.0.0b3
```

## 3 快速开始

### 推理
```bash
# 图片理解
CUDA_VISIBLE_DEVICES=0 python paddlemix/examples/mPLUG_Owl3/run_inference.py --dtype "bfloat16"
```

注意：mPLUG-Owl3-7B 模型不支持在V100上推理，请使用A100进行推理，推理显存约需20G。

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
