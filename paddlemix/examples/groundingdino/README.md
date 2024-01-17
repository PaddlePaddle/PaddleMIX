# Grounding DINO

## 1. 模型简介

[Grounding DINO](https://arxiv.org/abs/2303.05499) 是一个开集（Open-Set）的目标检测模型，根据输入的文本提示，进行目标检测。
<p align="center">
  <img src="https://github.com/IDEA-Research/GroundingDINO/blob/main/.asset/hero_figure.png" align="middle" width = "600" />
</p>

注：图片引用自[Grounding DINO](https://arxiv.org/abs/2303.05499)

本仓库是Grounding DINO的Paddle实现，提供推理代码和部署代码。



## 2 自定义op安装（可选）
```bash
#Multi-scale deformable attention custom OP compilation
cd paddlemix/models/groundingdino/csrc/
python setup_ms_deformable_attn_op.py install

```
## 3 快速开始
```bash
python run_predict.py \
--input_image image_you_want_to_detect.jpg \
--prompt "cat"
```

## 4. 模型部署

模型部署可参考[deploy](../../../deploy/groundingdino/README.md)

## 参考文献
```bibtex
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```
