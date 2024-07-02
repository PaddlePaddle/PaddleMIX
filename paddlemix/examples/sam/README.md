# Segment Anything

## 1. 模型简介

[Segment Anything](https://ai.facebook.com/research/publications/segment-anything/) 是 Meta AI Research, FAIR
发布的图像分割模型。根据输入提示（如点或框）生成高质量mask，可为图像中的所有对象进行分割。它已经在1100万张图像和11亿个掩模的数据集上进行了训练，并在各种分割任务上具有强大的零样本性能。

<p align="center">
  <img src="https://github.com/facebookresearch/segment-anything/blob/main/assets/model_diagram.png" align="middle" width = "600" />
</p>

注：图片引用自[Segment Anything](https://ai.facebook.com/research/publications/segment-anything/)

本仓库提供该模型的Paddle实现，并提供了推理代码和部署代码。


## 2. 快速开始

```bash
#box
python run_predict.py \
--input_image mage_you_want_to_seg.jpg \
--box_prompt  x y x y \
--input_type boxs

#points
python run_predict.py \
--input_image mage_you_want_to_seg.jpg \
--points_prompt points x y \
--input_type points
```

## 3. 模型部署

模型部署可参考[deploy](../../../deploy/sam/README.md)

## 参考文献
```
@article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```
