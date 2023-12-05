# Segment Anything

## 1. 模型简介

该模型是 [Segment Anything](https://ai.facebook.com/research/publications/segment-anything/) 的 paddle 实现, 可输入点或框进行分割。



## 2. 示例

## 2.1 动态图推理
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
