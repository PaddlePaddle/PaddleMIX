# 

## 1. 模型简介

Paddle implementation of [Segment Anything](https://ai.facebook.com/research/publications/segment-anything/), produces high quality object masks from input prompts such as points or boxes.


## 2. Demo

## 2.1 dynamic inference
```bash
#box
python run_predict.py \
--input_image mage_you_want_to_seg.jpg \
--box_prompt  x y x y \
--input_type boxs

#points
python run_predict.py \
--input_image mage_you_want_to_seg.jpg \
--points_prompt points x y
--input_type points
```


