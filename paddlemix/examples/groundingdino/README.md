# Grounding DINO

## 1. 模型简介

Paddle implementation of [Grounding DINO](https://arxiv.org/abs/2303.05499), a stronger open-set object detector.


## 2. Demo

## 2.1 prepare
```bash
#Multi-scale deformable attention custom OP compilation
cd /paddlemix/models/groundingdino/csrc/
python setup_ms_deformable_attn_op.py install

```
## 2.2 dynamic inference
```bash
python run_predict.py \
--input_image image_you_want_to_detect.jpg \
--prompt "cat" \
```
