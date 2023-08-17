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
## 2.2 Export model for static inference
```bash
#export grounding dino model
python export.py


#inference
 python predict.py  \
 --text_encoder_type GroundingDino/groundingdino-swint-ogc
 --model_path output_groundingdino \
 --input_image image_you_want_to_detect.jpg \
 -output_dir "dir you want to save the output" \
 -prompt "Detect Cat"

```
