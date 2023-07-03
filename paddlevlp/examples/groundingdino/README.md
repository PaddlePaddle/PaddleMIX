# Grounding DINO

## 1. 模型简介

Paddle implementation of [Grounding DINO](https://arxiv.org/abs/2303.05499), a stronger open-set object detector.


## 2. Demo

## 2.1 prepare
```bash
#Multi-scale deformable attention custom OP compilation
cd /paddlevlp/models/groundingdino/csrc/
python setup_ms_deformable_attn_op.py install

```
## 2.2 dynamic inference
```bash
python3.8 run_predict.py -dt groundingdino-swint-ogc 
-i image_you_want_to_detect.jpg \
-o "dir you want to save the output" \
-t "Detect Cat"
```


