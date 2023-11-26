# Grounding DINO

## 1. 模型简介

该模型是 [Grounding DINO](https://arxiv.org/abs/2303.05499) 的 paddle 实现。


## 2. Demo

## 2.1 依赖安装（可选）
```bash
#Multi-scale deformable attention custom OP compilation
cd paddlemix/models/groundingdino/csrc/
python setup_ms_deformable_attn_op.py install

```
## 2.2 动态图推理
```bash
python run_predict.py \
--input_image image_you_want_to_detect.jpg \
--prompt "cat"
```
