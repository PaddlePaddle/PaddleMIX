# AudioLDM2

## 1. 模型简介

该模型是 [AudioLDM2](https://arxiv.org/abs/2308.05734) 的 paddle 实现。


## 2. Demo

## 2.1 依赖安装
```bash
cd /paddlemix/models/audioldm2
pip install -r requirement.txt

```
## 2.2 动态图推理
```bash
python run_predict.py \
--text "Musical constellations twinkling in the night sky, forming a cosmic melody." \
--model_name_or_path "/home/aistudio/data/data252967" \
--seed 1001 \
```
