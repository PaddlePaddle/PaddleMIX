# AudioLDM2

## 1. 模型简介

该模型是 [AudioLDM2](https://arxiv.org/pdf/2308.05734) 的 paddle 实现。


## 2. Demo

### 2.1 依赖安装(如符合则跳过)

- 请确保已安装 ppdiffusers 
```bash
cd ppdiffusers
pip install -e .
```
- 其余依赖安装：
```bash
librosa
unidecode
phonemizer
espeak
```


### 2.2 动态图推理
```bash
python paddlemix/examples/audioldm2/run_predict.py \
--text "Musical constellations twinkling in the night sky, forming a cosmic melody." \
--model_name_or_path "haoheliu/audioldm2-full" \
--seed 1001 \
```
