# Qwen-VL

## 1. 模型简介

该模型是 [Qwen-VL](https://arxiv.org/pdf/2308.12966.pdf) 的 paddle 实现。


## 2. Demo

## 2.1 环境准备
- python >= 3.8
- paddlepaddle-gpu >= 2.5.1
- paddlenlp >= 2.6.1
- tiktoken
> 注：请确保安装了以上依赖，否则无法运行。同时，需要安装[此目录](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/gpt-3/external_ops)下的自定义OP, `python setup.py install`。如果安装后仍然找不到算子，需要额外设置PYTHONPATH

## 2.2 动态图推理
```bash
# qwen-vl
python paddlemix/examples/qwen_vl/run_predict.py \
--input_image "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg" \
--prompt "Generate the caption in English with grounding:" \
--dtype "float32"

# qwen-vl-chat demo
python3.8 paddlemix/examples/qwen_vl/chat_demo.py
```
