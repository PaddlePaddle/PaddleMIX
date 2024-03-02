# CogAgent

## 1. 模型简介

该模型是 [CogAgent](https://arxiv.org/abs/2312.08914) 的 paddle 实现。对齐的是 huggingface 上的 `THUDM/cogagent-chat-hf`, tokenizer 采用的是 huggingface 上的 `lmsys/vicuna-7b-v1.5`


## 2. Demo

### 2.1 依赖安装

1） 安装PaddleNLP develop版本
```
pip install --pre --upgrade paddlenlp -f https://www.paddlepaddle.org.cn/whl/paddlenlp.html
```

2）安装 PaddleMix 环境依赖包

```
pip install -r requirements.txt
```

### 2.2 多轮对话

```bash
python paddlemix/examples/cogagent/chat_demo.py
```
