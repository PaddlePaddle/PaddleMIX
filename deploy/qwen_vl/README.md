# Qwen-VL

## 1. 模型简介

该模型是 [Qwen-VL](https://arxiv.org/abs/2308.12966) 的 paddle 实现。

## 2. 安装依赖

* `paddlenlp_ops`依赖安装

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
pip install -e .
cd csrc
python setup_cuda.py install
```

* `fused_ln`需要安装[此目录](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/gpt-3/external_ops)下的自定义OP, `python setup.py install`

## 3. 示例

### 3.0 权重键转换

* `qwen-vl/qwen-vl-7b`中权重的`key`与`paddlenlp`模型加载需要的`key`存在差异，需要转换
* 下载`qwen-vl/qwen-vl-7b`模型，拷贝一份

```bash
cp -r /path/to/.paddlenlp/models/qwen-vl/qwen-vl-7b/ /path/to/.paddlenlp/models/qwen-vl/qwen-vl-7b-inference/
```

* 将`qwen-vl/qwen-vl-7b`的权重进行转换

```python
import paddle

state_dict = paddle.load("/path/to/.paddlenlp/models/qwen-vl/qwen-vl-7b-inference/model_state.pdparams")
new_state_dict = {}
for key in state_dict.keys():
    if key.startswith("transformer"):
        new_key = key.replace("transformer", "qwen")
        new_state_dict[new_key] = state_dict[key]
    else:
        new_state_dict[key] = state_dict[key]
paddle.save(new_state_dict, "/path/to/.paddlenlp/models/qwen-vl/qwen-vl-7b-inference/model_state.pdparams")
```

### 3.1 转出静态图推理所需的视觉模型

* 在`PaddleMIX`目录下，执行转换脚本，得到视觉模型部分静态图

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/PaddleNLP/:/path/to/PaddleMIX

python deploy/qwen_vl/export_image_encoder.py \
    --model_name_or_path "/path/to/.paddlenlp/models/qwen-vl/qwen-vl-7b-inference/" \
    --save_path "./checkpoints/encode_image/vision"
```

### 3.2 转出静态图推理所需的语言模型

* 在`PaddleNLP/llm`目录下，执行转换脚本，得到语言模型部分静态图

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/PaddleNLP/:/path/to/PaddleMIX

python export_model.py \
    --model_name_or_path "/path/to/.paddlenlp/models/qwen-vl/qwen-vl-7b-inference/" \
    --output_path ./checkpoints/encode_text/ \
    --dtype float16 \
    --inference_model \
    --model_prefix qwen \
    --model_type qwen-img2txt
```

### 3.3 静态图推理

* 在`PaddleMIX`目录下，运行执行脚本，进行静态图推理

```bash
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/PaddleNLP/:/path/to/PaddleMIX

python deploy/qwen_vl/run_static_predict.py \
    --first_model_path "/path/to/checkpoints/encode_image/vision" \
    --second_model_path "/path/to/checkpoints/encode_text/qwen" \
    --qwen_vl_config_path "/path/to/.paddlenlp/models/qwen-vl/qwen-vl-7b-inference/" \
```

### 3.4 A100 性能数据

* batch_size=1, dtype=float16
* torch 耗时 1012.11ms
* paddle 耗时 425.36ms
* 加速比 2.379X