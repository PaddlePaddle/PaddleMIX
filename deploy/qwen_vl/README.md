0. 转换权重

* 下载`qwen-vl/qwen-vl-7b`模型
* 使用脚本`PaddleMIX/change.py`转换权重的key

1. 导出vision模型

* paddlemix 脚本：PaddleMIX/export_qwen_vl_vision.sh

```python
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/PaddleNLP/:/path/to/PaddleMIX

python deploy/qwen_vl/export_image_encoder.py \
    --model_name_or_path "qwen-vl/qwen-vl-7b" \
```

2. 导出language模型

* paddlenlp 脚本：PaddleNLP/llm/export_qwen_vl_qwen.sh

```python
#!/bin/bash

export PYTHONPATH=xxx
export CUDA_VISIBLE_DEVICES=0

python export_model.py \
    --model_name_or_path qwen-vl/qwen-vl-7b-change \
    --output_path ./checkpoints/encode_text/ \
    --dtype float16 \
    --inference_model \
    --model_prefix qwen \
    --model_type qwen-img2txt
```

3. 运行

* paddlemix 脚本：PaddleMIX/run_qwen_vl.sh

```python
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/PaddleNLP/:/path/to/PaddleMIX

python deploy/qwen_vl/run_static_predict.py \
    --first_model_path "/path/to/checkpoints/encode_image_fp16/vision" \
    --second_model_path "/path/to/checkpoints/encode_text_fp16/qwen" \
    --qwen_vl_config_path "qwen-vl/qwen-vl-7b" \
```