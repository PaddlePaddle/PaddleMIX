1. 导出vision模型

* paddlemix 脚本：PaddleMIX/export_qwen_vl_vision.sh

```python
#!/bin/bash
python deploy/qwen_vl/export_image_encoder.py \
    --qwen_vl_7b_path "qwen-vl/qwen-vl-7b"
```

2. 导出language模型

* paddlenlp 脚本：PaddleNLP/llm/export_qwen_vl_qwen.sh

```python
#!/bin/bash

export PYTHONPATH=xxx
export CUDA_VISIBLE_DEVICES=0

python export_model.py \
    --model_name_or_path qwen/qwen-7b \
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

python deploy/qwen_vl/run_static_predict.py \
--first_model_path "./checkpoints/encode_image/encode_image" \
--second_model_path "./checkpoints/encode_text/qwen" \
--qwen_tokenizer_path "qwen/qwen-7b" \
--qwen_vl_config_path "qwen-vl/qwen-vl-7b" \
```