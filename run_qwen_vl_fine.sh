#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/PaddleNLP/:/path/to/PaddleMIX

python paddlemix/examples/qwen_vl/run_predict.py \
    --model_name_or_path "qwen-vl/qwen-vl-7b" \
    --input_image "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg" \
    --prompt "Generate the caption in English with grounding:" \
    --dtype "float16"