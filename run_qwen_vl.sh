#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/PaddleNLP/:/path/to/PaddleMIX

python deploy/qwen_vl/run_static_predict.py \
    --first_model_path "/path/to/checkpoints/encode_image_fp16/vision" \
    --second_model_path "/path/to/checkpoints/encode_text_fp16/qwen" \
    --qwen_vl_config_path "qwen-vl/qwen-vl-7b" \