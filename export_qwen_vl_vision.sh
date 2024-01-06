#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/path/to/PaddleNLP/:/path/to/PaddleMIX

python deploy/qwen_vl/export_image_encoder.py \
    --model_name_or_path "qwen-vl/qwen-vl-7b" \
