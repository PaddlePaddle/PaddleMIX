#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python deploy/qwen_vl/run_static_predict.py \
--first_model_path "./checkpoints/encode_image/encode_image" \
--second_model_path "./checkpoints/encode_text/qwen" \
--qwen_tokenizer_path "qwen/qwen-7b" \
--qwen_vl_config_path "qwen-vl/qwen-vl-7b" \