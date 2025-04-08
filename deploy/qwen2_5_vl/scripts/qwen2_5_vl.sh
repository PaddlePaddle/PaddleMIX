# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export CUDA_VISIBLE_DEVICES=0
#fp16  高性能推理
python deploy/qwen2_5_vl/qwen2_5_vl_infer.py \
    --model_name_or_path Qwen/Qwen2.5-VL-72B-Instruct \
    --media_type "image" \
    --question "Describe this image." \
    --image_file paddlemix/demo_images/examples_image1.jpg \
    --min_length 128 \
    --max_length 128 \
    --top_k 1 \
    --top_p 0.001 \
    --temperature 0.1 \
    --repetition_penalty 1.05 \
    --block_attn True \
    --inference_model True \
    --mode dynamic \
    --dtype bfloat16 \
    --output_via_mq False \
    --benchmark True 
    

# # weight only int8 量化推理
# python deploy/qwen2_5_vl/qwen2_5_vl_infer.py \
#     --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
#     --media_type "image" \
#     --question "Describe this image." \
#     --image_file paddlemix/demo_images/examples_image1.jpg \
#     --min_length 128 \
#     --max_length 128 \
#     --top_k 1 \
#     --top_p 0.001 \
#     --temperature 0.1 \
#     --repetition_penalty 1.05 \
#     --block_attn True \
#     --inference_model True \
#     --mode dynamic \
#     --dtype bfloat16 \
#     --quant_type "weight_only_int8" \
#     --output_via_mq False \
#     --benchmark True 


# # 多卡推理功能
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# python -m paddle.distributed.launch --gpus "0,1,2,3" deploy/qwen2_5_vl/qwen2_5_vl_infer.py \
#     --model_name_or_path Qwen/Qwen2.5-VL-72B-Instruct \
#     --media_type "image" \
#     --question "Describe this image." \
#     --image_file paddlemix/demo_images/examples_image1.jpg \
#     --min_length 128 \
#     --max_length 128 \
#     --top_k 1 \
#     --top_p 0.001 \
#     --temperature 0.1 \
#     --repetition_penalty 1.05 \
#     --block_attn True \
#     --inference_model True \
#     --mode dynamic \
#     --append_attn 1 \
#     --dtype bfloat16 \
#     --output_via_mq False \
#     --benchmark True 
