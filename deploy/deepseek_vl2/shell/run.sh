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

# export CUDA_VISIBLE_DEVICES=0
# export FLAGS_cascade_attention_max_partition_size=163840
# # export FLAGS_mla_use_tensorcore=1
# export FLAGS_mla_use_tensorcore=0
# # export PREFILL_USE_SAGE_ATTN=1

# python deploy/deepseek_vl2/deepseek_vl2_infer.py \
#     --model_name_or_path deepseek-ai/deepseek-vl2-small \
#     --question "Describe this image." \
#     --image_file paddlemix/demo_images/examples_image1.jpg \
#     --min_length 128 \
#     --max_length 128 \
#     --top_k 1 \
#     --top_p 0.001 \
#     --temperature 0.1 \
#     --repetition_penalty 1.05 \
#     --inference_model True \
#     --append_attn True \
#     --mode dynamic \
#     --dtype bfloat16 \
#     --quant_type "weight_only_int4" \
#     --mla_use_matrix_absorption \
#     --benchmark


export CUDA_VISIBLE_DEVICES=0
export FLAGS_mla_use_tensorcore=0
export FLAGS_cascade_attention_max_partition_size=128
export FLAGS_cascade_attention_deal_each_time=16

# nsys profile -o binbin_deepseek_appd 
python deploy/deepseek_vl2/deepseek_vl2_infer.py \
    --model_name_or_path deepseek-ai/deepseek-vl2-small \
    --question "Describe this image." \
    --image_file paddlemix/demo_images/examples_image1.jpg \
    --min_length 128 \
    --max_length 128 \
    --inference_model True \
    --append_attn True \
    --mode dynamic \
    --dtype bfloat16 \
    --top_k 1 \
    --top_p 0.001 \
    --temperature 0.1 \
    --repetition_penalty 1.05 \
    --benchmark \
    --quant_type "weight_only_int8" \
    --output_via_mq False