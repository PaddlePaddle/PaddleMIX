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

#fp16  高性能推理
python deploy/qwen2_5_vl/qwen2_5_vl_infer.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --question "Describe this image." \
    --image_file paddlemix/demo_images/examples_image1.jpg \
    --min_length 128 \
    --max_length 128 \
    --top_k 1 \
    --top_p 0.001 \
    --temperature 0.1 \
    --repetition_penalty 1.05 \
    --dtype bfloat16 \
    --benchmark True 
    

# weight only int8 量化推理
# python deploy/qwen2_5_vl/qwen2_5_vl_infer.py \
#     --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
#     --question "Describe this image." \
#     --image_file paddlemix/demo_images/examples_image1.jpg \
#     --min_length 128 \
#     --max_length 128 \
#     --top_k 1 \
#     --top_p 0.001 \
#     --temperature 0.1 \
#     --repetition_penalty 1.05 \
#     --dtype bfloat16 \
#     --quant_type "weight_only_int8" \
#     --benchmark True 
