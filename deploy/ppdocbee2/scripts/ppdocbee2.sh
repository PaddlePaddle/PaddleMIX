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
export FLAGS_cascade_attention_max_partition_size=512
export FLAGS_cascade_attention_deal_each_time=16
export USE_FASTER_TOP_P_SAMPLING=1

# 单卡高性能推理
python deploy/ppdocbee2/ppdocbee2_infer.py \
    --model_name_or_path PaddleMIX/PPDocBee2-3B \
    --media_type "image" \
    --question "识别这份表格的内容, 以markdown格式输出" \
    --image_file paddlemix/demo_images/medal_table.png \
    --min_length 0 \
    --max_length 2048 \
    --top_k 1 \
    --top_p 0.001 \
    --temperature 0.1 \
    --repetition_penalty 1.05 \
    --block_attn True \
    --append_attn True \
    --inference_model True \
    --llm_mode static \
    --dtype bfloat16 \
    --output_via_mq False \
    --benchmark True 


# # 多卡推理功能
# export CUDA_VISIBLE_DEVICES=0,1
# python -m paddle.distributed.launch --gpus "0,1" deploy/ppdocbee2/ppdocbee2_infer.py \
#     --model_name_or_path PaddleMIX/PPDocBee2-3B \
#     --media_type "image" \
#     --question "识别这份表格的内容, 以markdown格式输出" \
#     --image_file paddlemix/demo_images/medal_table.png \
#     --min_length 0 \
#     --max_length 2048 \
#     --top_k 1 \
#     --top_p 0.001 \
#     --temperature 0.1 \
#     --repetition_penalty 1.05 \
#     --block_attn True \
#     --append_attn True \
#     --inference_model True \
#     --llm_mode static \
#     --dtype bfloat16 \
#     --output_via_mq False \
#     --benchmark True 
