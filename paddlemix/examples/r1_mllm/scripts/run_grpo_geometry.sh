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

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-16}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
tensor_parallel_degree=${tensor_parallel_degree:-1}
sharding_parallel_degree=$((GPUS / tensor_parallel_degree))

NUM_GENERATIONS=${NUM_GENERATIONS:-8}

OUTPUT_DIR="work_dirs/Qwen2-VL-2B-GRPO-Geometry_${GPUS}"
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

DATASET_NAME="data/GEO/GEOQA_R1V_Train_8K"

TRAINING_MODEL_RESUME="None"
TRAINER_INSTANCES='127.0.0.1'
MASTER='127.0.0.1:8080'

TRAINING_PYTHON="python -m paddle.distributed.launch --master ${MASTER} --nnodes 1 --nproc_per_node ${GPUS} --rank 0 --ips ${TRAINER_INSTANCES} --run_mode=collective"
${TRAINING_PYTHON} --log_dir ${OUTPUT_DIR}/paddle_distributed_logs \
    paddlemix/examples/r1_mllm/train/grpo_r1-v.py \
    --do_train \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --output_dir ${OUTPUT_DIR} \
    --logging_dir ${OUTPUT_DIR}/logs \
    --dataset_name $DATASET_NAME \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --num_generations $NUM_GENERATIONS \
    --bf16 True \
    --fp16 False \
    --fp16_opt_level "O2" \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACC} \
    --freeze_vision True \
    --recompute True \
    --logging_steps 1 \
    --report_to "visualdl" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --max_pixels 401408 \
    --tensor_parallel_degree=${tensor_parallel_degree} \
    --sharding_parallel_degree=${sharding_parallel_degree} \
    --pipeline_parallel_degree=1 \
    --sep_parallel_degree=1 \
    --sharding "stage2" \
    --amp_master_grad=1 \
    --hybrid_parallel_topo_order="sharding_first" \
