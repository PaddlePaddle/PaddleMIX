# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

set -x

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-32}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}

GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
tensor_parallel_degree=${tensor_parallel_degree:-1}
sharding_parallel_degree=$((GPUS / tensor_parallel_degree))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='work_dirs/baseline_330k_3b_bs32_1e8'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

TRAINING_MODEL_RESUME="None"
TRAINER_INSTANCES='127.0.0.1'
MASTER='127.0.0.1:8080'

meta_path="paddlemix/examples/qwen2_5_vl/configs/baseline_6data_330k.json"

TRAINING_PYTHON="python -m paddle.distributed.launch --master ${MASTER} --nnodes 1 --nproc_per_node ${GPUS} --rank 0 --ips ${TRAINER_INSTANCES} --run_mode=collective"
${TRAINING_PYTHON} --log_dir ${OUTPUT_DIR}/paddle_distributed_logs \
  paddlemix/examples/qwen2_5_vl/qwen2_5_vl_finetune.py \
  --do_train \
  --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
  --output_dir ${OUTPUT_DIR} \
  --logging_dir ${OUTPUT_DIR}/logs \
  --meta_path ${meta_path} \
  --overwrite_output_dir True \
  --dataloader_num_workers 8 \
  --bf16 True \
  --fp16 False \
  --fp16_opt_level "O2" \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --freeze_vit True \
  --max_seq_length 8192 \
  --image_resolution 512 \
  --recompute False \
  --max_grad_norm 1.0 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 1 \
  --learning_rate 1e-8 \
  --warmup_ratio 0.1 \
  --warmup_steps 100 \
  --weight_decay 0.1 \
  --optim "adamw" \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --report_to "visualdl" \
  --tensor_parallel_degree=${tensor_parallel_degree} \
  --sharding_parallel_degree=${sharding_parallel_degree} \
  --pipeline_parallel_degree=1 \
  --sep_parallel_degree=1 \
  --sharding="stage1" \
  --amp_master_grad=1 \
  --hybrid_parallel_topo_order="sharding_first" \
  --lora True \
  --lora_rank=12 \
  --lora_alpha=256 \
  --lora_dropout=0.0 \
  --lora_target_modules="model.layers.*q_proj.*,model.layers.*k_proj.*,model.layers.*v_proj.*,model.layers.*gate_proj.*,model.layers.*up_proj.*,model.layers.*down_proj.*,model.layers.*o_proj.*" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
