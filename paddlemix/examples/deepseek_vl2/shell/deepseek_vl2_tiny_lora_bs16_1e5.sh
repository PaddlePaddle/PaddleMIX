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

set -x

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-16}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}

GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
tensor_parallel_degree=${tensor_parallel_degree:-1}
sharding_parallel_degree=$((GPUS / tensor_parallel_degree))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR='work_dirs/deepseekvl2_tiny_lora_bs16_1e5'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

TRAINING_MODEL_RESUME="None"
TRAINER_INSTANCES='127.0.0.1'
MASTER='127.0.0.1:8080'

meta_path="paddlemix/examples/deepseek_vl2/configs/LaTeX_OCR.json"

TRAINING_PYTHON="python -m paddle.distributed.launch --master ${MASTER} --nnodes 1 --nproc_per_node ${GPUS} --rank 0 --ips ${TRAINER_INSTANCES} --run_mode=collective"
${TRAINING_PYTHON} --log_dir ${OUTPUT_DIR}/paddle_distributed_logs \
  paddlemix/examples/deepseek_vl2/deepseek_vl2_finetune.py \
  --do_train \
  --model_name_or_path "deepseek-ai/deepseek-vl2-tiny" \
  --output_dir ${OUTPUT_DIR} \
  --logging_dir ${OUTPUT_DIR}/logs \
  --meta_path ${meta_path} \
  --overwrite_output_dir True \
  --dataloader_num_workers 0 \
  --bf16 True \
  --fp16 False \
  --fp16_opt_level "O1" \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --freeze_vit True \
  --image_resolution 384 \
  --max_grad_norm 1.0 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 1 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.05 \
  --optim "adamw" \
  --lr_scheduler_type "constant" \
  --logging_steps 1 \
  --report_to "visualdl" \
  --recompute True \
  --tensor_parallel_degree=${tensor_parallel_degree} \
  --sharding_parallel_degree=${sharding_parallel_degree} \
  --pipeline_parallel_degree=1 \
  --sep_parallel_degree=1 \
  --sharding="stage1" \
  --amp_master_grad=1 \
  --hybrid_parallel_topo_order="sharding_first" \
  --lora True \
  --lora_rank=8 \
  --lora_alpha=32 \
  --lora_dropout=0.05 \
  --lora_target_modules="language.model.layers.*.self_attn.q_proj.*,language.model.layers.*.self_attn.k_proj.*,language.model.layers.*.self_attn.v_proj.*,language.model.layers.*.self_attn.*o_proj.*,language.model.layers.*.mlp.experts.*.gate_proj.*,language.model.layers.*.mlp.experts.*.up_proj.*,language.model.layers.*.mlp.experts.*.down_proj.*,language.model.layers.*.mlp.gate_proj.*,language.model.layers.*.mlp.up_proj.*,language.model.layers.*.mlp.down_proj.*" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
