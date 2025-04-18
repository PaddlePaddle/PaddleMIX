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
BATCH_SIZE=${BATCH_SIZE:-512}

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
tensor_parallel_degree=${tensor_parallel_degree:-1}
sharding_parallel_degree=$((GPUS / tensor_parallel_degree))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=paddle

EXAMPLE_DIR="paddlemix/examples/internvl2"
OUTPUT_DIR='work_dirs/pretrain_internvl2_1b_qwen2_0_5b_dynamic_res_bs512'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

python -m paddle.distributed.launch \
  --nnodes=1 \
  --rank=0 \
  --master=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  ${EXAMPLE_DIR}/internvl_chat_pretrain.py \
  --vision_path "OpenGVLab/InternViT-300M-448px" \
  --llm_path "Qwen/Qwen2-0.5B-Instruct" \
  --conv_style "Hermes-2" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "${EXAMPLE_DIR}/shell/data/internvl_pretrain.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --fp16 False \
  --fp16_opt_level "O1" \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 1 \
  --learning_rate 2e-4 \
  --weight_decay 0.01 \
  --warmup_steps 100 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --recompute True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version "v2" \
  --report_to "visualdl" \
  --tensor_parallel_degree=${tensor_parallel_degree} \
  --sharding_parallel_degree=${sharding_parallel_degree} \
  --pipeline_parallel_degree=1 \
  --sep_parallel_degree=1 \
  --sharding="stage1" \
  --amp_master_grad=1 \
  --hybrid_parallel_topo_order="sharding_first" \
