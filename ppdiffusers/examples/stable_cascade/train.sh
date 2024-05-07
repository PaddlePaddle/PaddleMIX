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

# TRAINING_MODEL_RESUME="None"
# TRAINER_INSTANCES='127.0.0.1'
# MASTER='127.0.0.1:8080'
# TRAINERS_NUM=1 # nnodes, machine num
# TRAINING_GPUS_PER_NODE=3 # nproc_per_node
# DP_DEGREE=3 # dp_parallel_degree
# MP_DEGREE=1 # tensor_parallel_degree
# SHARDING_DEGREE=1 # sharding_parallel_degree

export CUDA_VISIBLE_DEVICES=0
export GLOG_minloglevel=7
USE_AMP=False
fp16_opt_level="O2"
enable_tensorboard=True

TRAINING_PYTHON="python -u"
${TRAINING_PYTHON} train_stage_c_trainer.py \
    --do_train \
    --dataset_path /root/lxl/0_SC/Paddle-SC/dataset/haerbin \
    --output_dir ./train_output \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1.0e-4 \
    --resolution 512 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --max_steps 1000 \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 1000000 \
    --logging_steps 1 \
    --save_steps 5000 \
    --save_total_limit 50 \
    --seed 1 \
    --dataloader_num_workers 0 \
    --num_inference_steps 200 \
    --model_max_length 77 \
    --fp16 ${USE_AMP} \
    --fp16_opt_level=${fp16_opt_level}