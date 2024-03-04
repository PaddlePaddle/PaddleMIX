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

TRAINING_MODEL_RESUME="None"
TRAINER_INSTANCES='127.0.0.1'
MASTER='127.0.0.1:8080'
TRAINERS_NUM=1 # nnodes, machine num
TRAINING_GPUS_PER_NODE=4 # nproc_per_node
DP_DEGREE=2 # dp_parallel_degree
MP_DEGREE=1 # tensor_parallel_degree
SHARDING_DEGREE=2 # sharding_parallel_degree

uvit_config_file=config/uvit_t2i_small.json
output_dir=output_trainer/uvit_t2i_small_trainer

feature_path=./datasets/coco256_features
per_device_train_batch_size=8 #32
dataloader_num_workers=1 #8
max_steps=1000000
save_steps=10 #5000
warmup_steps=5000
logging_steps=1 #20
image_logging_steps=10000
seed=1234

USE_AMP=True
fp16_opt_level="O1" # "O2" bf16 bug now
enable_tensorboard=True
recompute=True
enable_xformers=True

TRAINING_PYTHON="python -m paddle.distributed.launch --master ${MASTER} --nnodes ${TRAINERS_NUM} --nproc_per_node ${TRAINING_GPUS_PER_NODE} --ips ${TRAINER_INSTANCES}"
${TRAINING_PYTHON} train_txt2img_mscoco_uvit_trainer.py \
    --do_train \
    --feature_path ${feature_path} \
    --output_dir ${output_dir} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps 1 \
    --learning_rate 0.0002 \
    --weight_decay 0.03 \
    --adam_beta1 0.9 \
    --adam_beta2 0.9 \
    --max_steps ${max_steps} \
    --lr_scheduler_type "constant_with_warmup" \
    --warmup_steps ${warmup_steps} \
    --image_logging_steps ${image_logging_steps} \
    --logging_steps ${logging_steps} \
    --save_steps ${save_steps} \
    --seed ${seed} \
    --dataloader_num_workers ${dataloader_num_workers} \
    --max_grad_norm -1 \
    --uvit_config_file ${uvit_config_file} \
    --num_inference_steps 50 \
    --model_max_length 77 \
    --use_ema True \
    --overwrite_output_dir True \
    --disable_tqdm True \
    --recompute ${recompute} \
    --fp16 ${USE_AMP} \
    --fp16_opt_level=${fp16_opt_level} \
    --enable_xformers_memory_efficient_attention ${enable_xformers} \
    --dp_degree ${DP_DEGREE} \
    --tensor_parallel_degree ${MP_DEGREE} \
    --sharding_parallel_degree ${SHARDING_DEGREE} \
    --pipeline_parallel_degree 1 \


# [2024-03-03 18:06:33,568] [    INFO] - loss: 1.08327329, learning_rate: 4e-08, global_step: 1, interval_runtime: 4.8704, interval_samples_per_second: 6.570357298475159, interval_steps_per_second: 0.2053236655773487, progress_or_epoch: 0.0008
# [2024-03-03 18:06:33,717] [    INFO] - loss: 1.07373798, learning_rate: 8e-08, global_step: 2, interval_runtime: 0.1491, interval_samples_per_second: 214.5962589716409, interval_steps_per_second: 6.706133092863778, progress_or_epoch: 0.0016
# [2024-03-03 18:06:33,849] [    INFO] - loss: 1.0802269, learning_rate: 1.2e-07, global_step: 3, interval_runtime: 0.1317, interval_samples_per_second: 242.9407661216607, interval_steps_per_second: 7.5918989413018965, progress_or_epoch: 0.0024
# [2024-03-03 18:06:33,989] [    INFO] - loss: 1.07584763, learning_rate: 1.6e-07, global_step: 4, interval_runtime: 0.1405, interval_samples_per_second: 227.69839207130667, interval_steps_per_second: 7.1155747522283335, progress_or_epoch: 0.0032
# [2024-03-03 18:06:34,172] [    INFO] - loss: 1.07868755, learning_rate: 2e-07, global_step: 5, interval_runtime: 0.1826, interval_samples_per_second: 175.25625949124944, interval_steps_per_second: 5.476758109101545, progress_or_epoch: 0.004
# [2024-03-03 18:06:34,323] [    INFO] - loss: 1.07524371, learning_rate: 2.4e-07, global_step: 6, interval_runtime: 0.1511, interval_samples_per_second: 211.8422283742704, interval_steps_per_second: 6.62006963669595, progress_or_epoch: 0.0047


# [2024-03-03 18:08:45,238] [    INFO] - loss: 1.08284163, learning_rate: 4e-08, global_step: 1, interval_runtime: 4.7601, interval_samples_per_second: 6.72256685991911, interval_steps_per_second: 0.21008021437247218, progress_or_epoch: 0.0008
# [2024-03-03 18:08:45,369] [    INFO] - loss: 1.07367396, learning_rate: 8e-08, global_step: 2, interval_runtime: 0.1317, interval_samples_per_second: 243.02874494586897, interval_steps_per_second: 7.594648279558405, progress_or_epoch: 0.0016
# [2024-03-03 18:08:45,501] [    INFO] - loss: 1.08071303, learning_rate: 1.2e-07, global_step: 3, interval_runtime: 0.1321, interval_samples_per_second: 242.14919255173857, interval_steps_per_second: 7.56716226724183, progress_or_epoch: 0.0024
# [2024-03-03 18:08:45,634] [    INFO] - loss: 1.07535195, learning_rate: 1.6e-07, global_step: 4, interval_runtime: 0.133, interval_samples_per_second: 240.6310226005515, interval_steps_per_second: 7.519719456267234, progress_or_epoch: 0.0032
# [2024-03-03 18:08:45,767] [    INFO] - loss: 1.07917476, learning_rate: 2e-07, global_step: 5, interval_runtime: 0.1329, interval_samples_per_second: 240.84951585233676, interval_steps_per_second: 7.526547370385524, progress_or_epoch: 0.004
