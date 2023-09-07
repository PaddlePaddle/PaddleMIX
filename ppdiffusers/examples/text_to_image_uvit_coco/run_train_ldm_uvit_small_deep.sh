# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

export FLAGS_embedding_deterministic=1
export FLAGS_cudnn_deterministic=1
export NVIDIA_TF32_OVERRIDE=0
export NCCL_ALGO=Tree
export FLAGS_set_to_1d=False


unet_config_file=config/uvit_t2i_small_deep.json
output_dir=output_dir/uvit_t2i_small_deep

per_device_train_batch_size=32
dataloader_num_workers=1

max_steps=1000000
save_steps=5000
warmup_steps=5000
image_logging_steps=5000
logging_steps=20

seed=1234
USE_AMP=False
fp16_opt_level="O1"
recompute=True

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m paddle.distributed.launch \
    train_txt2img_uvit_coco_trainer.py \
    --do_train \
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
    --image_logging_steps ${image_logging_steps}  \
    --logging_steps ${logging_steps} \
    --save_steps ${save_steps} \
    --seed ${seed}\
    --dataloader_num_workers ${dataloader_num_workers} \
    --max_grad_norm -1 \
    --unet_config_file ${unet_config_file} \
    --num_inference_steps 50 \
    --model_max_length 77 \
    --use_ema True \
    --overwrite_output_dir \
    --disable_tqdm True \
    --recompute ${recompute} \
    --fp16 ${USE_AMP} \
    --fp16_opt_level=${fp16_opt_level} \
    --enable_xformers_memory_efficient_attention True \

# visualdl --logdir output_dir/uvit_t2i_small_deep/runs/ --host 0.0.0.0 --port 8041