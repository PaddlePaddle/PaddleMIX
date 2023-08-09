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
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_conv_workspace_size_limit=4096
export FLAG_USE_EMA=0
export FLAG_BENCHMARK=1
export FLAG_RECOMPUTE=1
export FLAG_XFORMERS=1
# use flash attention
export FLAG_XFORMERS_ATTENTION_OP=flash
# use fused linear
export FLAG_FUSED_LINEAR=1

export OUTPUT_DIR="bf16_o2_paddle"
export BATCH_SIZE=64
export MAX_ITER=200000

nohup python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" train_txt2img_laion400m_trainer.py \
    --do_train \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_steps ${MAX_ITER} \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 1000 \
    --logging_steps 10 \
    --resolution 256 \
    --save_steps 10000 \
    --save_total_limit 20 \
    --seed 23 \
    --dataloader_num_workers 8 \
    --pretrained_model_name_or_path ./CompVis-stable-diffusion-v1-4-paddle-init \
    --file_list ./data/filelist/train.filelist.list \
    --model_max_length 77 \
    --max_grad_norm -1 \
    --disable_tqdm True \
    --bf16 True \
    --fp16_opt_level O2 \
    --overwrite_output_dir > paddle_sd_bf16_o2.log 2>&1 &
