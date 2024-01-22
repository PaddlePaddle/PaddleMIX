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

export FLAGS_conv_workspace_size_limit=4096

# 2.6.0的时候会有很多类型提升的warning，GLOG_minloglevel=2将会关闭这些warning
export GLOG_minloglevel=2
export OUTPUT_DIR="sdxl_lcm_lora_outputs"
export BATCH_SIZE=12
export MAX_ITER=10000

# 如果使用自定义数据
FILE_LIST=./processed_data/filelist/custom_dataset.filelist.list
# 如果使用laion400m_demo数据集，需要把下面的注释取消
# FILE_LIST=./data/filelist/train.filelist.list

# 如果使用sd15
# MODEL_NAME_OR_PATH="runwayml/stable-diffusion-v1-5"
# IS_SDXL=False
# RESOLUTION=512

# 如果使用sdxl
MODEL_NAME_OR_PATH="stabilityai/stable-diffusion-xl-base-1.0"
IS_SDXL=True
RESOLUTION=1024

python train_lcm.py \
    --do_train \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --max_steps ${MAX_ITER} \
    --lr_scheduler_type "constant" \
    --warmup_steps 0 \
    --image_logging_steps 100 \
    --logging_steps 10 \
    --resolution ${RESOLUTION} \
    --save_steps 2000 \
    --save_total_limit 20 \
    --seed 23 \
    --dataloader_num_workers 4 \
    --pretrained_model_name_or_path ${MODEL_NAME_OR_PATH} \
    --file_list ${FILE_LIST} \
    --model_max_length 77 \
    --max_grad_norm 1 \
    --disable_tqdm True \
    --overwrite_output_dir \
    --recompute True \
    --loss_type "huber" \
    --lora_rank 64 \
    --is_sdxl ${IS_SDXL} \
    --is_lora True \
    --bf16 True \
    --fp16_opt_level O2