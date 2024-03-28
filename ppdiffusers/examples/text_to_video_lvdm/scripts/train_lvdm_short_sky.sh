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

ppdiffusers_path=PaddleMIX/ppdiffusers
export PYTHONPATH=$ppdiffusers_path:$PYTHONPATH
set -eux
# export CUDA_VISIBLE_DEVICES=1
# export NVIDIA_TF32_OVERRIDE=0

# ppdiffusers train
cd $ppdiffusers_path/examples/text_to_video_lvdm
python -u train_lvdm_short.py \
    --do_train \
    --do_eval \
    --label_names pixel_values \
    --eval_steps 5 \
    --vae_type 3d \
    --output_dir temp/checkpoints_short \
    --unet_config_file unet_configs/lvdm_short_sky_no_ema/unet/config.json \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 6e-5 \
    --max_steps 1000000000 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --image_logging_steps 10 \
    --logging_steps 1 \
    --save_steps 5000 \
    --seed 23 \
    --dataloader_num_workers 0 \
    --weight_decay 0.01 \
    --max_grad_norm 0 \
    --overwrite_output_dir False \
    --pretrained_model_name_or_path westfish/lvdm_short_sky_no_ema \
    --train_data_root your_data_path_to/sky_timelapse_lvdm \
    --eval_data_root your_data_path_to/sky_timelapse_lvdm