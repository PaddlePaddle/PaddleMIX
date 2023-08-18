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
export FLAGS_conv_workspace_size_limit=4096
echo use_fp16_true-num_nodes_1-gpu_num_4-batch_size_4-accumulate_grad_batches_2-num_workers_8-use_recompute-use_xformers_auto

# ppdiffusers train
cd $ppdiffusers_path/examples/text_to_video_lvdm
python -u -m paddle.distributed.launch --gpus "4,5,6,7" train_lvdm_text2video.py \
    --do_train \
    --do_eval \
    --label_names pixel_values \
    --eval_steps 1000 \
    --vae_type 2d \
    --vae_name_or_path  None \
    --output_dir temp/checkpoints_text2video \
    --unet_config_file unet_configs/lvdm_text2video_orig_webvid_2m/unet/config.json \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 6e-5 \
    --max_steps 100 \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --image_logging_steps 1000 \
    --logging_steps 50 \
    --save_steps 5000 \
    --seed 23 \
    --dataloader_num_workers 8 \
    --weight_decay 0.01 \
    --max_grad_norm 0 \
    --overwrite_output_dir True \
    --pretrained_model_name_or_path westfish/lvdm_text2video_orig_webvid_2m \
    --recompute True \
    --fp16 --fp16_opt_level O1