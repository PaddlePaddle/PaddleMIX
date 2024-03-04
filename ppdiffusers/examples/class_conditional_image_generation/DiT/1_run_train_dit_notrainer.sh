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

config_file=config/DiT_XL_patch2.json
results_dir=./output_notrainer/DiT_XL_patch2_notrainer

feature_path=./data/fastdit_imagenet256
global_batch_size=256
num_workers=8
max_steps=7000000
logging_steps=50
save_steps=5000

python -u -m paddle.distributed.launch --gpus "0,1,2,3,4,5,6,7" \
    train_image_generation_notrainer.py \
    --config_file ${config_file} \
    --feature_path ${feature_path} \
    --global_batch_size ${global_batch_size} \
    --num_workers ${num_workers} \
    --log_every ${logging_steps} \
    --ckpt_every ${save_steps} \
