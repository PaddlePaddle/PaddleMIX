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

# export http_proxy=agent.baidu.com:8188
# export https_proxy=agent.baidu.com:8188
# export no_proxy=bcebos.com
# python -m paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 --log_dir logs scripts/train_sd3.py --config config/grpo.py:geneval_sd3
python -m paddle.distributed.launch --gpus=0,1 --log_dir logs_ocr scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_4gpu
# CUDA_VISIBLE_DEVICES=6,7 python -m paddle.distributed.launch --log_dir logs scripts/train_sd3.py --config config/grpo.py:geneval_sd3