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

which python

CUDA_VISIBLE_DEVICES=6 python ../text_to_image_generation_tgate.py \
--prompt "A cat holding a sign that says hello world" \
--model 'flux' \
--gate_step 25 \
--sp_interval 2 \
--fi_interval 1 \
--warm_up 2 \
--saved_path './generated_tmp/flux/' \
--inference_step 50 \
--seed 42