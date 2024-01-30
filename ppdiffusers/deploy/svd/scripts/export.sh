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

export FLAGS_use_cuda_managed_memory=true
# export CUDA_VISIBLE_DEVICES=1

export USE_PPXFORMERS=False
python export_model.py --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt --output_path static_model/stable-video-diffusion-img2vid-xt