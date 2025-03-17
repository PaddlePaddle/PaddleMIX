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

# ==============================================================================
# This is the script to tune and tensorrt.
# with this method, you can get the fastest inference speed.
# ==============================================================================
export USE_PPXFORMERS=False
export FLAGS_set_to_1d=1
export FLAGS_use_cuda_managed_memory=False
export CUDA_VISIBLE_DEVICES=6
# 1. export the model to static_model.
python export_model.py --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt --output_path static_model/stable-video-diffusion-img2vid-xt

# 2. tune the shapes of the model for tensorrt 
python infer.py --model_dir static_model/stable-video-diffusion-img2vid-xt --backend paddle --width 256 --height 256 --device gpu --task_name img2video --use_fp16 False --inference_steps 1

# 3. convert the model to tensorrt
python infer.py --model_dir static_model/stable-video-diffusion-img2vid-xt --backend paddle_tensorrt --width 256 --height 256 --device gpu --task_name img2video --inference_steps 25

# performance like this:
# --width 512 --height 512 --inference_steps 50 --benchmark_steps 10
# ==> Test text2img performance.
# Mean latency: 5.543108 s, p50 latency: 5.544122 s, p90 latency: 5.553590 s, p95 latency: 5.555381 s.

# ==> Test img2img performance.
# Mean latency: 4.588691 s, p50 latency: 4.588102 s, p90 latency: 4.600521 s, p95 latency: 4.601433 s.

# ==> Test inpaint_legacy performance.
# Mean latency: 4.625243 s, p50 latency: 4.626935 s, p90 latency: 4.629911 s, p95 latency: 4.633960 s.