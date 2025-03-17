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
# 1. export the model to static_model.
python export_model.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --ipadapter_pretrained_model_name_or_path h94/IP-Adapter --ipadapter_model_subfolder models --ipadapter_weight_name ip-adapter_sd15.safetensors --output_path static_model/stable-diffusion-v1-5-ipadapter

# 2. tune the shapes of the model for tensorrt 
python infer.py --model_dir static_model/stable-diffusion-v1-5-ipadapter/ --scheduler "ddim" --backend paddle --device gpu --task_name all --width 512 --height 512 --inference_steps 50 --tune True --use_fp16 False

# 3. convert the model to tensorrt
python infer.py --model_dir static_model/stable-diffusion-v1-5-ipadapter/ --scheduler "ddim" --backend paddle_tensorrt --device gpu --task_name all --width 512 --height 512 --inference_steps 50

# performance like this:
# --width 512 --height 512 --inference_steps 50 --benchmark_steps 10
# ==> Test text2img performance.
# Mean latency: 3.936669 s, p50 latency: 3.932006 s, p90 latency: 3.959874 s, p95 latency: 3.962563 s.

# ==> Test img2img performance.
# Mean latency: 3.225104 s, p50 latency: 3.224078 s, p90 latency: 3.226989 s, p95 latency: 3.231290 s.

# ==> Test inpaint_legacy performance.
# Mean latency: 3.231211 s, p50 latency: 3.230408 s, p90 latency: 3.240048 s, p95 latency: 3.240094 s.
