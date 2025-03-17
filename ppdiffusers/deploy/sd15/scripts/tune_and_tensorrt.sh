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
python export_model.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --output_path static_model/stable-diffusion-v1-5

# 2. tune the shapes of the model for tensorrt 
python infer.py --model_dir static_model/stable-diffusion-v1-5/ --scheduler "ddim" --backend paddle --device gpu --task_name all --width 512 --height 512 --inference_steps 50 --tune True --use_fp16 False

# 3. convert the model to tensorrt
python infer.py --model_dir static_model/stable-diffusion-v1-5/ --scheduler "ddim" --backend paddle_tensorrt --device gpu --task_name all --width 512 --height 512 --inference_steps 50

# performance like this:
# --width 512 --height 512 --inference_steps 50 --benchmark_steps 10
# ==> Test text2img performance.
# Mean latency: 3.681746 s, p50 latency: 3.679480 s, p90 latency: 3.690280 s, p95 latency: 3.693445 s.

# ==> Test img2img performance.
# Mean latency: 3.005522 s, p50 latency: 3.005312 s, p90 latency: 3.009360 s, p95 latency: 3.015067 s.

# ==> Test inpaint_legacy performance.
# Mean latency: 3.021427 s, p50 latency: 3.020824 s, p90 latency: 3.027628 s, p95 latency: 3.028496 s.
