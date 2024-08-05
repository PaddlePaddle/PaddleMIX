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

export FLAGS_use_cuda_managed_memory=False

export USE_PPXFORMERS=False

# python infer.py --model_dir static_model/stable-video-diffusion-img2vid-xt --scheduler "ddim" --backend paddle --width 576 --height 576 --device gpu --task_name img2video
python infer.py --model_dir static_model/stable-video-diffusion-img2vid-xt --scheduler "ddim" --backend paddle --width 1024 --height 576 --device gpu --task_name img2video --tune True --inference_steps 1