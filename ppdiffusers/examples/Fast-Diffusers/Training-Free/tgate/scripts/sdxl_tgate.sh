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

python ../text_to_image_generation_tgate.py \
--prompt 'Astronaut in a jungle, cold color palette, muted colors, detailed, 8k' \
--model 'sdxl' \
--gate_step 10 \
--sp_interval 5 \
--fi_interval 1 \
--warm_up 2 \
--saved_path './generated_tmp/sd_xl/' \
--inference_step 25 \
--seed 42