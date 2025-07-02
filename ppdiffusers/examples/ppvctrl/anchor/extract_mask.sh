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

python anchor/extract_mask.py \
    --input_path=./examples/mask/case1/pixel_values.mp4 \
    --control_video_path=./examples/mask/case1/guide_values.mp4 \
    --mask_video_path=./examples/mask/case1/mask_values.mp4 \
    --reference_image_path=./examples/mask/case1/reference_image.jpg \
    --prompt="A dark gray Mini Cooper is parked on a city street"