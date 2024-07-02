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

dtype: str = "float32"  # float32, float16
clip_image_model_name: str = "openai/clip-vit-large-patch14"
clip_text_model_name: str = "laion/pp_CLIP-ViT-bigG-14-laion2B-39B-b160k"

# 以下内容请修改为你的模型权重路径
WEIGHTS_PATH = "./stable_cascade_weights"

effnet_checkpoint_path: str = f"{WEIGHTS_PATH}/effnet_encoder.pdparams"
stage_a_checkpoint_path: str = f"{WEIGHTS_PATH}/stage_a.pdparams"
model_b_version: str = "700M"  # 700M, 3B可选，700M对应stage_b_lite版
stage_b_checkpoint_path: str = f"{WEIGHTS_PATH}/stage_b_lite.pdparams"
model_c_version: str = "1B"  # 1B, 3.6B可选，1B对应stage_c_lite版
stage_c_checkpoint_path: str = f"{WEIGHTS_PATH}/stage_c_lite.pdparams"
effnet_checkpoint_path: str = f"{WEIGHTS_PATH}/effnet_encoder.pdparams"
previewer_checkpoint_path: str = f"{WEIGHTS_PATH}/previewer.pdparams"
