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
# use paddle as backend to inference static model is not fast,
# this script is used to make sure the inference is correct.
# ==============================================================================
# text2img
python infer_dygraph_torch.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --controlnet_pretrained_model_name_or_path lllyasviel/sd-controlnet-canny --scheduler "ddim" --task_name text2img_control

# img2img
python infer_dygraph_torch.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --controlnet_pretrained_model_name_or_path lllyasviel/sd-controlnet-canny --scheduler "ddim" --task_name img2img_control

# inpaint
python infer_dygraph_torch.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --controlnet_pretrained_model_name_or_path lllyasviel/sd-controlnet-canny --scheduler "ddim" --task_name inpaint_legacy_control
