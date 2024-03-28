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

export CUDA_VISIBLE_DEVICES=2 # 填写: GPU卡号
LOCAL_PATH=/root/lxl/ # 填写: PaddleMIX文件夹所在的本地路径
cd $LOCAL_PATH/PaddleMIX/ppdiffusers/deploy/controlnet

export USE_PPXFORMERS=False
export FLAGS_set_to_1d=1
# 1. export the model to static_model.
python export_model.py --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 --controlnet_pretrained_model_name_or_path  lllyasviel/sd-controlnet-canny --output_path static_model/stable-diffusion-v1-5-canny

# 2. tune the shapes of the model for tensorrt 
python infer.py --model_dir static_model/stable-diffusion-v1-5-canny/ --scheduler "euler" --backend paddle --device gpu --task_name all --width 512 --height 512 --inference_steps 30 --tune True --use_fp16 False

# 3. convert the model to tensorrt
python infer.py --model_dir static_model/stable-diffusion-v1-5-canny/ --scheduler "euler" --backend paddle_tensorrt --device gpu --task_name all --width 512 --height 512 --inference_steps 50