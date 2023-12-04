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

export CUDA_VISIBLE_DEVICES=1
LOCAL_PATH=/root/lxl/DEVELOP_PR # 填写PaddleMIX文件夹所在的本地路径
cd $LOCAL_PATH/PaddleMIX/ppdiffusers/deploy/sdxl

echo "### 1. export model"
export USE_PPXFORMERS=False
python export_model.py --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 --output_path static_model/stable-diffusion-xl-base-1.0 

echo "### 2. inference"
rm -rf infer_op_raw_fp16
python infer.py --model_dir static_model/stable-diffusion-xl-base-1.0 --scheduler "preconfig-euler-ancestral" --backend paddle --device gpu --task_name all

echo "### 3. test diff"
echo "### 3.1 test_image_diff text2img"
python test_image_diff.py --source_image ./infer_op_raw_fp16/text2img.png  --target_image https://paddlenlp.bj.bcebos.com/models/community/baicai/sdxl_infer_op_raw_fp16/text2img.png

echo "### 3.2 test_image_diff img2img"
python test_image_diff.py --source_image ./infer_op_raw_fp16/img2img.png --target_image https://paddlenlp.bj.bcebos.com/models/community/baicai/sdxl_infer_op_raw_fp16/img2img.png

echo "### 3.3 test_image_diff inpaint"
python test_image_diff.py --source_image ./infer_op_raw_fp16/inpaint.png --target_image https://paddlenlp.bj.bcebos.com/models/community/baicai/sdxl_infer_op_raw_fp16/inpaint.png
