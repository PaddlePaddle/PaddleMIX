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

set -uex


python infer_dygraph_torch.py --device_id 5 --use_fp16 True --attention_type raw --task text2img
python infer_dygraph_torch.py --device_id 5 --use_fp16 True --attention_type raw --task img2img
python infer_dygraph_torch.py --device_id 5 --use_fp16 True --attention_type raw --task text2img_with_refiner
python infer_dygraph_torch.py --device_id 5 --use_fp16 True --attention_type raw --task inpainting
python infer_dygraph_torch.py --device_id 5 --use_fp16 True --attention_type raw --task instruct_pix2pix

python infer_dygraph_torch.py --device_id 5 --use_fp16 True --attention_type sdp --task text2img
python infer_dygraph_torch.py --device_id 5 --use_fp16 True --attention_type sdp --task img2img
python infer_dygraph_torch.py --device_id 5 --use_fp16 True --attention_type sdp --task text2img_with_refiner
python infer_dygraph_torch.py --device_id 5 --use_fp16 True --attention_type sdp --task inpainting
python infer_dygraph_torch.py --device_id 5 --use_fp16 True --attention_type sdp --task instruct_pix2pix

python infer_dygraph_torch.py --device_id 5 --use_fp16 False --attention_type raw --task text2img
python infer_dygraph_torch.py --device_id 5 --use_fp16 False --attention_type raw --task img2img
python infer_dygraph_torch.py --device_id 5 --use_fp16 False --attention_type raw --task text2img_with_refiner
python infer_dygraph_torch.py --device_id 5 --use_fp16 False --attention_type raw --task inpainting
python infer_dygraph_torch.py --device_id 5 --use_fp16 False --attention_type raw --task instruct_pix2pix

python infer_dygraph_torch.py --device_id 5 --use_fp16 False --attention_type sdp --task text2img
python infer_dygraph_torch.py --device_id 5 --use_fp16 False --attention_type sdp --task img2img
python infer_dygraph_torch.py --device_id 5 --use_fp16 False --attention_type sdp --task text2img_with_refiner
python infer_dygraph_torch.py --device_id 5 --use_fp16 False --attention_type sdp --task inpainting
python infer_dygraph_torch.py --device_id 5 --use_fp16 False --attention_type sdp --task instruct_pix2pix
