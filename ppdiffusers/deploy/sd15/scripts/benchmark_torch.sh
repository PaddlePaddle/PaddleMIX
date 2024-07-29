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

# attention raw
python infer_dygraph_torch.py --scheduler "ddim" --task_name all --attention_type raw --use_fp16 True --inference_steps 50 --height 512 --width 512 --benchmark_steps 10

# attention sdp
python infer_dygraph_torch.py --scheduler "ddim" --task_name all --attention_type sdp --use_fp16 True --inference_steps 50 --height 512 --width 512 --benchmark_steps 10


# attention raw fp32
python infer_dygraph_torch.py --scheduler "ddim" --task_name all --attention_type raw --use_fp16 False --inference_steps 50 --height 512 --width 512 --benchmark_steps 10

# attention sdp fp32
python infer_dygraph_torch.py --scheduler "ddim" --task_name all --attention_type sdp --use_fp16 False --inference_steps 50 --height 512 --width 512 --benchmark_steps 10