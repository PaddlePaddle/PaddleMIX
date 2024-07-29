#!/bin/bash

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

export CUDA_VISIBLE_DEVICES=6 # 填写: GPU卡号
LOCAL_PATH=/root/lxl/project/paddlemx/20240723/PaddleMIX/ppdiffusers/deploy # 填写: deploy文件夹所在的本地路径

find $LOCAL_PATH -type d -name 'scripts' | while read script_dir; do
    parent_dir=$(dirname "$script_dir")

    script_path="$script_dir/benchmark_paddle_deploy_tensorrt.sh"
    
    if [ -f "$script_path" ]; then
        echo "Executing $script_path in directory $parent_dir..."
        cd "$parent_dir"
        bash "$script_path"
        cd - > /dev/null
    else
        echo "Script not found: $script_path"
    fi
done
