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

#!/bin/bash

export USE_PPXFORMERS=False
export FLAGS_set_to_1d=1

# Define the base path to the static model directory
base_model_dir="static_model"
model_dir=$(find $base_model_dir -mindepth 1 -maxdepth 1 -type d | head -n 1)

if [ -z "$model_dir" ]; then
    echo "No model directory found under $base_model_dir, starting to export static model..."

    if sh scripts/export.sh; then
        echo "Model exported successfully."
        model_dir=$(find $base_model_dir -mindepth 1 -maxdepth 1 -type d | head -n 1)
    else
        echo "Failed to export model."
        exit 1
    fi
else
    echo "Using model directory: $model_dir"
fi

# Tune static model shape info
echo "############### egnore this info - Begin ###############"
python infer.py --model_dir $model_dir --scheduler "ddim" --backend paddle_tensorrt --device gpu --task_name all --width 512 --height 512 --inference_steps 50 --tune True --use_fp16 False --benchmark_steps 3
echo "############### egnore this info - End #################"

# Inference with FP16
echo "Running inference with FP16..."
python infer.py --model_dir $model_dir --scheduler "ddim" --backend paddle_tensorrt --device gpu --task_name all --width 512 --height 512 --inference_steps 50 --tune False --use_fp16 True --benchmark_steps 10

# Inference with FP32
echo "Running inference with FP32..."
python infer.py --model_dir $model_dir --scheduler "ddim" --backend paddle_tensorrt --device gpu --task_name all --width 512 --height 512 --inference_steps 50 --tune False --use_fp16 False --benchmark_steps 10
