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

export CUDA_VISIBLE_DEVICES=1
export FLAGS_enable_pir_api=0


export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/output/changwenbin/Research/TensorRT-10.3.0.26/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/output/changwenbin/Paddle/paddle/phi/kernels/fusion/cutlass/conv2d/build:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/root/paddlejob/workspace/env_run/output/changwenbin/Paddle/paddle/phi/kernels/fusion/cutlass/gemm_epilogue/build:$LD_LIBRARY_PATH
export TRITON_KERNEL_CACHE_DIR=/root/paddlejob/workspace/env_run/output/changwenbin/PaddleMIX/ppdiffusers/examples/vctrl/tmp/triton_kernel

# nsys profile -o vctrl_static_rope \
python infer_cogvideox_i2v_vctrl_cli.py \
  --pretrained_model_name_or_path "paddlemix/cogvideox-5b-i2v-vctrl" \
  --vctrl_path "vctrl_pose_5b_i2v.pdparams" \
  --vctrl_config "vctrl_configs/cogvideox_5b_i2v_vctrl_config.json" \
  --control_video_path "pose/guide_values_0.mp4" \
  --ref_image_path "pose/reference_image_0.jpg" \
  --output_dir "infer_outputs/pose2video" \
  --prompt "An animated character with blue hair and a playful expression dances energetically on a reflective stage, wearing a black dress with white lace and ruffles, and black stockings. She is surrounded by a futuristic setting with geometric shapes and neon lights in shades of blue, red, and orange. As she dances, her outfit changes slightly, including a black top with a heart-shaped neckline and a purple tail. Her dynamic poses and expressions convey joy and confidence. The backdrop's neon lights and geometric patterns enhance the vibrant and lively atmosphere of the performance." \
  --task "pose" \
  --width 480 \
  --height 720 \
  --max_frame 49 \
  --guidance_scale 3.5 \
  --benchmark 1 \
  --inference_optimize 0 \
  --num_inference_steps 25
