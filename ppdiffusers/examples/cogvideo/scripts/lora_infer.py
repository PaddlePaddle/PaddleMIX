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

import os

os.environ["USE_PEFT_BACKEND"] = "True"
import argparse

import paddle

from ppdiffusers import CogVideoXPipeline
from ppdiffusers.utils import export_to_video_2

parser = argparse.ArgumentParser(description="Generate a video from a text prompt using Lora CogVideoX")
parser.add_argument(
    "--model_path", type=str, default="THUDM/CogVideoX-2b", help="The path of the pre-trained model to be used"
)
parser.add_argument("--prompt", type=str, help="Text prompt for the video generation")
parser.add_argument("--lora_path", type=str, help="Path to the LoRA weights file")
parser.add_argument("--lora_rank", type=int, default=64, help="The rank of the LoRA weights")
parser.add_argument("--lora_alpha", type=float, default=64, help="The alpha value of the LoRA weights")
parser.add_argument("--output_path", type=str, help="Output path for the generated video")

args = parser.parse_args()

pipe = CogVideoXPipeline.from_pretrained(args.model_path, paddle_dtype=paddle.float16)
pipe.load_lora_weights(args.lora_path, adapter_name="cogvideox-lora")

# Assuming lora_alpha=64 and rank=64 for training. If different, set accordingly
pipe.set_adapters(["cogvideox-lora"], [args.lora_alpha / args.lora_rank])


frames = pipe(args.prompt, guidance_scale=6, use_dynamic_cfg=True).frames[0]
export_to_video_2(frames, args.output_path, fps=8)
