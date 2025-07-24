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

import argparse
import os

os.environ["USE_PEFT_BACKEND"] = "True"
import paddle

from ppdiffusers import FluxPipeline

parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
    "--path_to_lora",
    type=str,
    required=True,
    help="Path to paddle_lora_weights.safetensors",
)
parser.add_argument(
    "--prompt",
    type=str,
    required=True,
    default="a beautiful girl",
)
parser.add_argument(
    "--guidance_scale",
    type=float,
    required=False,
    default=3.5,
)
parser.add_argument(
    "--height",
    type=int,
    required=False,
    default=1024,
)
parser.add_argument(
    "--width",
    type=int,
    required=False,
    default=1024,
)
parser.add_argument(
    "--lora_scale",
    type=float,
    required=False,
    default=0.25,
)
parser.add_argument(
    "--step",
    type=int,
    required=False,
    default=4,
)
parser.add_argument(
    "--seed",
    type=int,
    required=False,
    default=42,
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=False,
    default="./",
)
args = parser.parse_args()

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", map_location="cpu", paddle_dtype=paddle.bfloat16)
pipe.load_lora_weights(args.path_to_lora)

with paddle.no_grad():
    result_image = pipe(
        prompt=args.prompt,
        negative_prompt="",
        height=args.height,
        width=args.width,
        num_inference_steps=args.step,
        guidance_scale=args.guidance_scale,
        generator=paddle.Generator().manual_seed(args.seed),
        joint_attention_kwargs={"scale": args.lora_scale},
    ).images[0]
result_image.save(os.path.join(args.output_dir, "test_flux_lightning.png"))
