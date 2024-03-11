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

import paddle

from ppdiffusers import DiffusionPipeline

# load both base & refiner
base = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    paddle_dtype=paddle.float16,
)
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    paddle_dtype=paddle.float16,
    variant="fp16",
)

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8

prompt = "A majestic lion jumping from a big stone at night"
prompt = "a photo of an astronaut riding a horse on mars"
generator = paddle.Generator().manual_seed(42)

# run both experts
image = base(
    prompt=prompt,
    output_type="latent",
    generator=generator,
).images

image = refiner(
    prompt=prompt,
    image=image,
    generator=generator,
).images[0]
image.save("text_to_image_generation-sdxl-base-with-refiner-result.png")
