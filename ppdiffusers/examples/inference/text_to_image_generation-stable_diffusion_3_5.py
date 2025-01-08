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

import paddle

from ppdiffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium", paddle_dtype=paddle.float16
)
generator = paddle.Generator().manual_seed(42)
prompt = "A cat holding a sign that says hello world"
image = pipe(prompt, 
             generator=generator,
             num_inference_steps=40,
             guidance_scale=4.5,
        ).images[0]
image.save("text_to_image_generation-stable_diffusion_3.5-result.png")
