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
from ppdiffusers import FluxImg2ImgPipeline
from ppdiffusers.utils import load_image

pipe = FluxImg2ImgPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16, low_cpu_mem_usage=True, map_location="cpu",
)


url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
init_image = load_image(url).resize((768, 512))

prompt = "A fantasy landscape, trending on artstation"

images = pipe(
    height=512,
    width=768,
    prompt=prompt, image=init_image, num_inference_steps=50, strength=0.95, guidance_scale=0.0
).images[0]

images.save("text_to_image_generation-flux-dev-result_img2img.png")