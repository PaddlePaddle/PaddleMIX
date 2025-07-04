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

from ppdiffusers import FluxControlNetModel
from ppdiffusers.pipelines import FluxControlNetImg2ImgPipeline
from ppdiffusers.utils import load_image

controlnet = FluxControlNetModel.from_pretrained("InstantX/FLUX.1-dev-Controlnet-Canny", paddle_dtype=paddle.bfloat16)
pipe = FluxControlNetImg2ImgPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    paddle_dtype=paddle.bfloat16,
    low_cpu_mem_usage=True,
    map_location="cpu",
)

control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")
init_image = load_image(
    "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
)
prompt = "A girl in city, 25 years old, cool, futuristic"
image = pipe(
    prompt,
    image=init_image,
    control_image=control_image,
    control_guidance_start=0.2,
    control_guidance_end=0.8,
    controlnet_conditioning_scale=1.0,
    strength=0.7,
    num_inference_steps=2,
    guidance_scale=3.5,
    generator=paddle.Generator().manual_seed(42),
).images[0]
image.save("text_to_image_generation-flux-dev-controlnet-img2img-result.png")
