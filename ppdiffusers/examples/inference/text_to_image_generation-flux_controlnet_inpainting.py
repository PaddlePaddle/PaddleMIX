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
from ppdiffusers.pipelines import FluxControlNetInpaintPipeline
from ppdiffusers.utils import load_image

controlnet = FluxControlNetModel.from_pretrained("InstantX/FLUX.1-dev-Controlnet-Canny", paddle_dtype=paddle.bfloat16)
pipe = FluxControlNetInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    paddle_dtype=paddle.bfloat16,
    low_cpu_mem_usage=True,
    map_location="cpu",
)

control_image = load_image("https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny-alpha/resolve/main/canny.jpg")
init_image = load_image(
    "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
)
mask_image = load_image(
    "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
)
prompt = "A girl holding a sign that says InstantX"
image = pipe(
    prompt,
    image=init_image,
    control_image=control_image,
    mask_image=mask_image,
    control_guidance_start=0.2,
    control_guidance_end=0.8,
    controlnet_conditioning_scale=0.7,
    strength=0.7,
    num_inference_steps=28,
    guidance_scale=3.5,
    generator=paddle.Generator().manual_seed(42),
).images[0]
image.save("text_to_image_generation-flux-dev-controlnet-inpaint-result.png")
