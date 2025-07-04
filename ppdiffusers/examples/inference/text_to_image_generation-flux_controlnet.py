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
from ppdiffusers.pipelines import FluxControlNetPipeline
from ppdiffusers.utils import load_image

controlnet = FluxControlNetModel.from_pretrained("InstantX/FLUX.1-dev-Controlnet-Canny", paddle_dtype=paddle.bfloat16)
pipe = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    paddle_dtype=paddle.bfloat16,
    low_cpu_mem_usage=True,
    map_location="cpu",
)

control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")
prompt = "A girl in city, 25 years old, cool, futuristic"
image = pipe(
    prompt,
    control_image=control_image,
    controlnet_conditioning_scale=0.5,
    width=control_image.size[0],
    height=control_image.size[1],
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=paddle.Generator().manual_seed(42),
).images[0]
image.save("text_to_image_generation-flux-dev-controlnet-result.png")
