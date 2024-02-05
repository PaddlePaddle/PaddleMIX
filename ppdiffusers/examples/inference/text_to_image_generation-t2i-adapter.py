# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from PIL import Image

from ppdiffusers import StableDiffusionAdapterPipeline, T2IAdapter
from ppdiffusers.utils import load_image

image = load_image("https://hf-mirror.com/datasets/diffusers/docs-images/resolve/main/t2i-adapter/color_ref.png")
color_palette = image.resize((8, 8))
color_palette = color_palette.resize((512, 512), resample=Image.Resampling.NEAREST)
adapter = T2IAdapter.from_pretrained("TencentARC/t2iadapter_color_sd14v1", paddle_dtype=paddle.float16)
pipe = StableDiffusionAdapterPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    adapter=adapter,
    safety_checker=None,
    paddle_dtype=paddle.float16,
)
out_image = pipe(
    "At night, glowing cubes in front of the beach",
    image=color_palette,
).images[0]
out_image.save("text_to_image_generation-t2i-adapter-result-color_adapter.png")
