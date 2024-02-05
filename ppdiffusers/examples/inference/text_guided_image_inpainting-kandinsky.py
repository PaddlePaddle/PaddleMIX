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

import numpy as np
import paddle

from ppdiffusers import KandinskyInpaintPipeline, KandinskyPriorPipeline
from ppdiffusers.utils import load_image

pipe_prior = KandinskyPriorPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-1-prior", paddle_dtype=paddle.float16
)

prompt = "a hat"
image_emb, zero_image_emb = pipe_prior(prompt, return_dict=False)

pipe = KandinskyInpaintPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-1-inpaint", paddle_dtype=paddle.float16
)

init_image = load_image(
    "https://hf-mirror.com/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png"
)

mask = np.zeros((768, 768), dtype=np.float32)
mask[:250, 250:-250] = 1

out = pipe(
    prompt,
    image=init_image,
    mask_image=mask,
    image_embeds=image_emb,
    negative_image_embeds=zero_image_emb,
    height=768,
    width=768,
    num_inference_steps=50,
)

image = out.images[0]
image.save("text_guided_image_inpainting-kandinsky-result-cat_with_hat.png")
