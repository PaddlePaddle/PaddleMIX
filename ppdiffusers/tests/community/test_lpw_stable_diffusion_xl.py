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

from ppdiffusers import DiffusionPipeline
from ppdiffusers.utils.testing_utils import get_examples_pipeline

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    paddle_dtype=paddle.float16,
    use_safetensors=True,
    variant="fp16",
    custom_pipeline=get_examples_pipeline("lpw_stable_diffusion_xl"),
)

prompt = "photo of a cute (white) cat running on the grass" * 20
prompt2 = "chasing (birds:1.5)" * 20
prompt = f"{prompt},{prompt2}"
neg_prompt = "blur, low quality, carton, animate"

images = pipe(prompt=prompt, negative_prompt=neg_prompt, generator=paddle.Generator().manual_seed(452)).images[0]
images.save("out.png")
