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
from paddlenlp.trainer import set_seed

from ppdiffusers import UniDiffuserPipeline

model_id_or_path = "thu-ml/unidiffuser-v1"
pipe = UniDiffuserPipeline.from_pretrained(model_id_or_path, paddle_dtype=paddle.float16)
set_seed(42)

# Image variation can be performed with a image-to-text generation followed by a text-to-image generation:
from ppdiffusers.utils import load_image

# 1. Image-to-text generation
image_url = "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/unidiffuser/unidiffuser_example_image.jpg"
init_image = load_image(image_url).resize((512, 512))
sample = pipe(image=init_image, num_inference_steps=20, guidance_scale=8.0)
i2t_text = sample.text[0]
print(i2t_text)

# 2. Text-to-image generation
sample = pipe(prompt=i2t_text, num_inference_steps=20, guidance_scale=8.0)
final_image = sample.images[0]
final_image.save("image_variation-unidiffuser-result.png")
