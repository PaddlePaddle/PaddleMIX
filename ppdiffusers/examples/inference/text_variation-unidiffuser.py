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

# Text variation can be performed with a text-to-image generation followed by a image-to-text generation:
# 1. Text-to-image generation
prompt = "an elephant under the sea"
sample = pipe(prompt=prompt, num_inference_steps=20, guidance_scale=8.0)
t2i_image = sample.images[0]

# 2. Image-to-text generation
sample = pipe(image=t2i_image, num_inference_steps=20, guidance_scale=8.0)
final_prompt = sample.text[0]
with open("text_variation-unidiffuser-result.txt", "w") as f:
    print("{}\n".format(final_prompt), file=f)
