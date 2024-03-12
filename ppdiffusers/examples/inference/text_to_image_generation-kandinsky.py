# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ppdiffusers import KandinskyPipeline, KandinskyPriorPipeline

pipe_prior = KandinskyPriorPipeline.from_pretrained("kandinsky-community/Kandinsky-2-1-prior")

prompt = "red cat, 4k photo"
out = pipe_prior(prompt)
image_emb = out.image_embeds
negative_image_emb = out.negative_image_embeds

pipe = KandinskyPipeline.from_pretrained("kandinsky-community/kandinsky-2-1")

image = pipe(
    prompt,
    image_embeds=image_emb,
    negative_image_embeds=negative_image_emb,
    height=768,
    width=768,
    num_inference_steps=100,
).images

image[0].save("text_to_image_generation-kandinsky-result.png")
