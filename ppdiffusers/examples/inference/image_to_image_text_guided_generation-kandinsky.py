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

# import paddle

# from ppdiffusers import KandinskyImg2ImgPipeline, KandinskyPriorPipeline
# from ppdiffusers.utils import load_image

# pipe_prior = KandinskyPriorPipeline.from_pretrained(
#     "kandinsky-community/kandinsky-2-1-prior", paddle_dtype=paddle.float16
# )
# prompt = "A red cartoon frog, 4k"
# image_emb, zero_image_emb = pipe_prior(prompt, return_dict=False)
# pipe = KandinskyImg2ImgPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", paddle_dtype=paddle.float16)
# init_image = load_image(
#     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main" "/kandinsky/frog.png"
# )
# image = pipe(
#     prompt,
#     image=init_image,
#     image_embeds=image_emb,
#     negative_image_embeds=zero_image_emb,
#     height=768,
#     width=768,
#     num_inference_steps=100,
#     strength=0.2,
# ).images
# image[0].save("image_to_image_text_guided_generation-kandinsky-result-red_frog.png")

# todo
