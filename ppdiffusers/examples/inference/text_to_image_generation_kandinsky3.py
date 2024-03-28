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

# can not run in paddle with v100 GPU
# from ppdiffusers import Kandinsky3Pipeline
# import paddle

# pipe = Kandinsky3Pipeline.from_pretrained("kandinsky-community/kandinsky-3", variant="fp16", paddle_dtype=paddle.float16)
# pipe.enable_model_cpu_offload()

# prompt = "A photograph of the inside of a subway train. There are raccoons sitting on the seats. One of them is reading a newspaper. The window shows the city in the background."

# generator = paddle.Generator().manual_seed(0)
# image = pipe(prompt, num_inference_steps=25, generator=generator).images[0]
# image.save("text_to_image_generation_kandinsky3")
