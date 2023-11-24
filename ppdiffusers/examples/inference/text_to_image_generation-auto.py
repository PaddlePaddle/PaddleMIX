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

from ppdiffusers import AutoPipelineForText2Image

pipe_t2i = AutoPipelineForText2Image.from_pretrained(
    "runwayml/stable-diffusion-v1-5", requires_safety_checker=False, paddle_dtype=paddle.float16
)
prompt = "photo a majestic sunrise in the mountains, best quality, 4k"
image = pipe_t2i(prompt).images[0]
image.save("text_to_image_generation-auto-result1.png")


from ppdiffusers import AutoPipelineForImage2Image

pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i)
image = pipe_i2i("sunrise in snowy mountains, best quality", image).images[0]
image.save("text_to_image_generation-auto-result2.png")
