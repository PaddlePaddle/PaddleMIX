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

from ppdiffusers import WuerstchenDecoderPipeline, WuerstchenPriorPipeline

prior_pipe = WuerstchenPriorPipeline.from_pretrained("warp-ai/wuerstchen-prior", paddle_dtype=paddle.float16)
gen_pipe = WuerstchenDecoderPipeline.from_pretrained("warp-ai/wuerstchen", paddle_dtype=paddle.float16)

prompt = "an image of a shiba inu, donning a spacesuit and helmet"
prior_output = prior_pipe(prompt)
image = gen_pipe(prior_output.image_embeddings, prompt=prompt)[0][0]
image.save("text_to_image_generation_wuerstchen.png")
