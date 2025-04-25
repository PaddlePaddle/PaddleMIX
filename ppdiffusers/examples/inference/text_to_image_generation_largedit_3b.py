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
from paddlenlp.trainer import set_seed

from ppdiffusers import DPMSolverMultistepScheduler
from ppdiffusers.pipelines import LDMTextToImageLargeDiTPipeline

# note that: will cost about 36G gpu memory
dtype = paddle.bfloat16
pipe = LDMTextToImageLargeDiTPipeline.from_pretrained("Alpha-VLLM/Large-DiT-T2I-3B-1024", paddle_dtype=dtype)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
set_seed(42)

prompt = "a dog is running on the grass"
image = pipe(prompt, height=1024, width=1024, guidance_scale=4, num_inference_steps=10).images[0]
image.save("text_to_image_generation-large_dit_t2i_3b_1024-result.png")
