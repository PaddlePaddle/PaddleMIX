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
from ppdiffusers.pipelines import LDMTextToImageUViTPipeline

dtype = paddle.float32
pipe = LDMTextToImageUViTPipeline.from_pretrained("baofff/ldm-uvit_t2i-small-256-mscoco", paddle_dtype=dtype)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
set_seed(42)

prompt = "People are at a stop light on a snowy street."
image = pipe(prompt, guidance_scale=7.5, num_inference_steps=50).images[0]

image.save("result.png")
print(f"\nGPU memory usage: {paddle.device.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB")
