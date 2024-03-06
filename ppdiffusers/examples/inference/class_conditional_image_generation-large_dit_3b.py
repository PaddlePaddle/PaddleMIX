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

from ppdiffusers import DDIMScheduler, DiTPipeline

dtype = paddle.float32
with paddle.LazyGuard():
    pipe = DiTPipeline.from_pretrained("Alpha-VLLM/Large-DiT-3B-256", paddle_dtype=dtype)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
set_seed(0)

words = ["golden retriever"]  # class_ids [207]
class_ids = pipe.get_label_ids(words)

image = pipe(class_labels=class_ids, num_inference_steps=25).images[0]
image.save("class_conditional_image_generation-large_dit_3b-result.png")
