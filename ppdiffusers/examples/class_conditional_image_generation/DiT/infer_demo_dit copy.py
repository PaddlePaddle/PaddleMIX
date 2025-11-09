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

from ppdiffusers import DDIMScheduler, DiTPipeline, DPMSolverMultistepScheduler

dtype = paddle.float32
pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", paddle_dtype=dtype)
# import ipdb; ipdb.set_trace()
# use DDIMScheduler for inference
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.scheduler.config.algorithm_type = "dpmsolver"
pipe.scheduler.config.solver_order = 3
words = ["golden retriever"]  # class_ids [207]
class_ids = pipe.get_label_ids(words)
class_ids = [206,207]
import ipdb; ipdb.set_trace()
# import ipdb; ipdb.set_trace()
timesteps_list = [999, 899, 799, 699, 599, 499, 399, 299, 199, 99]
order_list = [1, 2, 3, 1, 1, 2, 2, 2, 2, 1]
# generate image
set_seed(42)
generator = paddle.Generator().manual_seed(0)
image = pipe(class_labels=class_ids, num_inference_steps=10, generator=generator, timesteps_list = timesteps_list, order_list = order_list).images[0]
import ipdb; ipdb.set_trace()
image.save("result_DiT_golden_retriever_dpm_10_2.png")
