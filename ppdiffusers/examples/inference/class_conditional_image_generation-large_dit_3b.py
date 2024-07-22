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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import paddle
from paddlenlp.trainer import set_seed
import datetime

from ppdiffusers import DDIMScheduler, DiTPipeline

dtype = paddle.bfloat16

os.environ['callZKK']= "True"

with paddle.LazyGuard():
    pipe = DiTPipeline.from_pretrained("Alpha-VLLM/Large-DiT-3B-256", paddle_dtype=dtype)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
set_seed(0)

words = ["golden retriever"]  # class_ids [207]
class_ids = pipe.get_label_ids(words)


# for kkk in range(3):
#     image = pipe(class_labels=class_ids, num_inference_steps=25).images[0]


paddle.device.cuda.synchronize(0)
starttime = datetime.datetime.now()

for kk in range(1):
    image = pipe(class_labels=class_ids, num_inference_steps=25).images[0]

paddle.device.cuda.synchronize(0)
endtime = datetime.datetime.now()
duringtime = endtime-starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
msg = "total_time_cost: " + str(time_ms/5) + "ms\n\n"
print(msg)
with open("/tyk/PaddleMIX/ppdiffusers/examples/inference/kai/res/time_3B_722.txt", "a") as time_file:
    time_file.write(msg)


image.save("class_conditional_image_generation-large_dit_3b-result.png")
