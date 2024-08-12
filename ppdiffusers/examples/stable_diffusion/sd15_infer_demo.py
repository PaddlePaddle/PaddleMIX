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

os.environ["USE_PPXFORMERS"] = "False"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import paddle

from ppdiffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

# 加载公开发布的 unet 权重
unet_model_name_or_path = "runwayml/stable-diffusion-v1-5/unet"
unet = UNet2DConditionModel.from_pretrained(unet_model_name_or_path)
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None, unet=unet)
prompt = "a photo of an astronaut riding a horse on mars"  # or a little girl dances in the cherry blossom rain
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

#pipe.text_encoder.forward = paddle.incubate.jit.inference(pipe.text_encoder.forward, with_trt=False)

pipe.unet.forward = paddle.incubate.jit.inference(
    pipe.unet.forward,
    cache_static_model=True,
    with_trt=True, trt_precision_mode="float16", trt_use_static=True
)

pipe.vae.decode = paddle.incubate.jit.inference(
    pipe.vae.decode,
    cache_static_model=True,
    with_trt=True, trt_precision_mode="float16", trt_use_static=True
)

for i in range(5):
    image = pipe(prompt, guidance_scale=7.5, width=512, height=512).images[0]
    image.save("astronaut_rides_horse.png")

import datetime
import time

warm_up_times = 5
repeat_times = 5

starttime = datetime.datetime.now()

for i in range(5):
    image = pipe(prompt, guidance_scale=7.5, width=512, height=512).images[0]

image.save("astronaut_rides_horse.png")
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("The whole end to end time : ", time_ms / repeat_times, "ms")
