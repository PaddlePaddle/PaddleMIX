import os
os.environ["USE_PPXFORMERS"] = "False"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from ppdiffusers import StableDiffusionPipeline, UNet2DConditionModel
from ppdiffusers import StableDiffusionPipeline, UNet2DConditionModel, EulerAncestralDiscreteScheduler

# 加载公开发布的 unet 权重
unet_model_name_or_path = "runwayml/stable-diffusion-v1-5/unet"
unet = UNet2DConditionModel.from_pretrained(unet_model_name_or_path)
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None, unet=unet)
prompt = "a red photo of a village"  # or a little girl dances in the cherry blossom rain
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

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
print("The whoel end to end time : ", time_ms / repeat_times, "ms")


