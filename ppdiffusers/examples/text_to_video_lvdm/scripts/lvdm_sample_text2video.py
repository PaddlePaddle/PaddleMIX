import numpy as np
import os
from PIL import Image
import paddle
from ppdiffusers import LVDMTextToVideoPipeline

# 加载模型和scheduler
pipe = LVDMTextToVideoPipeline.from_pretrained(
    'westfish/lvdm_text2video_orig_webvid_2m')

# 执行pipeline进行推理
seed = 2013
generator = paddle.Generator().manual_seed(seed)
samples = pipe(
    prompt="cutting in kitchen",
    num_frames=16,
    height=256,
    width=256,
    num_inference_steps=50,
    generator=generator,
    guidance_scale=15,
    eta=1,
    save_dir='.',
    save_name='ddim_lvdm_text_to_video_ucf',
    encoder_type='2d',
    scale_factor=0.18215,
    shift_factor=0, )
