import numpy as np
import os
from PIL import Image
import paddle
from ppdiffusers import LVDMUncondPipeline

# 加载模型和scheduler
pipe = LVDMUncondPipeline.from_pretrained(
    'westfish/lvdm_short_sky_epoch2239_step150079')

# 执行pipeline进行推理
seed = 1000
generator = paddle.Generator().manual_seed(seed)
samples = pipe(
    batch_size=1,
    num_frames=4,
    num_inference_steps=50,
    generator=generator,
    eta=1,
    save_dir='.',
    save_name='ddim_lvdm_short_sky_epoch2239_step150079',
    scale_factor=0.33422927,
    shift_factor=1.4606637, )
