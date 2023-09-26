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

import numpy as np
import paddle

from ppdiffusers import (
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetInpaintPipeline,
)
from ppdiffusers.utils import load_image

init_image = load_image(
    "https://paddlenlp.bj.bcebos.com/models/community/westfish/develop-upgrade0193/stable_diffusion_inpaint_boy.png"
)
init_image = init_image.resize((512, 512))
generator = paddle.Generator().manual_seed(1)
mask_image = load_image(
    "https://paddlenlp.bj.bcebos.com/models/community/westfish/develop-upgrade0193/stable_diffusion_inpaint_boy_mask.png"
)
mask_image = mask_image.resize((512, 512))


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = paddle.to_tensor(image)
    return image


control_image = make_inpaint_condition(init_image, mask_image)
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_inpaint", paddle_dtype=paddle.float16)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, paddle_dtype=paddle.float16
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# generate image
image = pipe(
    "a handsome man with ray-ban sunglasses",
    num_inference_steps=20,
    generator=generator,
    eta=1.0,
    image=init_image,
    mask_image=mask_image,
    control_image=control_image,
).images[0]
image.save("text_guided_image_inpainting-stable_diffusion_controlnet-result.png")
