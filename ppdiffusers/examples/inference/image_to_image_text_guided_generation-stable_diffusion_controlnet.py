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

import cv2
import numpy as np
import paddle
from PIL import Image

from ppdiffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    UniPCMultistepScheduler,
)
from ppdiffusers.utils import load_image

# download an image
image = load_image(
    "https://paddlenlp.bj.bcebos.com/models/community/ppdiffusers/paddle-internal-testing/input_image_vermeer.png"
)
np_image = np.array(image)
# get canny image
np_image = cv2.Canny(np_image, 100, 200)
np_image = np_image[:, :, None]
np_image = np.concatenate([np_image, np_image, np_image], axis=2)
canny_image = Image.fromarray(np_image)
# load control net and stable diffusion v1-5
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", paddle_dtype=paddle.float16)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, paddle_dtype=paddle.float16
)
# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# generate image
generator = paddle.Generator().manual_seed(0)
image = pipe(
    "futuristic-looking woman",
    num_inference_steps=20,
    generator=generator,
    image=image,
    control_image=canny_image,
).images[0]
image.save("image_to_image_text_guided_generation-stable_diffusion_controlnet-result.png")
