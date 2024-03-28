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

import paddle

from ppdiffusers import StableVideoDiffusionPipeline
from ppdiffusers.utils import load_image

pipeline = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", paddle_dtype=paddle.float16
)
pipeline.enable_xformers_memory_efficient_attention()

image = load_image("https://paddlenlp.bj.bcebos.com/models/community/hf-internal-testing/diffusers-images/rocket.png")
generator = paddle.Generator().manual_seed(1024)
frames = pipeline(
    image=image,
    num_inference_steps=25,
    width=1024,
    height=576,
    generator=generator,
    fps=7,
    decode_chunk_size=2,
).frames
# save gif
frames[0][0].save(
    "image_to_video_generation_stable_video_diffusion.gif",
    save_all=True,
    append_images=frames[0][1:],
    duration=1,
    loop=0,
)
