# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from ppdiffusers import AutoencoderKLWan, WanPipeline
from ppdiffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from ppdiffusers.utils import export_to_video_2

# Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", paddle_dtype=paddle.float32)
pipe = WanPipeline.from_pretrained(model_id, vae=vae, paddle_dtype=paddle.bfloat16)

flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
scheduler = UniPCMultistepScheduler(
    prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift
)
pipe.scheduler = scheduler

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0,
).frames[0]

export_to_video_2(output, "output.mp4", fps=16)
