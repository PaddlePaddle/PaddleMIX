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
from PIL import Image

from ppdiffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from ppdiffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", paddle_dtype=paddle.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
prompt = "spiderman running in the desert"
video_frames = pipe(prompt, num_inference_steps=40, height=320, width=576, num_frames=24).frames
# safe low-res video
video_path = export_to_video(
    video_frames, output_video_path="./text_to_video_generation-synth_img2img-result-video_1024_spiderman_lowres.mp4"
)
# and load the image-to-image model
pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_XL", paddle_dtype=paddle.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# The VAE consumes A LOT of memory, let's make sure we run it in sliced mode
pipe.vae.enable_slicing()
# now let's upscale it
video = [Image.fromarray(frame).resize((1024, 576)) for frame in video_frames]
# and denoise it
video_frames = pipe(prompt, video=video, strength=0.6).frames
video_path = export_to_video(
    video_frames, fps=7, output_video_path="text_to_video_generation-synth_img2img-result-video_1024_spiderman.mp4"
)
