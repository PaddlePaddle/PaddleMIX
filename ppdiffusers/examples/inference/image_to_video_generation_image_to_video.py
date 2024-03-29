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

import paddle

from ppdiffusers import ImgToVideoSDPipeline
from ppdiffusers.utils import export_to_video, load_image

pipe = ImgToVideoSDPipeline.from_pretrained("Yang-Changhui/img-to-video-paddle", paddle_dtype=paddle.float32)
img = load_image(
    "https://paddlenlp.bj.bcebos.com/models/community/westfish/develop-upgrade0193/stable_diffusion_inpaint_boy.png"
)
video_frames = pipe(img).frames
video_path = export_to_video(video_frames, output_video_path="image_to_video_generation_image_to_video.mp4")
