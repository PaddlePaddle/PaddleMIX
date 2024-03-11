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

# can not run in paddle with v100 GPU
# import paddle
# from ppdiffusers import VideoToVideoModelscopePipeline
# from ppdiffusers.utils import export_to_video

# pipe = VideoToVideoModelscopePipeline.from_pretrained(
#     "Yang-Changhui/video-to-video-paddle")
# video_path = './image_to_video_generation_image_to_video.mp4'
# prompt = "A panda is surfing on the sea"
# video_frames = pipe(prompt=prompt,video_path=video_path).frames
# video_path = "video_to_video_generation_video_to_video.mp4"
# export_to_video(video_frames, video_path)
