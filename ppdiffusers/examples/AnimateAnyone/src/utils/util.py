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

import av
import cv2
import numpy as np
from PIL import Image


def read_frames(video_path):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames


def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps


def save_video_as_mp4(tensor, filename, fps):
    tensor = (tensor * 255).numpy().astype(np.uint8)
    video_data = tensor.squeeze().transpose(1, 2, 3, 0)

    frames, height, width, channels = video_data.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for i in range(frames):
        frame = cv2.cvtColor(video_data[i], cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
