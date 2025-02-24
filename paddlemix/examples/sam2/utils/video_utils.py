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

import os

import cv2
from tqdm import tqdm


def create_video_from_images(image_folder, output_video_path, frame_rate=25):
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    image_files = [f for f in os.listdir(image_folder) if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()
    print(image_files)
    if not image_files:
        raise ValueError("No valid image files found in the specified folder.")
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = tuple(first_image.shape)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)
    video_writer.release()
    print(f"Video saved at {output_video_path}")


def create_video(annotated_frames, height, width, output_video_path, frame_rate=25):

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    for image in tqdm(annotated_frames):
        video_writer.write(image)

    video_writer.release()
    print(f"Video saved at {output_video_path}")
