# -*- coding: utf-8 -*-

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

import argparse
import os
import os.path as osp

import paddle
from PIL import Image

from paddlemix.models.emu3.modeling_emu3visionvq import Emu3VisionVQModel
from paddlemix.processors.image_processing_emu3 import Emu3VisionVQImageProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="BAAI/Emu3-VisionTokenizer")
parser.add_argument("--image_path", type=str, default="paddlemix/examples/emu3/demo")
parser.add_argument("--video_path", type=str, default="paddlemix/examples/emu3/demo")

args = parser.parse_args()

model = Emu3VisionVQModel.from_pretrained(args.model_path).eval()
processor = Emu3VisionVQImageProcessor.from_pretrained(args.model_path)

# TODO: you need to modify the path here

if not os.path.exists(args.video_path):
    video = []
else:
    video = os.listdir(args.video_path)
    if len(video) > 0:
        video.sort()
        video = [Image.open(osp.join(args.video_path, v)) for v in video]
        images = processor(video, return_tensors="pd")["pixel_values"]
        images = images.unsqueeze(0)

# image autoencode
image = [Image.open(args.image_path)]
image = processor(image, return_tensors="pd")["pixel_values"]
print(image.shape)
with paddle.no_grad():
    # encode
    codes = model.encode(image)
    # decode
    recon = model.decode(codes)

recon = recon.reshape([-1, *recon.shape[2:]])
recon_image = processor.postprocess(recon)["pixel_values"][0]
recon_image.save("recon_image.png")

# video autoencode while frames are divisible by temporal_downsample_factor
if len(video) > 1 and len(images) % model.config.temporal_downsample_factor == 0:
    images = images.reshape(
        [-1, model.config.temporal_downsample_factor] + images.shape[2:],
    )

    print(images.shape)
    with paddle.no_grad():
        # encode
        codes = model.encode(images)
        # decode
        recon = model.decode(codes)

    recon = recon.reshape([-1, *recon.shape[2:]])
    recon_images = processor.postprocess(recon)["pixel_values"]
    for idx, im in enumerate(recon_images):
        im.save(f"recon_video_{idx}.png")
