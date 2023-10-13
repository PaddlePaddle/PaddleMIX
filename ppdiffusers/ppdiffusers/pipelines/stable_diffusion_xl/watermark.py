# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from ...utils import is_invisible_watermark_available

if is_invisible_watermark_available():
    from imwatermark import WatermarkEncoder

# Copied from https://github.com/Stability-AI/generative-models/blob/613af104c6b85184091d42d374fef420eddb356d/scripts/demo/streamlit_helpers.py#L66
WATERMARK_MESSAGE = 197828617679262
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]


class StableDiffusionXLWatermarker:
    def __init__(self):
        self.watermark = WATERMARK_BITS
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark("bits", self.watermark)

    def apply_watermark(self, images: paddle.Tensor):
        # can't encode images that are smaller than 256
        if images.shape[-1] < 256:
            return images
        images = (255 * (images / 2 + 0.5)).cpu().transpose(perm=[0, 2, 3, 1]).astype(dtype="float32").numpy()
        images = [self.encoder.encode(image, "dwtDct") for image in images]
        images = paddle.to_tensor(data=np.array(images)).transpose(perm=[0, 3, 1, 2])
        images = paddle.clip(x=2 * (images / 255 - 0.5), min=-1.0, max=1.0)
        return images
