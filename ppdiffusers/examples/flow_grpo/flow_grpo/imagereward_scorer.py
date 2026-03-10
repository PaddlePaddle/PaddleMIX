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

import ImageReward as RM
import paddle
from PIL import Image


class ImageRewardScorer(paddle.nn.Layer):
    def __init__(self, device="cuda", dtype=paddle.float32):
        super().__init__()
        self.model_path = "ImageReward-v1.0"
        self.device = device
        self.dtype = dtype
        self.model = RM.load(self.model_path, device=device).eval().to(dtype=dtype)
        self.model.requires_grad_(False)

    @paddle.no_grad()
    def __call__(self, prompts, images):
        rewards = []
        for prompt, image in zip(prompts, images):
            _, reward = self.model.inference_rank(prompt, [image])
            rewards.append(reward)
        return rewards


# Usage example
def main():
    scorer = ImageRewardScorer(device="cuda", dtype=paddle.float32)

    images = [
        "astronaut.jpg",
    ]
    pil_images = [Image.open(img) for img in images]
    prompts = [
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    print(scorer(prompts, pil_images))


if __name__ == "__main__":
    main()
