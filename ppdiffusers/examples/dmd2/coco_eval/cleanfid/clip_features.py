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

# pip install git+https://github.com/openai/CLIP.git

import clip
import numpy as np
import torch
import torchvision.transforms as transforms
from cleanfid.fid import compute_fid
from PIL import Image


def img_preprocess_clip(img_np):
    x = Image.fromarray(img_np.astype(np.uint8)).convert("RGB")
    T = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
        ]
    )
    return np.asarray(T(x)).clip(0, 255).astype(np.uint8)


class CLIP_fx:
    def __init__(self, name="ViT-B/32", device="cuda"):
        self.model, _ = clip.load(name, device=device)
        self.model.eval()
        self.name = "clip_" + name.lower().replace("-", "_").replace("/", "_")

    def __call__(self, img_t):
        img_x = img_t / 255.0
        T_norm = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        img_x = T_norm(img_x)
        assert torch.is_tensor(img_x)
        if len(img_x.shape) == 3:
            img_x = img_x.unsqueeze(0)
        B, C, H, W = img_x.shape
        with torch.no_grad():
            z = self.model.encode_image(img_x)
        return z
