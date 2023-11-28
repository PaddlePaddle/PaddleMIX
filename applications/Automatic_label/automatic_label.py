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

import matplotlib.pyplot as plt
import numpy as np

from paddlemix.appflow import Appflow
from ppdiffusers.utils import load_image


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


task = Appflow(
    app="auto_label",
    models=["paddlemix/blip2-caption-opt2.7b", "GroundingDino/groundingdino-swint-ogc", "Sam/SamVitH-1024"],
)
url = "https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/overture-creations.png"
image_pil = load_image(url)
blip2_prompt = "describe the image"
result = task(image=image_pil, blip2_prompt=blip2_prompt)

plt.figure(figsize=(10, 10))
plt.imshow(result["image"])
for mask in result["seg_masks"]:
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box, label in zip(result["boxes"], result["labels"]):
        show_box(box, plt.gca(), label)

plt.axis("off")
plt.savefig(
    "mask_pred.jpg",
    bbox_inches="tight",
    dpi=300,
    pad_inches=0.0,
)
