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

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.vision.transforms as T
from PIL import Image


class MyDataset(paddle.io.Dataset):
    def __init__(self, dataset_path, resolution=512):
        self.dataset_path = dataset_path
        self.image_files = sorted([f for f in os.listdir(dataset_path) if f.endswith(".jpg") or f.endswith(".png")])
        self.caption_files = sorted([f for f in os.listdir(dataset_path) if f.endswith(".txt")])
        self.resolution = resolution

    def __getitem__(self, index):
        image_filename = self.image_files[index]
        caption_filename = self.caption_files[index]
        image_path = os.path.join(self.dataset_path, image_filename)
        caption_path = os.path.join(self.dataset_path, caption_filename)

        image = Image.open(image_path).convert("RGB")

        # 缩小图像，如需对paddle与torch的图像输入可以用以下代码取代后面的T.Resize逻辑
        # w, h = image.size
        # if w > h:
        #     image = image.resize((self.resolution, int(h * self.resolution / w)))
        # else:
        #     image = image.resize((int(w * self.resolution / h), self.resolution))

        # 读取标注
        with open(caption_path, "r") as file:
            caption = file.read().strip()

        img_preprocess = T.Compose(
            [
                T.ToTensor(),
                T.Resize(self.resolution, interpolation="bicubic"),
                # 按需进行图像裁剪等操作
                # T.RandomCrop(self.resolution, pad_if_needed=True),
            ]
        )

        image = img_preprocess(image)

        return {"caption": caption, "image": image}

    def __len__(self):
        return len(self.image_files)


def worker_init_fn(_):
    worker_info = paddle.io.get_worker_info()
    dataset = worker_info.dataset
    worker_id = worker_info.id

    local_rank = dist.get_rank()
    num_workers = worker_info.num_workers
    worker_id = worker_info.id
    worker_global_id = local_rank * num_workers + worker_id

    dataset.rng = np.random.RandomState(worker_global_id)
    return np.random.seed(np.random.get_state()[1][0] + worker_id)
