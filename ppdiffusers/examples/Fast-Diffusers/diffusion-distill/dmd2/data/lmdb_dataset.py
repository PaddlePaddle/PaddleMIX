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


import lmdb
import numpy as np
import paddle
from utils import get_array_shape_from_lmdb, retrieve_row_from_lmdb


class LMDBDataset(paddle.io.Dataset):
    # LMDB version of an ImageDataset. It is suitable for large datasets.
    def __init__(self, dataset_path):
        # for supporting new datasets, please adapt the data type according to the one used in "main/data/create_imagenet_lmdb.py"
        self.KEY_TO_TYPE = {
            "labels": np.int64,
            "images": np.uint8,
        }

        self.dataset_path = dataset_path

        self.env = lmdb.open(dataset_path, readonly=True, lock=False, readahead=False, meminit=False)

        self.image_shape = get_array_shape_from_lmdb(self.env, "images")
        self.label_shape = get_array_shape_from_lmdb(self.env, "labels")

    def __len__(self):
        return self.image_shape[0]

    def __getitem__(self, idx):
        # final ground truth rgb image
        image = retrieve_row_from_lmdb(self.env, "images", self.KEY_TO_TYPE["images"], self.image_shape[1:], idx)
        # image = torch.tensor(image, dtype=torch.float32)
        image = image.astype(np.float32)

        label = retrieve_row_from_lmdb(self.env, "labels", self.KEY_TO_TYPE["labels"], self.label_shape[1:], idx)

        # label = torch.tensor(label, dtype=torch.long)
        label = label.astype(np.float64)
        image = image / 255.0

        output_dict = {"images": image, "class_labels": label}

        return output_dict
