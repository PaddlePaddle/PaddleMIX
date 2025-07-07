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

import zipfile

import numpy as np
import paddle
from coco_eval.cleanfid.resize import build_resizer
from PIL import Image


class ToTensor:
    def __call__(self, x):
        if x.dtype == "uint8":
            x = x.astype("float32") / 255.0
        return x.transpose([2, 0, 1])


class ResizeDataset(paddle.io.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, files, mode, size=(299, 299), fdir=None):
        self.files = files
        self.fdir = fdir
        self.transforms = ToTensor()
        self.size = size
        self.fn_resize = build_resizer(mode)
        self.custom_image_tranform = lambda x: x
        self._zipfile = None

    def _get_zipfile(self):
        assert self.fdir is not None and ".zip" in self.fdir
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.fdir)
        return self._zipfile

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = str(self.files[i])
        if self.fdir is not None and ".zip" in self.fdir:
            with self._get_zipfile().open(path, "r") as f:
                img_np = np.array(Image.open(f).convert("RGB"))
        elif ".npy" in path:
            img_np = np.load(path)
        else:
            img_pil = Image.open(path).convert("RGB")
            img_np = np.array(img_pil)

        # apply a custom image transform before resizing the image to 299x299
        img_np = self.custom_image_tranform(img_np)
        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)

        if img_resized.dtype == "uint8":
            img_t = self.transforms(np.array(img_resized))
        elif img_resized.dtype == "float32":
            img_t = self.transforms(img_resized)

        return img_t


EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp", "npy", "JPEG", "JPG", "PNG"}


class ResizeArrayDataset(paddle.io.Dataset):
    """
    A placeholder Dataset that enables parallelizing the resize operation
    using multiple CPU cores

    files: list of all files in the folder
    fn_resize: function that takes an np_array as input [0,255]
    """

    def __init__(self, array, mode, size=(299, 299)):
        self.array = array
        self.transforms = ToTensor()
        self.size = size
        self.fn_resize = build_resizer(mode)
        self.custom_image_tranform = lambda x: x

    def __len__(self):
        return len(self.array)

    def __getitem__(self, i):
        img_np = self.array[i]

        # apply a custom image transform before resizing the image to 299x299
        img_np = self.custom_image_tranform(img_np)
        # fn_resize expects a np array and returns a np array
        img_resized = self.fn_resize(img_np)

        if img_resized.dtype == "uint8":
            img_t = self.transforms(np.array(img_resized))
        elif img_resized.dtype == "float32":
            img_t = self.transforms(img_resized)

        return img_t
