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

import collections
import os

__all__ = ["VGCaption"]
from paddlemix.datasets.caption_dataset import CaptionDataset


class VGCaption(CaptionDataset):
    """
    VG Caption dataset.
    """

    URL = "https://bj.bcebos.com/paddlemix/datasets/vg.tar.gz"
    META_INFO = collections.namedtuple("META_INFO", ("images", "annotations", "images_md5", "annotations_md5"))
    MD5 = ""
    SPLITS = {
        "train": META_INFO(
            os.path.join("coco", "images"),
            os.path.join("coco", "annotations/vg_caption.json"),
            "",
            "",
        ),
    }
