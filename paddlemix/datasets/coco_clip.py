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
import json
import os
from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url

from paddlemix.utils.env import DATA_HOME
from paddlemix.utils.log import logger

__all__ = ["CaptionCLIP"]

from .dataset import DatasetBuilder


class CaptionCLIP(DatasetBuilder):

    URL = "https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/coco.tar"
    META_INFO = collections.namedtuple(
        "META_INFO", ("images", "annotations", "images_md5", "annotations_md5"))
    MD5 = "e670ce82b14b3f45d08c9370808ee1e7"
    SPLITS = {
        "train": META_INFO(
            os.path.join("coco", "images"),
            os.path.join("coco", "annotations/coco_karpathy_train.json"),
            "",
            "aa31ac474cf6250ebb81d18348a07ed8",
        ),
        "val": META_INFO(
            os.path.join("coco", "images"),
            os.path.join("coco", "annotations/coco_karpathy_val.json"),
            "",
            "b273847456ef5580e33713b1f7de52a0",
        ),
        "test": META_INFO(
            os.path.join("coco", "images"),
            os.path.join("coco", "annotations/coco_karpathy_test.json"),
            "",
            "3ff34b0ef2db02d01c37399f6a2a6cd1",
        ),
    }

    def _get_data(self, mode, **kwargs):
        logger.info("default dataset root is {}".format(DATA_HOME))
        images, annotations, image_hash, anno_hash = self.SPLITS[mode]
        image_fullname = os.path.join(DATA_HOME, images)
        anno_fullname = os.path.join(DATA_HOME, annotations)
        if (not os.path.exists(image_fullname) or not os.path.exists(anno_fullname) or not md5file(anno_fullname) == anno_hash):
            get_path_from_url(self.URL, DATA_HOME, self.MD5)

        return image_fullname, anno_fullname, mode

    def _gen_image_id(self, anno):
        img_ids = {}
        n = 0
        for ann in anno:
            img_id = ann["image_id"]
            if img_id not in img_ids.keys():
                img_ids[img_id] = n
                n += 1
        return img_ids

    def _read(self, filename, *args):
        image_root, anno_path, mode = filename
        annotations = json.load(open(anno_path, "r"))

        for ann in annotations:
            image_path = os.path.join(image_root, ann["image"])
            yield_data = {"image": image_path}
            if mode == "train":
                # only train mode has text input
                yield_data["text"] = ann["caption"]
            yield yield_data