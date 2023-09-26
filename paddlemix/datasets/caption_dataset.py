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

from paddle.utils.download import get_path_from_url

from paddlemix.utils.env import DATA_HOME
from paddlemix.utils.log import logger

from .dataset import DatasetBuilder

__all__ = ["CaptionDataset"]


class CaptionDataset(DatasetBuilder):
    """
    Caption dataset.
    """

    URL = "https://bj.bcebos.com/v1/paddlenlp/datasets/paddlemix/coco.tar"
    META_INFO = collections.namedtuple("META_INFO", ("images", "annotations", "images_md5", "annotations_md5"))
    MD5 = ""
    SPLITS = {
        "train": META_INFO(
            os.path.join("coco", "images"),
            os.path.join("coco", "annotations/coco_karpathy_train.json"),
            "",
            "",
        ),
        "val": META_INFO(
            os.path.join("coco", "images"),
            os.path.join("coco", "annotations/coco_karpathy_val.json"),
            "",
            "",
        ),
        "test": META_INFO(
            os.path.join("coco", "images"),
            os.path.join("coco", "annotations/coco_karpathy_test.json"),
            "",
            "",
        ),
    }

    def _get_data(self, mode, **kwargs):
        logger.info("default dataset root is {}".format(DATA_HOME))
        images, annotations, image_hash, anno_hash = self.SPLITS[mode]
        image_fullname = os.path.join(DATA_HOME, images)
        anno_fullname = os.path.join(DATA_HOME, annotations)
        if not os.path.exists(image_fullname) or not os.path.exists(anno_fullname):
            get_path_from_url(self.URL, DATA_HOME)

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

    def _gen_image_id_eval(self, anno):
        img_ids = {}
        n = 0
        for ann in anno:
            img_id = ann["image"].split("/")[-1].strip(".jpg").split("_")[-1]
            if img_id not in img_ids.keys():
                img_ids[img_id] = n
                n += 1
        return img_ids

    def _read(self, filename, *args):
        image_root, anno_path, mode = filename
        annotations = json.load(open(anno_path, "r"))
        if mode == "val" or mode == "test":
            image_ids = self._gen_image_id_eval(annotations)
        else:
            image_ids = self._gen_image_id(annotations)
        for ann in annotations:
            image_path = os.path.join(image_root, ann["image"])
            if mode == "train":
                yield_data = {
                    "image": image_path,
                    "image_id": image_ids[ann["image_id"]],
                }
                # only train mode has text input
                yield_data["text_input"] = ann["caption"]
            else:
                yield_data = {
                    "image": image_path,
                    "image_id": ann["image"].split("/")[-1].strip(".jpg").split("_")[-1],
                }
            yield yield_data
