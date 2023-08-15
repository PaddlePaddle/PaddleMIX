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

from paddlevlp.utils.env import DATA_HOME
from paddlevlp.utils.log import logger

from .dataset import DatasetBuilder

# from paddle.dataset.common import md5file
# from paddle.utils.download import get_path_from_url

__all__ = ["VQADataset", "VQAEvalDataset"]


class VQADataset(DatasetBuilder):
    """
    Caption dataset.
    """

    URL = "https://bj.bcebos.com/paddlemix/datasets/coco.tar.gz"
    META_INFO = collections.namedtuple(
        "META_INFO", ("images", "annotations", "images_md5", "annotations_md5"))
    MD5 = ""
    SPLITS = {
        "train": META_INFO(
            os.path.join("coco", "images"),
            [
                os.path.join("coco", "annotations/vqa_train.json"),
                os.path.join("coco", "annotations/vqa_val.json")
            ],
            "",
            "aa31ac474cf6250ebb81d18348a07ed8", ),
        "val": META_INFO(
            os.path.join("coco", "images"),
            [
                os.path.join("coco", "annotations/vqa_val_eval.json"),
                os.path.join("coco", "annotations/answer_list.json"),
                os.path.join(
                    "coco",
                    "annotations/v2_OpenEnded_mscoco_val2014_questions.json"),
                os.path.join("coco",
                             "annotations/v2_mscoco_val2014_annotations.json"),
            ],
            "",
            "b273847456ef5580e33713b1f7de52a0", ),
        "test": META_INFO(
            os.path.join("coco", "images"),
            [
                os.path.join("coco", "annotation/vqa_test.json"),
                os.path.join("coco", "annotation/vqa_test.json"),
            ],
            "",
            "3ff34b0ef2db02d01c37399f6a2a6cd1", ),
    }

    def _get_data(self, mode, **kwargs):
        # default_root = '/paddle/wangguanzhong/blip-jinman/PaddleNLP/blip2'
        logger.info("default dataset root is {}".format(DATA_HOME))
        images, annotations, image_hash, anno_hash = self.SPLITS[mode]
        image_fullname = os.path.join(DATA_HOME, images)
        if isinstance(annotations, (list, tuple)):
            anno_fullname = []
            for ann in annotations:
                anno_fullname.append(os.path.join(DATA_HOME, ann))
        else:
            anno_fullname = os.path.join(DATA_HOME, annotations)
        return image_fullname, anno_fullname, mode

    def _read(self, filename, *args):
        image_root, anno_path, mode = filename
        annotations = []
        if mode == "val" or mode == "test":
            annotations = json.load(open(anno_path[0]))
            image_ids = self._gen_image_id_eval(annotations)
        else:
            for ann_p in anno_path:
                annotations.extend(json.load(open(ann_p, "r")))
            image_ids = self._gen_image_id(annotations)
        for ann in annotations:
            image_path = os.path.join(image_root, ann["image"])
            if mode == "train":
                yield_data = {"image": image_path, }
                yield_data["text_input"] = ann["question"]
                yield_data["answers"]: ann["answers"]
                yield_data["image_ids"]: ann["image_ids"]

            else:
                yield_data = {
                    "image": image_path,
                    "text_input": ann["question"],
                    "question_id": ann["question_id"],
                    "image_id":
                    ann["image"].split("/")[-1].strip(".jpg").split("_")[-1]
                }
                yield_data["image_ids"]: ann["image_ids"]
            yield yield_data

    def _gen_image_id(self, anno):
        img_ids = {}
        n = 0
        for ann in anno:
            if "image_id" not in ann.keys():
                img_id = ann["image"].split("/")[-1].strip(".jpg").split("_")[
                    -1]
            else:
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
