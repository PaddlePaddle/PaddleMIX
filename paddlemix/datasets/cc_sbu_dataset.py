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

# from dataset import DatasetBuilder
from .dataset import DatasetBuilder

__all__ = ["CCSBUAlignDataset"]


class CCSBUAlignDataset(DatasetBuilder):
    """
    CCSBUAlignDataset dataset.
    """

    URL = "https://paddlenlp.bj.bcebos.com/datasets/cc_sbu_align.zip"
    META_INFO = collections.namedtuple("META_INFO", ("images", "annotations", "num_images", "annotations_md5"))
    MD5 = "d5fa38be915c8a2aee7ebf3a9c56a95c"
    SPLITS = {
        "train": META_INFO(
            os.path.join("cc_sbu_align", "image"),
            os.path.join("cc_sbu_align", "filter_cap.json"),
            3439,
            "fa3508b6ac29e0ddc7246683d0c3d9a2",
        ),
    }

    def count_files(self, path):
        if not os.path.isdir(path):
            raise ValueError("A directory expected for path, but received {}".format(path))
        pathes = os.listdir(path)
        return len(pathes)

    def _get_data(self, mode, **kwargs):
        logger.info("default dataset root is {}".format(DATA_HOME))
        images, annotations, num_images, anno_hash = self.SPLITS[mode]
        image_fullname = os.path.join(DATA_HOME, images)
        anno_fullname = os.path.join(DATA_HOME, annotations)

        if (
            (not os.path.exists(image_fullname))
            or (not os.path.exists(anno_fullname))
            or (not md5file(anno_fullname) == anno_hash)
            or num_images != self.count_files(image_fullname)
        ):
            get_path_from_url(self.URL, DATA_HOME, self.MD5)

        return image_fullname, anno_fullname, mode

    def _gen_image_id(self, anno):
        img_ids = {}
        n = 0
        for ann in anno:
            # an ann example: {'image_id': '2', 'caption': 'The image shows a man fishing on a lawn next to a river with a bridge in the background. Trees can be seen on the other side of the river, and the sky is cloudy.'}
            img_id = ann["image_id"]
            if img_id not in img_ids.keys():
                img_ids[img_id] = n
                n += 1
        return img_ids

    def _read(self, filename, *args):
        image_root, anno_path, mode = filename
        with open(anno_path, "r", encoding="utf8") as f:
            annotations = json.load(f)["annotations"]
            image_ids = self._gen_image_id(annotations)

            for ann in annotations:
                image_path = os.path.join(image_root, ann["image_id"] + ".jpg")
                yield_data = {"image": image_path, "image_id": image_ids[ann["image_id"]]}
                if mode == "train":
                    # only train mode has text input
                    yield_data["text_input"] = ann["caption"]
                yield yield_data
