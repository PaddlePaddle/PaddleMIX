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


import json

from .dataset import DatasetBuilder

__all__ = ["ChatMLDataset"]


class ChatMLDataset(DatasetBuilder):
    """
    ChatMLDataset dataset.
    """

    SPLITS = {"train": "train.json", "val": "eval.json", "test": "test.json"}
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    def _read(self, filename, *args):
        raw_data = json.load(open(filename, "r"))
        annotations = raw_data
        for ann in annotations:
            conversations = ann["conversations"]
            yield_data = []
            for conversation in conversations:
                sentence = {
                    "from": self.roles[conversation["from"]],
                    "value": conversation["value"],
                }
                yield_data.append(sentence)

            yield yield_data
