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

from paddlenlp.transformers.tokenizer_utils import ChatTemplateMixin

from .dataset import DatasetBuilder

__all__ = ["ChatMLDataset"]


class ChatMLDataset(DatasetBuilder, ChatTemplateMixin):
    """
    ChatMLDataset dataset.
    """

    SPLITS = {"train": "train.json", "val": "eval.json", "test": "test.json"}

    def _read(self, filename, *args):
        if self.config["chat_template"] is not None:
            self.init_chat_template(self.config["chat_template"])
        raw_data = json.load(open(filename, "r"))
        annotations = raw_data

        for ann in annotations:
            yield_data = {}
            conversations = ann["conversations"]
            if self.config["chat_template"] is not None:
                conversations.append([""])
                yield_data["conversations"] = self.apply_chat_template(conversations, tokenize=False)
            else:
                yield_data["conversations"] = conversations

            if "image" in ann.keys():
                yield_data["image"] = ann["image"]

            yield yield_data
