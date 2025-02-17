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

import json
from typing import Dict, List

import PIL.Image


def load_pil_images(conversations: List[Dict[str, str]]) -> List[PIL.Image.Image]:
    """

        Args:
            conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
                [
                    {
                        "role": "User",
                        "content": "<image>
    Extract all information from this image and convert them into markdown format.",
                        "images": ["./examples/table_datasets.png"]
                    },
                    {"role": "Assistant", "content": ""},
                ]

        Returns:
            pil_images (List[PIL.Image.Image]): the list of PIL images.

    """
    pil_images = []
    for message in conversations:
        if "images" not in message:
            continue
        for image_path in message["images"]:
            pil_img = PIL.Image.open(image_path)
            pil_img = pil_img.convert("RGB")
            pil_images.append(pil_img)
    return pil_images


def load_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
        return data
