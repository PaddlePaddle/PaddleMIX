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

from functools import partial
from typing import Dict

from ...core import MMDataset, register


# Define the conversion function
def convert_llava_item(item: Dict, image_path_prefix: str = "") -> Dict:
    """
    Convert each data item to the target format.

    Args:
        item (dict): Original data item containing 'image' and 'conversations' keys.
        image_path_prefix (str): Prefix for the image path. Defaults to an empty string.

    Returns:
        dict: Transformed data item containing 'image' and 'conversations' keys.
    """

    # Check if the 'image' key exists, if not, set it to an empty string
    image = item.get("image", "")  # Default to an empty string if 'image' key is missing

    # Skip this item if the 'image' field is empty
    if not image:
        return None  # Skip this item by returning None if no image exists

    # Concatenate the image path
    image = image_path_prefix + image
    # print(item['conversations'])

    conversations = []
    for i in range(0, len(item["conversations"]), 2):
        human_message = item["conversations"][i]["value"]
        gpt_message = item["conversations"][i + 1]["value"] if i + 1 < len(item["conversations"]) else ""
        conversations.append([human_message, gpt_message])

    # Construct the transformed data structure
    transformed_item = {"image": image, "conversations": conversations}

    return transformed_item


@register()
def llava_convert(dataset: MMDataset, image_path_prefix="") -> MMDataset:

    print("Converting llava dataset...")
    # Use the map operator for batch transformation
    filter_func = partial(convert_llava_item, image_path_prefix=image_path_prefix)

    # Apply dataset.map
    dataset = dataset.map(func=filter_func, max_workers=8, progress=True)

    return dataset
