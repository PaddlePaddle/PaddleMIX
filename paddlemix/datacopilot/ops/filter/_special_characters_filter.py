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


from typing import Optional
from ...core import MMDataset, register
from functools import partial


def is_special_char_ratio_valid(item, min_ratio: float = 0.0, max_ratio: float = 0.25) -> bool:
    """
    Checks whether the ratio of special characters in the sample is within the specified range.

    Args:
        item (dict): A dictionary containing text information.
        min_ratio (float): Minimum special character ratio. Default is 0.0.
        max_ratio (float): Maximum special character ratio. Default is 0.25.

    Returns:
        bool: True if the special character ratio is within [min_ratio, max_ratio], False otherwise.
    """
    # Concatenate conversation content
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

    # Count the number of special characters
    special_characters = [
        '|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^', '\'', '\"', '’',
        '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
    ]
    special_char_count = sum(1 for char in user_conv if char in special_characters)

    # Calculate the ratio of special characters
    total_chars = len(user_conv)
    special_char_ratio = special_char_count / total_chars if total_chars > 0 else 0.0

    # Check if the ratio is within the specified range
    return min_ratio <= special_char_ratio <= max_ratio


@register()
def special_characters_filter(
    dataset: MMDataset, 
    min_ratio: Optional[float] = 0.0, 
    max_ratio: Optional[float] = 0.25
) -> MMDataset:
    """
    Filters the dataset based on the ratio of special characters in the samples.

    Args:
        dataset (MMDataset): The dataset to be filtered.
        min_ratio (float): Minimum special character ratio. Default is 0.0.
        max_ratio (float): Maximum special character ratio. Default is 0.25.

    Returns:
        MMDataset: The filtered dataset.
    """
    print("Filtering samples with invalid special character ratios...")
    # Create the filter function
    filter_func = partial(is_special_char_ratio_valid, min_ratio=min_ratio, max_ratio=max_ratio)
    
    # Apply dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset