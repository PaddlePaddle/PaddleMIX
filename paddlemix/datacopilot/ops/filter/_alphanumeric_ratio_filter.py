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


from typing import Optional
from ...core import MMDataset, register
from functools import partial


def is_alnum_ratio_valid(item, min_ratio: float = 0.25, max_ratio: float = float('inf')) -> bool:
    """
    Checks whether the ratio of alphanumeric characters (letters or digits) 
    to the total number of characters in a sample is within the specified range.

    Args:
        item (dict): A dictionary containing text information.
        min_ratio (float): Minimum ratio (default: 0.25).
        max_ratio (float): Maximum ratio (default: infinity).

    Returns:
        bool: True if the ratio is within the range [min_ratio, max_ratio], False otherwise.
    """
    # Retrieve text content from conversations
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

    # Count the total number of alphanumeric characters
    alnum_count = sum(1 for char in user_conv if char.isalnum())
    
    # Calculate the ratio of alphanumeric characters
    alnum_ratio = alnum_count / len(user_conv) if len(user_conv) > 0 else 0.0

    # Check if the ratio is within the specified range
    return min_ratio <= alnum_ratio <= max_ratio


@register()
def alphanumeric_ratio_filter(
    dataset: MMDataset, 
    min_ratio: Optional[float] = 0.25, 
    max_ratio: Optional[float] = float('inf')
) -> MMDataset:
    """
    Filters the dataset based on the ratio of alphanumeric characters in each sample.

    Args:
        dataset (MMDataset): The dataset to be filtered.
        min_ratio (float): Minimum ratio (default: 0.25).
        max_ratio (float): Maximum ratio (default: infinity).

    Returns:
        MMDataset: The filtered dataset.
    """
    print("Filtering samples based on the ratio of alphanumeric characters...")
    # Create the filter function
    filter_func = partial(is_alnum_ratio_valid, min_ratio=min_ratio, max_ratio=max_ratio)
    
    # Apply dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset