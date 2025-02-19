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


def is_avg_line_length_valid(item, min_length: int = 10, max_length: float = float('inf')) -> bool:
    """
    Checks whether the average line length of a conversation is within the specified range.

    Args:
        item (dict): A dictionary containing conversation information.
        min_length (int): Minimum average line length (default: 10).
        max_length (float): Maximum average line length (default: infinity).

    Returns:
        bool: True if the average line length is within [min_length, max_length], False otherwise.
    """
    # Concatenate conversations
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

    # Split content into lines
    lines = user_conv.splitlines()

    # Return False if there are no valid lines
    if not lines:
        return False

    # Calculate average line length
    avg_line_length = sum(len(line) for line in lines) / len(lines)

    # Check if the average line length is within the specified range
    return min_length <= avg_line_length <= max_length


@register()
def average_line_length_filter(
    dataset: MMDataset, 
    min_length: Optional[int] = 10, 
    max_length: Optional[float] = float('inf')  # Default is no upper limit
) -> MMDataset:
    """
    Filters the dataset based on the average line length of conversations.

    Args:
        dataset (MMDataset): The dataset to be filtered.
        min_length (int): Minimum average line length (default: 10).
        max_length (float): Maximum average line length (default: infinity).

    Returns:
        MMDataset: The filtered dataset.
    """
    print("Filtering samples with invalid average line lengths...")
    # Create the filter function
    filter_func = partial(is_avg_line_length_valid, min_length=min_length, max_length=max_length)
    
    # Apply dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset