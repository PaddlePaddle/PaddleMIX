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


def is_max_line_length_valid(item, min_length: int = 10, max_length: float = float('inf')) -> bool:
    """
    Checks whether the maximum line length in the conversations is within the specified range.

    Args:
        item (dict): A dictionary containing conversation information.
        min_length (int): Minimum maximum line length. Default is 10.
        max_length (float): Maximum maximum line length. Default is infinity.

    Returns:
        bool: True if the maximum line length is within [min_length, max_length], False otherwise.
    """
    # Clean the conversations by removing <image> placeholders
    cleaned_conversations = [
        [q.replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '').strip(), 
        a.strip()]
        for q, a in item['conversations']
    ]

    # Calculate the maximum line length in the conversations
    max_line_length = 0
    for q, a in cleaned_conversations:
        # Compare the length of questions and answers, and track the maximum
        max_line_length = max(max_line_length, len(q), len(a))

    # Check if the maximum line length is within the specified range
    return min_length <= max_line_length <= max_length


@register()
def maximum_line_length_filter(
    dataset, 
    min_length: Optional[int] = 10, 
    max_length: Optional[float] = float('inf')  # No upper limit by default
) -> MMDataset:
    """
    Filters the dataset based on the maximum line length in conversations.

    Args:
        dataset (MMDataset): The dataset to be filtered.
        min_length (int): Minimum maximum line length. Default is 10.
        max_length (float): Maximum maximum line length. Default is infinity.

    Returns:
        MMDataset: The filtered dataset.
    """
    print("Filtering samples with invalid maximum line lengths...")
    # Create the filter function
    filter_func = partial(is_max_line_length_valid, min_length=min_length, max_length=max_length)
    
    # Apply dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset