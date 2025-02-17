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


import numpy as np
from paddlemix.datacopilot.core import MMDataset, register


@register()
def conversation_percentage_filter(dataset: MMDataset, min_percentile: float, max_percentile: float) -> MMDataset:
    """
    Filter dataset items based on the percentile range of the number of conversations.

    Args:
        dataset (MMDataset): The dataset to be filtered.
        min_percentile (float): The minimum percentile (e.g., 0 for the 0th percentile).
        max_percentile (float): The maximum percentile (e.g., 95 for the 95th percentile).
        
    Returns:
        MMDataset: The filtered dataset.
    """
    print("Filtering conversations based on percentiles.")
    if not (0 <= min_percentile <= 100 and 0 <= max_percentile <= 100):
        raise ValueError("Percentile values must be between 0 and 100.")

    # Count the number of conversations
    conversation_counts = np.array([
        len(item.get("conversations", [])) for item in dataset.items
    ])
    
    # Calculate the percentile thresholds
    min_threshold = np.percentile(conversation_counts, min_percentile)
    max_threshold = np.percentile(conversation_counts, max_percentile)

    print(f"Filtering conversations within range: {min_threshold} to {max_threshold}")

    # Filter the dataset
    filtered_items = [
        item for item, count in zip(dataset.items, conversation_counts)
        if min_threshold <= count <= max_threshold
    ]

    return MMDataset(filtered_items)