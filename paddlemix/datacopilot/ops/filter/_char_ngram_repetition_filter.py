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
import numpy as np


def is_char_ngram_valid(item, rep_len: int = 10, min_ratio: float = 0.0, max_ratio: float = 0.5) -> bool:
    """
    Checks whether the character n-gram repetition ratio in a conversation is within the specified range.

    Args:
        item (dict): A dictionary containing conversation information.
        rep_len (int): Length of the n-gram (default: 10).
        min_ratio (float): Minimum repetition ratio (default: 0.0).
        max_ratio (float): Maximum repetition ratio (default: 0.5).

    Returns:
        bool: True if the repetition ratio is within [min_ratio, max_ratio], False otherwise.
    """
    # Concatenate conversation content
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

    # Return False if the text length is smaller than n-gram length
    if len(user_conv) < rep_len:
        return False

    # Generate n-grams
    char_ngrams = [
        user_conv[i:i + rep_len]
        for i in range(len(user_conv) - rep_len + 1)
    ]

    # Count the frequency of each n-gram
    freq_char_ngrams = {}
    for ngram in char_ngrams:
        freq_char_ngrams[ngram] = freq_char_ngrams.get(ngram, 0) + 1

    # Return False if no valid n-grams are found
    if len(freq_char_ngrams) == 0:
        return False

    # Calculate the ratio of repetitive n-grams
    freq_values = list(freq_char_ngrams.values())
    total_ngrams = sum(freq_values)
    num_no_rep_ngrams = len([freq for freq in freq_values if freq == 1])
    num_rep_ngrams = min(
        int(np.sqrt(len(freq_values))),
        len(freq_values) - num_no_rep_ngrams
    )
    rep_ratio = sum(sorted(freq_values, reverse=True)[:num_rep_ngrams]) / total_ngrams

    # Check if the repetition ratio is within the specified range
    return min_ratio <= rep_ratio <= max_ratio


@register()
def char_ngram_repetition_filter(
    dataset: MMDataset, 
    rep_len: Optional[int] = 10, 
    min_ratio: Optional[float] = 0.0, 
    max_ratio: Optional[float] = 0.5
) -> MMDataset:
    """
    Filters the dataset based on the character n-gram repetition ratio in conversations.

    Args:
        dataset (MMDataset): The dataset to be filtered.
        rep_len (int): Length of the n-gram (default: 10).
        min_ratio (float): Minimum repetition ratio (default: 0.0).
        max_ratio (float): Maximum repetition ratio (default: 0.5).

    Returns:
        MMDataset: The filtered dataset.
    """
    print("Filtering samples with invalid character n-gram repetition ratios...")
    # Create the filter function
    filter_func = partial(is_char_ngram_valid, rep_len=rep_len, min_ratio=min_ratio, max_ratio=max_ratio)
    
    # Apply dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset