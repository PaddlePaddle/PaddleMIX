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


def is_word_ngram_valid(item, rep_len: int = 10, min_ratio: float = 0.0, max_ratio: float = 0.5) -> bool:
    """
    Checks whether the word n-gram repetition ratio in the conversation is within the specified range.

    Args:
        item (dict): A dictionary containing conversation information.
        rep_len (int): Length of the n-gram. Default is 10.
        min_ratio (float): Minimum repetition ratio. Default is 0.0.
        max_ratio (float): Maximum repetition ratio. Default is 0.5.

    Returns:
        bool: True if the repetition ratio is within [min_ratio, max_ratio], otherwise False.
    """
    # Concatenate conversation content
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

    # Return False if the text length is smaller than the n-gram length
    if len(user_conv.split()) < rep_len:  # Based on word count
        return False

    # Generate n-grams
    words = user_conv.split()  # Split by whitespace
    word_ngrams = [
        ' '.join(words[i:i + rep_len])
        for i in range(len(words) - rep_len + 1)
    ]

    # Count the frequency of each n-gram
    freq_word_ngrams = {}
    for ngram in word_ngrams:
        freq_word_ngrams[ngram] = freq_word_ngrams.get(ngram, 0) + 1

    # Return False if there are no valid n-grams
    if len(freq_word_ngrams) == 0:
        return False

    # Calculate the ratio of repetitive n-grams
    freq_values = list(freq_word_ngrams.values())
    total_ngrams = sum(freq_values)
    num_no_rep_ngrams = len([freq for freq in freq_values if freq == 1])
    num_rep_ngrams = min(
        int(np.sqrt(len(freq_values))),
        len(freq_values) - num_no_rep_ngrams
    )
    rep_ratio = sum(sorted(freq_values, reverse=True)[:num_rep_ngrams]) / total_ngrams

    # Check if the ratio is within the specified range
    return min_ratio <= rep_ratio <= max_ratio


@register()
def word_ngram_repetition_filter(
    dataset, 
    rep_len: Optional[int] = 10, 
    min_ratio: Optional[float] = 0.0, 
    max_ratio: Optional[float] = 0.2
) -> MMDataset:
    """
    Filters the dataset based on the word n-gram repetition ratio in the conversations.

    Args:
        dataset (MMDataset): The dataset to be filtered.
        rep_len (int): Length of the n-gram. Default is 10.
        min_ratio (float): Minimum repetition ratio. Default is 0.0.
        max_ratio (float): Maximum repetition ratio. Default is 0.2.

    Returns:
        MMDataset: The filtered dataset.
    """
    print("Filtering samples with invalid word n-gram repetition ratios...")
    # Create the filter function
    filter_func = partial(is_word_ngram_valid, rep_len=rep_len, min_ratio=min_ratio, max_ratio=max_ratio)
    
    # Apply dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset