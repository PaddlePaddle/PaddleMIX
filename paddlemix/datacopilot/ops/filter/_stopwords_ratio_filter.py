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


import os
from functools import partial
from typing import Optional
import nltk
from nltk.corpus import stopwords
from ...core import MMDataset, register

# Global variable to track stopwords set
_stop_words = None

def ensure_stopwords_downloaded():
    """
    Ensure NLTK stopwords are downloaded only once.
    """
    nltk_data_path = os.path.expanduser('~') + '/nltk_data/corpora/stopwords'
    if not os.path.exists(nltk_data_path):
        nltk.download('stopwords', quiet=True)

def load_stopwords():
    """
    Load the stopwords set, ensuring thread safety and one-time initialization.
    """
    global _stop_words
    if _stop_words is None:
        ensure_stopwords_downloaded()  # Ensure stopwords are downloaded
        _stop_words = set(stopwords.words('english'))
    return _stop_words

def is_stopwords_ratio_valid(item, stop_words: set, min_ratio: float = 0.25) -> bool:
    """
    Check if the ratio of stopwords in the sample is greater than or equal to the specified minimum value.

    Args:
        item (dict): A dictionary containing text information.
        stop_words (set): A set of stopwords.
        min_ratio (float): Minimum stopword ratio. Default is 0.25.

    Returns:
        bool: True if the stopword ratio is greater than or equal to min_ratio; otherwise, False.
    """
    # Concatenate conversation content
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

    # Split text into words
    words = user_conv.split()

    # Count the number of stopwords
    stopword_count = sum(1 for word in words if word.lower() in stop_words)

    # Calculate the stopword ratio
    stopword_ratio = stopword_count / len(words) if len(words) > 0 else 0.0

    # Check if the ratio meets the requirement
    return stopword_ratio >= min_ratio

@register()
def stopwords_ratio_filter(
    dataset, 
    min_ratio: Optional[float] = 0.25
) -> MMDataset:
    """
    Filter the dataset based on the stopword ratio of the samples.

    Args:
        dataset (MMDataset): The dataset to be filtered.
        min_ratio (float): Minimum stopword ratio. Default is 0.25.

    Returns:
        MMDataset: The filtered dataset.
    """
    print("Filtering samples that do not meet the stopword ratio requirement...")
    
    # Load stopwords once
    stop_words = load_stopwords()
    
    # Create the filter function
    filter_func = partial(is_stopwords_ratio_valid, stop_words=stop_words, min_ratio=min_ratio)
    
    # Apply dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset
