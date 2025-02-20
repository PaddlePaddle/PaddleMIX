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
import sys
from typing import Optional
from functools import partial
from ...core import MMDataset, register


# Define the function to compute token count
def compute_token_count(user_conv: str, tokenizer) -> int:
    """
    Compute the number of tokens in the sample (conversation).

    Args:
        user_conv (str): Merged conversation text.
        tokenizer (AutoTokenizer): Tokenizer instance used for tokenization.

    Returns:
        int: The number of tokens in the sample.
    """
    tokens = tokenizer(user_conv, truncation=True, return_tensors="pd", use_fast=True)["input_ids"].flatten()
    return len(tokens)


@register()
def token_num_filter(
    dataset: MMDataset, 
    tokenizer_model: str = "Qwen/Qwen2.5-7B", 
    min_tokens: Optional[int] = 10, 
    max_tokens: Optional[int] = sys.maxsize
) -> MMDataset:
    """
    Filter the dataset based on the number of tokens in each sample.

    Args:
        dataset (MMDataset): The dataset to be filtered.
        tokenizer_model (str): Name of the tokenizer model to use, default is `Qwen/Qwen2.5-7B`.
        min_tokens (int): Minimum number of tokens. Default is 10.
        max_tokens (int): Maximum number of tokens. Default is `sys.maxsize`.

    Returns:
        MMDataset: The filtered dataset.
    """
    print(f"Filtering samples based on token count: min tokens = {min_tokens}, max tokens = {max_tokens}...")

    # Initialize the tokenizer
    from paddlenlp.transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    def filter_func(item):
        # Get and clean conversation text
        user_conv = '\n\n'.join(
            ''.join(conversation) for conversation in item['conversations']
        ).replace('<image>', '').replace('\n', '')  # Clean `<image>` tags and newlines

        # Compute the number of tokens
        num_tokens = compute_token_count(user_conv, tokenizer)

        # Check if the token count is within the specified range
        return min_tokens <= num_tokens <= max_tokens

    # Apply dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset