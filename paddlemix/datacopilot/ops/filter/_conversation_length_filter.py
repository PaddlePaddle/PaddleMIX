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


def is_chat_length_valid(item, max_length: int = 2048) -> bool:
    """
    Check if the length of the conversation is less than the specified maximum length.

    Args:
        item (dict): A dictionary containing conversation information.
        max_length (int): The maximum allowed length for conversations (default: 2048).

    Returns:
        bool: Returns True if the conversation length is less than max_length; otherwise, False.
    """
    # Concatenate the content of conversations
    user_conv = '\n\n'.join(
        ''.join(conversation) for conversation in item['conversations']
    ).replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')

    return len(user_conv) < max_length


@register()
def conversation_length_filter(
    dataset: MMDataset, 
    max_length: Optional[int] = 2048, 
) -> MMDataset:
    print("Filtering out conversations that are too long...")
    # Create the filter function
    filter_func = partial(is_chat_length_valid, max_length=max_length)
    
    # Apply dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    
    return filtered_dataset