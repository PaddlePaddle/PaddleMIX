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


from PIL import Image
from ...core import MMDataset, register


def image_compliance_operator(item) -> bool:
    """
    Validate whether the image in the dataset item can be loaded successfully.
    
    Args:
        item (dict): A dataset item containing the 'image' key.
        
    Returns:
        bool: Returns True if the image is valid, otherwise False.
    """
    try:
        image_path = item['image']
        with Image.open(image_path) as img:
            img.load()  # Force image data to load
        return True
    except Exception as e:
        print(f"Invalid image: {image_path}, Error: {e}")
        return False


def conversation_compliance_operator(item) -> bool:
    """
    Validate whether the 'conversations' in the dataset item are compliant.
    If they contain 'USER' or 'ASSISTANT', return False and print an error message.

    Args:
        item (dict): A dataset item containing the 'conversations' key.

    Returns:
        bool: Returns True if conversations are valid, otherwise False.
    """
    if 'conversations' not in item or not isinstance(item['conversations'], list):
        print(f"Invalid conversation format in dataset item: {item}")
        return False

    conversations = item['conversations']

    for conv in conversations:
        # Check if 'USER' or 'ASSISTANT' are present in any part of the conversation
        if any('USER' in part or 'ASSISTANT' in part for part in conv):
            print(f"Conversation contains 'USER' or 'ASSISTANT': {item}")
            return False
        
        # Each conversation pair must be a list or tuple with exactly two elements
        if not isinstance(conv, (list, tuple)) or len(conv) != 2:
            print(f"Invalid conversation pair structure in dataset item: {item}")
            return False
        
        # Each part of the conversation pair must be a string
        if not all(isinstance(part, str) for part in conv):
            print(f"Conversation pair contains non-string elements: {item}")
            return False
        
        # Content must not be empty
        if not all(part.strip() for part in conv):
            print(f"Conversation contains empty content in dataset item: {item}")
            return False
        
    return True


@register()
def valid_data_filter(dataset: MMDataset) -> MMDataset:
    # Filter out images that cannot be loaded
    print("Filtering invalid images...")
    dataset = dataset.filter(
        func=image_compliance_operator, 
        max_workers=8, 
        progress=True
    )

    # Filter out invalid conversations
    print("Filtering invalid conversations...")
    dataset = dataset.filter(
        func=conversation_compliance_operator, 
        max_workers=8, 
        progress=True
    )
    return dataset