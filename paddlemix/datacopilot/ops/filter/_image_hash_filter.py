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
from PIL import Image
from typing import Optional
from ...core import MMDataset, register
import imagehash
from functools import partial


def is_valid_image_hash(
    item,
    seen_hashes: set,
    hash_method: str = "phash"
) -> bool:
    """
    Determines whether an image should be kept (based on hash deduplication).

    Args:
        item (dict): A sample dictionary containing the image path.
        seen_hashes (set): A set to record already encountered hash values.
        hash_method (str): The type of hash algorithm to use. Supports "phash" (default), "dhash", and "average_hash".

    Returns:
        bool: True if the image should be kept; otherwise, False.
    """
    image_path = item.get('image')
    if not image_path or not os.path.exists(image_path):
        return False

    try:
        with Image.open(image_path) as img:
            # Compute the hash value
            if hash_method == "phash":
                img_hash = str(imagehash.phash(img))
            elif hash_method == "dhash":
                img_hash = str(imagehash.dhash(img))
            elif hash_method == "average_hash":
                img_hash = str(imagehash.average_hash(img))
            else:
                raise ValueError(f"Unsupported hash method: {hash_method}")
            
            # Check if the hash value already exists
            if img_hash in seen_hashes:
                return False
            seen_hashes.add(img_hash)
            return True
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False


@register()
def image_hash_filter(
    dataset,
    hash_method: Optional[str] = "phash"
) -> MMDataset:
    """
    Filters the dataset using image hash values.

    Args:
        dataset (MMDataset): The input dataset.
        hash_method (Optional[str]): The type of hash algorithm to use, default is "phash".

    Returns:
        MMDataset: The dataset after filtering.
    """
    print("Filtering duplicate images...")

    # Initialize a set to track encountered hash values
    seen_hashes = set()
    
    # Create the filter function, binding seen_hashes
    filter_func = partial(is_valid_image_hash, seen_hashes=seen_hashes, hash_method=hash_method)
    
    # Apply dataset.filter
    filtered_dataset = dataset.filter(
        func=filter_func,
        max_workers=8,
        progress=True
    )
    
    return filtered_dataset