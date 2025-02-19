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
from typing import Optional
from ...core import MMDataset, register
from functools import partial


# Define the file size filter function
def is_valid_image_file_size(
    item: dict,
    min_size_kb: Optional[float] = 10,
    max_size_kb: Optional[float] = None
) -> bool:
    """
    Checks whether the image file size is within the specified range.
    
    Args:
        item (dict): A dictionary containing image path and related information.
        min_size_kb (Optional[float]): Minimum file size in KB. Defaults to 10 KB.
        max_size_kb (Optional[float]): Maximum file size in KB. Defaults to None (no upper limit).
        
    Returns:
        bool: True if the file size meets the criteria; otherwise, False.
    """
    image_path = item.get('image')
    if not image_path or not os.path.exists(image_path):
        return False
    
    try:
        file_size_kb = os.path.getsize(image_path) / 1024  # Convert to KB
        if (min_size_kb is not None and file_size_kb < min_size_kb) or (max_size_kb is not None and file_size_kb > max_size_kb):
            return False
        return True
    except Exception as e:
        print(f"Error processing file size for {image_path}: {e}")
        return False


@register()
def image_filesize_filter(
    dataset: MMDataset, 
    min_size_kb: Optional[float] = 10, 
    max_size_kb: Optional[float] = None
) -> MMDataset:
    print("Filtering images with invalid file sizes...")
    
    # Use partial to bind parameters to the filter function
    filter_func = partial(
        is_valid_image_file_size, 
        min_size_kb=min_size_kb, 
        max_size_kb=max_size_kb
    )
    
    # Apply dataset.filter
    dataset = dataset.filter(
        func=filter_func, 
        max_workers=8, 
        progress=True
    )
    return dataset