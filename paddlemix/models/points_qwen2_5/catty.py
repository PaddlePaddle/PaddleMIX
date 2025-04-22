# -*- coding: utf-8 -*-

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

# @Time    : 2025/4/19 下午8:37
# @Author  : zhaop-l(zhaopuzxjc@126.com)

import os
from typing import List, Tuple

from PIL import Image

from .dynamic_high_resolution import factorize_number


def construct_mapping_dict(max_splits: int = 12) -> dict:
    """Construct a mapping dictionary for the given max_splits.

    Args:
        max_splits (int, optional): The maximum number of splits.
            Defaults to 12.

    Returns:
        dict: A mapping dictionary for the given max_splits.
    """
    mapping_dict = {}
    for i in range(1, max_splits + 1):
        factor_list = factorize_number(i)
        for factor in factor_list:
            ratio = factor[0] / factor[1]
            if ratio not in mapping_dict:
                mapping_dict[ratio] = [factor]
            else:
                mapping_dict[ratio].append(factor)
    return mapping_dict


def save_image_list(image_list: List[Image.Image], save_folder: str) -> None:
    """Save a list of images to a folder.

    Args:
        image_list (List[Image.Image]): A list of images.
        save_folder (str): The folder to save the images to.
    """
    os.makedirs(save_folder, exist_ok=True)
    for i, image in enumerate(image_list):
        image.save(os.path.join(save_folder, f"{i}.png"))


def resize_to_best_size(
    image: Image.Image,
    best_slices: tuple,
    width_slices: int,
    height_slices: int,
    sub_image_size: int,
) -> Image.Image:
    """Resize an image to the best size for the given number of slices.

    Args:
        image (Image.Image): The image to resize.
        best_slices (tuple): The best number of slices for the image.
        width_slices (int): The number of horizontal slices.
        height_slices (int): The number of vertical slices.
        sub_image_size (int): The size of the sub-images.

    Returns:
        Image.Image: The resized image.
    """
    width, height = image.size
    best_width_slices, best_height_slices = best_slices
    if width_slices < height_slices:
        new_image_width = best_width_slices * sub_image_size
        new_image_height = int(height / width * new_image_width)
    else:
        new_image_height = best_height_slices * sub_image_size
        new_image_width = int(width / height * new_image_height)
    new_image = image.resize((new_image_width, new_image_height), resample=2)
    return new_image


def compute_strides(height: int, width: int, sub_image_size: int, slices: Tuple[int, int]) -> Tuple[int, int]:
    """Compute the strides for the given image size and slices.

    Args:
        height (int): The height of the image.
        width (int): The width of the image.
        sub_image_size (int): The size of the sub-images.
        slices (Tuple[int, int]): The number of horizontal and vertical slices.

    Returns:
        Tuple[int, int]: The strides for the given image size and slices.
    """
    slice_width, slice_height = slices
    if slice_width > 1:
        stride_x = (width - sub_image_size) // (slice_width - 1)
    else:
        stride_x = 0
    if slice_height > 1:
        stride_y = (height - sub_image_size) // (slice_height - 1)
    else:
        stride_y = 0
    return stride_x, stride_y


def sliding_window_crop(image: Image.Image, window_size: int, slices: Tuple[int, int]) -> List[Image.Image]:
    """Crop an image into sub-images using a sliding window.

    Args:
        image (Image.Image): The image to crop.
        window_size (int): The size of the sub-images.
        slices (Tuple[int, int]): The number of horizontal and vertical slices.

    Returns:
        List[Image]: A list of cropped images.
    """
    width, height = image.size
    stride_x, stride_y = compute_strides(height, width, window_size, slices)
    sub_images = []
    if stride_x == 0:
        stride_x = window_size
    if stride_y == 0:
        stride_y = window_size
    for y in range(0, height - window_size + 1, stride_y):
        for x in range(0, width - window_size + 1, stride_x):
            sub_image = image.crop((x, y, x + window_size, y + window_size))
            sub_images.append(sub_image)
    return sub_images


def find_best_slices(width_slices: int, height_slices: int, aspect_ratio: float, max_splits: int = 12) -> list:
    """Find the best slices for the given image size and aspect ratio.

    Args:
        width_slices (int): The number of horizontal slices.
        height_slices (int): The number of vertical slices.
        aspect_ratio (float): The aspect ratio of the image.
        max_splits (int, optional): The maximum number of splits.
            Defaults to 12.

    Returns:
        list: the best slices for the given image.
    """
    mapping_dict = construct_mapping_dict(max_splits)
    if aspect_ratio < 1:
        mapping_dict = {k: v for k, v in mapping_dict.items() if k <= aspect_ratio}
    elif aspect_ratio > 1:
        mapping_dict = {k: v for k, v in mapping_dict.items() if k >= aspect_ratio}
    best_ratio = min(mapping_dict.keys(), key=lambda x: abs(x - aspect_ratio))
    best_image_sizes = mapping_dict[best_ratio]
    best_slices = min(best_image_sizes, key=lambda x: abs(x[0] * x[1] - width_slices * height_slices))
    return best_slices


def split_image_with_catty(
    pil_image: Image.Image,
    image_size: int = 336,
    max_crop_slices: int = 8,
    save_folder: str = None,
    add_thumbnail: bool = True,
    do_resize: bool = False,
    **kwargs,
) -> List[Image.Image]:
    """Split an image into sub-images using Catty.

    Args:
        pil_image (Image.Image): The image to split.
        image_size (int, optional): The size of the image.
            Defaults to 336.
        max_crop_slices (int, optional): The maximum number of slices.
            Defaults to 8.
        save_folder (str, optional): The folder to save the sub-images.
            Defaults to None.
        add_thumbnail (bool, optional): Whether to add a thumbnail.
            Defaults to False.
        do_resize (bool, optional): Whether to resize the image to fit the
            maximum number of slices. Defaults to False.

    Returns:
        List[Image.Image]: A list of cropped images.
    """
    width, height = pil_image.size
    ratio = width / height
    if ratio > max_crop_slices or ratio < 1 / max_crop_slices:
        if do_resize:
            print(f"Resizing image to fit maximum number of slices ({max_crop_slices})")
            if width > height:
                new_width = max_crop_slices * height
                new_height = height
            else:
                new_width = width
                new_height = max_crop_slices * width
            pil_image = pil_image.resize((new_width, new_height), resample=2)
            width, height = pil_image.size
            ratio = width / height
        else:
            print(
                f"Image aspect ratio ({ratio:.2f}) is out of range: ({1 / max_crop_slices:.2f}, {max_crop_slices:.2f})"
            )
            return None
    width_slices = width / image_size
    height_slices = height / image_size
    best_slices = find_best_slices(width_slices, height_slices, ratio, max_crop_slices)
    pil_image = resize_to_best_size(pil_image, best_slices, width_slices, height_slices, image_size)
    width, height = pil_image.size
    sub_images = sliding_window_crop(pil_image, image_size, best_slices)
    if add_thumbnail:
        thumbnail_image = pil_image.resize((image_size, image_size), resample=2)
        sub_images.append(thumbnail_image)
    if save_folder is not None:
        save_image_list(sub_images, save_folder)
    return sub_images
