# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from argparse import ArgumentParser
from typing import Union

import numpy as np
import PIL
import requests
from PIL import Image, ImageOps  # noqa


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--source_image", type=str, required=True)
    parser.add_argument("--target_image", type=str, required=True)
    args = parser.parse_args()
    return args


def load_image(image: Union[str, PIL.Image.Image]) -> PIL.Image.Image:
    """
    Loads `image` to a PIL Image.
    Args:
        image (`str` or `PIL.Image.Image`):
            The image to convert to the PIL Image format.
    Returns:
        `PIL.Image.Image`:
            A PIL Image.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = PIL.Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def test_diff(image1, image2):
    if isinstance(image1, PIL.Image.Image):
        image1 = np.array(image1)
    if isinstance(image2, PIL.Image.Image):
        image2 = np.array(image2)

    expected_max_diff = 30
    avg_diff = np.abs(image1 - image2).mean()

    assert avg_diff < expected_max_diff, f"FAILED: Error image deviates {avg_diff} pixels on average"
    print(f"PASSED: Image diff test passed with {avg_diff} pixels on average")


if __name__ == "__main__":
    args = parse_args()
    image = load_image(args.source_image)
    expected_image = load_image(args.target_image)
    test_diff(image, expected_image)
