import PIL
import os
import numpy as np
from typing import Union
import requests
from PIL import ImageOps, Image
from argparse import ArgumentParser

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

    expected_max_diff = 0.1
    avg_diff = np.abs(image1 - image2).mean()

    assert avg_diff < expected_max_diff, f"Fail: Error image deviates {avg_diff} pixels on average"
    print(f"Success: Image diff test passed with {avg_diff} pixels on average")

if __name__ == "__main__":
    args = parse_args()
    image = load_image(args.source_image)
    expected_image = load_image(args.target_image)
    test_diff(image, expected_image)
    