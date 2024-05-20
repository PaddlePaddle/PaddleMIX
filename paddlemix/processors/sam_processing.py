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
"""
Processor class for Sam.
"""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
from paddle.nn import functional as F
from paddle.vision.transforms.functional import resize

from .base_processing import ProcessorMixin
from .image_transform_utils import to_pil_image
from .image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    get_preprocess_shape,
    valid_images,
)
from .processing_utils import BaseImageProcessor, BaseTextProcessor

__all__ = [
    "SamProcessor",
    "SamImageProcessor",
    "SamPromptProcessor",
]


class SamProcessor(ProcessorMixin):

    attributes = ["image_processor", "prompt_processor"]
    image_processor_class = "SamImageProcessor"
    prompt_processor_class = "SamPromptProcessor"

    def __init__(self, image_processor, prompt_processor):
        super().__init__(image_processor, prompt_processor)

        self.original_size = None
        self.input_size = None
        self.encode_size = self.image_processor.size

    def __call__(
        self,
        images,
        input_type,
        point_coords=None,
        point_labels=None,
        box=None,
        **kwargs,
    ):

        if images is None or input_type is None:
            raise ValueError("You have to specify either images and input_type.")

        if input_type == "boxs" and box is None:
            raise ValueError("You have to specify either box.")

        if input_type == "points" and point_coords is None:
            raise ValueError("You have to specify either point_coords.")

        image_pil_numpy = np.array(images)
        image_seg = self.image_processor(image_pil_numpy)
        self.original_size = self.image_processor.original_size
        self.input_size = self.image_processor.input_size
        prompt = self.prompt_processor(
            self.original_size,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
        )

        return image_seg, prompt

    def postprocess_masks(self, low_res_masks, mask_threshold: float = 0.0):

        masks = F.interpolate(
            paddle.to_tensor(low_res_masks),
            (self.encode_size, self.encode_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : self.input_size[0], : self.input_size[1]]
        masks = F.interpolate(masks, self.original_size, mode="bilinear", align_corners=False)
        masks = masks > mask_threshold

        return masks

    @property
    def model_input_names(self):
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(image_processor_input_names))


class SamPromptProcessor(BaseTextProcessor):
    r"""
    Constructs a Sam prompt processor.
    """

    def __init__(
        self,
        size: int = 1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = size

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = get_preprocess_shape(original_size[0], original_size[1], self.size)
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape([-1, 2, 2]), original_size)
        return boxes.reshape([-1, 4])

    def __call__(
        self,
        original_size,
        point_coords=None,
        point_labels=None,
        box=None,
        **kwargs,
    ):
        # coords_paddle, labels_paddle, box_paddle, mask_input_paddle = (
        #     None,
        #     None,
        #     None,
        #     None,
        # )
        coords_paddle, box_paddle = (
            None,
            None,
        )
        if point_coords is not None:
            point_coords = self.apply_coords(point_coords, original_size)
            coords_paddle = paddle.to_tensor(point_coords).cast("float32")
            coords_paddle = coords_paddle[None, :, :]

            return coords_paddle

        if box is not None:
            box = self.apply_boxes(box, original_size)
            box_paddle = paddle.to_tensor(box).cast("float32")
            box_paddle = box_paddle[None, :]
            return box_paddle


class SamImageProcessor(BaseImageProcessor):
    r"""
    Constructs a Sam image processor.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        size: List[int] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        image_format: str = "RGB",
        original_size: List[int] = None,
        input_size: List[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        size = size if size is not None else 1024

        self.size = size
        self.image_format = image_format

        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

        self.original_size = original_size
        self.input_size = input_size

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = get_preprocess_shape(image.shape[0], image.shape[1], self.size)

        return np.array(resize(to_pil_image(image), target_size))

    def preprocess(
        self,
        images,
        size: Optional[Dict[str, int]] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        image_format: str = "RGB",
        **kwargs,
    ):
        """
        Preprocess an image or batch of images.

        """

        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        size = size if size is not None else self.size

        if not isinstance(images, (list, tuple)):
            images = [images]

        # if isinstance(images[0], str):
        #     images = [load_image(image) for image in images]

        if not valid_images(images):
            raise ValueError("Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, " "paddle.Tensor.")

        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.image_format:
            images = images[..., ::-1]

        input_image = [self.apply_image(image) for image in images]

        input_image_paddle = paddle.to_tensor(input_image).cast("int32")

        input_image_paddle = input_image_paddle.transpose([0, 3, 1, 2])

        original_image_size = images[0].shape[:2]

        self.original_size = original_image_size
        self.input_size = tuple(input_image_paddle.shape[-2:])

        mean = paddle.to_tensor(self.image_mean).reshape([-1, 1, 1])
        std = paddle.to_tensor(self.image_std).reshape([-1, 1, 1])
        input_image_paddle = (input_image_paddle.astype(std.dtype) - mean) / std

        # Pad
        h, w = input_image_paddle.shape[-2:]
        padh = self.size - h
        padw = self.size - w
        input_image = F.pad(input_image_paddle, (0, padw, 0, padh))

        return input_image
