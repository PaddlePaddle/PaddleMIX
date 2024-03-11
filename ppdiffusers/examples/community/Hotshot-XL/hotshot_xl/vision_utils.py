# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import sys

import numpy as np
import paddle
from PIL import Image


def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.

    See :class:`~ToPILImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """

    if isinstance(pic, paddle.Tensor):
        if pic.ndim == 3:
            pic = pic.transpose((1, 2, 0))
        pic = pic.numpy(force=True)
    elif not isinstance(pic, np.ndarray):
        raise TypeError(f"pic should be Tensor or ndarray. Got {type(pic)}.")

    if pic.ndim == 2:
        # if 2D image, add channel dimension (HWC)
        pic = np.expand_dims(pic, 2)
    if pic.ndim != 3:
        raise ValueError(f"pic should be 2/3 dimensional. Got {pic.ndim} dimensions.")

    if pic.shape[-1] > 4:
        raise ValueError(f"pic should not have > 4 channels. Got {pic.shape[-1]} channels.")

    npimg = pic

    if np.issubdtype(npimg.dtype, np.floating) and mode != "F":
        npimg = (npimg * 255).astype(np.uint8)

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = "L"
        elif npimg.dtype == np.int16:
            expected_mode = "I;16" if sys.byteorder == "little" else "I;16B"
        elif npimg.dtype == np.int32:
            expected_mode = "I"
        elif npimg.dtype == np.float32:
            expected_mode = "F"
        if mode is not None and mode != expected_mode:
            raise ValueError(f"Incorrect mode ({mode}) supplied for input type {np.dtype}. Should be {expected_mode}")
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ["LA"]
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError(f"Only modes {permitted_2_channel_modes} are supported for 2D inputs")

        if mode is None and npimg.dtype == np.uint8:
            mode = "LA"

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ["RGBA", "CMYK", "RGBX"]
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError(f"Only modes {permitted_4_channel_modes} are supported for 4D inputs")

        if mode is None and npimg.dtype == np.uint8:
            mode = "RGBA"
    else:
        permitted_3_channel_modes = ["RGB", "YCbCr", "HSV"]
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError(f"Only modes {permitted_3_channel_modes} are supported for 3D inputs")
        if mode is None and npimg.dtype == np.uint8:
            mode = "RGB"

    if mode is None:
        raise TypeError(f"Input type {npimg.dtype} is not supported")

    return Image.fromarray(npimg, mode=mode)


class ToPILImage:
    """Convert a tensor or an ndarray to PIL Image

    Converts a paddle.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while adjusting the value range depending on the ``mode``.

    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:

            - If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
            - If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
            - If the input has 2 channels, the ``mode`` is assumed to be ``LA``.
            - If the input has 1 channel, the ``mode`` is determined by the data type (i.e ``int``, ``float``, ``short``).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes
    """

    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.

        Returns:
            PIL Image: Image converted to PIL Image.

        """
        return to_pil_image(pic, self.mode)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        if self.mode is not None:
            format_string += f"mode={self.mode}"
        format_string += ")"
        return format_string
