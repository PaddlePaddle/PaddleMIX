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

import math
import pathlib
from typing import BinaryIO, List, Optional, Tuple, Union

import paddle
from PIL import Image


def save_image(
    tensor: Union[paddle.Tensor, List[paddle.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    grid = make_grid(tensor, **kwargs)
    ndarr = (grid * 255 + 0.5).clip(0, 255).transpose([1, 2, 0]).astype("uint8").cpu().numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


def make_grid(
    tensor: Union[paddle.Tensor, List[paddle.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> paddle.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """
    if not isinstance(tensor, (paddle.Tensor, list)):
        raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    if isinstance(tensor, list):
        tensor = paddle.stack(tensor, axis=0)

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 3:
        if tensor.shape[0] == 1:
            tensor = paddle.concat([tensor] * 3, axis=0)
        tensor = tensor.unsqueeze(0)

    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = paddle.concat([tensor] * 3, axis=1)

    if normalize:
        tensor = tensor.clone()
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range must be a tuple (min, max)")

        def norm_ip(img, low, high):
            img = paddle.clip(img, low, high)
            img = (img - low) / max(high - low, 1e-5)
            return img

        def norm_range(t, range_):
            if range_ is not None:
                return norm_ip(t, range_[0], range_[1])
            else:
                return norm_ip(t, float(t.min()), float(t.max()))

        if scale_each:
            for i in range(tensor.shape[0]):
                tensor[i] = norm_range(tensor[i], value_range)
        else:
            tensor = norm_range(tensor, value_range)

    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(nmaps / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = paddle.full(
        (num_channels, height * ymaps + padding, width * xmaps + padding), pad_value, dtype=tensor.dtype
    )

    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            y_start = y * height + padding
            y_end = y_start + height - padding
            x_start = x * width + padding
            x_end = x_start + width - padding
            grid[:, y_start:y_end, x_start:x_end] = tensor[k]
            k += 1
    return grid
