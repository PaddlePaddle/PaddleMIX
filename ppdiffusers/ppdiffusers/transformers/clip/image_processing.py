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
from typing import Dict, Optional, Union

import numpy as np
from paddlenlp.transformers.clip.image_processing import ChannelDimension
from paddlenlp.transformers.clip.image_processing import (
    CLIPImageProcessor as PPNLPCLIPImageProcessor,
)
from paddlenlp.transformers.clip.image_processing import (
    PILImageResampling,
    get_resize_output_image_size,
    get_size_dict,
    resize,
)

__all__ = ["CLIPImageProcessor"]


class CLIPImageProcessor(PPNLPCLIPImageProcessor):
    def resize(
        self,
        image: np.ndarray,
        size: Dict[str, int],
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Resize an image. The shortest edge of the image is resized to size["shortest_edge"], with the longest edge
        resized to keep the input aspect ratio.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                Resampling filter to use when resiizing the image.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the image. If not provided, it will be the same as the input image.
        """
        size = get_size_dict(size, default_to_square=False)
        if "shortest_edge" in size:
            # raise ValueError(f"The `size` parameter must contain the key `shortest_edge`. Got {size.keys()}")
            output_size = get_resize_output_image_size(image, size=size["shortest_edge"], default_to_square=False)
            return resize(image, size=output_size, resample=resample, data_format=data_format, **kwargs)
        elif "height" in size and "width" in size:
            size = (size["height"], size["width"])
            output_size = get_resize_output_image_size(image, size=size, default_to_square=True)
            return resize(image, size=output_size, resample=resample, data_format=data_format, **kwargs)
        else:
            raise ValueError(
                f"The `size` parameter must contain the key `shortest_edge` or `height` and `width`. Got {size.keys()}"
            )
