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

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import paddle

from ...utils import BaseOutput


@dataclass
class TextToVideoSDPipelineOutput(BaseOutput):
    """
    Output class for text-to-video pipelines.

    Args:
        frames (`List[np.ndarray]` or `paddle.Tensor`)
            List of denoised frames (essentially images) as NumPy arrays of shape `(height, width, num_channels)` or as
            a `paddle` tensor. The length of the list denotes the video length (the number of frames).
    """

    frames: Union[List[np.ndarray], paddle.Tensor]
