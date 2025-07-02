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

from typing import Dict, NamedTuple, Optional, Tuple, Union

import paddle
from PIL.Image import Image as pil_image

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
Image = Union[paddle.Tensor, pil_image]
BoundingBox = Tuple[float, float, float, float]
CropMethodType = Literal["none", "random", "center", "random-2d"]
SplitType = Literal["train", "validation", "test"]


class ImageDescription(NamedTuple):
    id: int
    file_name: str
    original_size: Tuple[int, int]
    url: Optional[str] = None
    license: Optional[int] = None
    coco_url: Optional[str] = None
    date_captured: Optional[str] = None
    flickr_url: Optional[str] = None
    flickr_id: Optional[str] = None
    coco_id: Optional[str] = None


class Category(NamedTuple):
    id: str
    super_category: Optional[str]
    name: str


class Annotation(NamedTuple):
    area: float
    image_id: str
    bbox: BoundingBox
    category_no: int
    category_id: str
    id: Optional[int] = None
    source: Optional[str] = None
    confidence: Optional[float] = None
    is_group_of: Optional[bool] = None
    is_truncated: Optional[bool] = None
    is_occluded: Optional[bool] = None
    is_depiction: Optional[bool] = None
    is_inside: Optional[bool] = None
    segmentation: Optional[Dict] = None
