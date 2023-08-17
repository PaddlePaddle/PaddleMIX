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
import paddle

from paddlemix.models.imagebind.utils.kaldi import *
from paddlemix.models.imagebind.utils.paddle_aux import *
from paddlemix.models.imagebind.utils.resample import *


class finfo:
    bits: int
    min: float
    max: float
    eps: float
    tiny: float
    smallest_normal: float
    resolution: float
    dtype: str

    def __init__(self, dtype=None) -> None:
        ...


setattr(paddle, "finfo", finfo)
