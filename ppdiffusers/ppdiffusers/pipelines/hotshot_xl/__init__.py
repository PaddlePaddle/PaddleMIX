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
from typing import Union

import numpy as np
import paddle

# don't remove these imports - they are needed to load from pretrain.
from ppdiffusers.models.hotshot_xl.unet import UNet3DConditionModel  # noqa: *
from ppdiffusers.models.modeling_utils import ModelMixin  # noqa: *
from ppdiffusers.utils import BaseOutput


@dataclass
class HotshotPipelineXLOutput(BaseOutput):
    videos: Union[paddle.Tensor, np.ndarray]
