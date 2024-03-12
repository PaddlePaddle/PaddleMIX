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
from typing import List, Optional, Union

import numpy as np
import PIL
from PIL import Image

from ...utils import (
    OptionalDependencyNotAvailable,
    is_paddle_available,
    is_paddlenlp_available,
)

try:
    if not (is_paddlenlp_available() and is_paddle_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_paddle_and_paddlenlp_objects import ShapEPipeline
else:
    from .blip_image_processing import BlipImageProcessor
    from .modeling_blip2 import Blip2QFormerModel
    from .modeling_ctx_clip import ContextCLIPTextModel
    from .pipeline_blip_diffusion import BlipDiffusionPipeline
