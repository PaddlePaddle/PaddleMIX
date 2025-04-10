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

from typing import Optional, Tuple, Union

import numpy as np
import paddle

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import logging
from ..utils.accelerate_utils import apply_forward_hook
from .activations import get_activation
from .downsampling import MochiDownsample3D
from .modeling_outputs import AutoencoderKLOutput
from .modeling_utils import ModelMixin
from .upsampling import CogVideoXUpsample3D
from .vae import DecoderOutput, DiagonalGaussianDistribution

logger = logging.get_logger(__name__)