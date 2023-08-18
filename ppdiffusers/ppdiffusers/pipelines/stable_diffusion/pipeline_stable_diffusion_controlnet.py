# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from ...utils import deprecate
from ..controlnet.multicontrolnet import MultiControlNetModel  # noqa: F401
from ..controlnet.pipeline_controlnet import (  # noqa: F401
    StableDiffusionControlNetPipeline,
)

deprecate(
    "stable diffusion controlnet",
    "0.22.0",
    "Importing `StableDiffusionControlNetPipeline` or `MultiControlNetModel` from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet is deprecated. Please import `from diffusers import StableDiffusionControlNetPipeline` instead.",
    standard_warn=False,
    stacklevel=3,
)
