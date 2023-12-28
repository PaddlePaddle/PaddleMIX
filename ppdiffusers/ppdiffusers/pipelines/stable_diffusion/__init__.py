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

from ...utils import (
    BaseOutput,
    OptionalDependencyNotAvailable,
    is_fastdeploy_available,
    is_paddle_available,
    is_paddlenlp_available,
)
from .pipeline_stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
)
from .safety_checker import StableDiffusionSafetyChecker

try:
    if not (is_paddlenlp_available() and is_paddle_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_paddle_and_paddlenlp_objects import *
else:
    from .pipeline_stable_diffusion import StableDiffusionPipeline  # noqa F401

try:
    if not (is_paddle_available() and is_fastdeploy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_fastdeploy_objects import *  # noqa F403
else:
    from .pipeline_fastdeploy_cycle_diffusion import FastDeployCycleDiffusionPipeline
    from .pipeline_fastdeploy_stable_diffusion import FastDeployStableDiffusionPipeline
    from .pipeline_fastdeploy_stable_diffusion_img2img import (
        FastDeployStableDiffusionImg2ImgPipeline,
    )
    from .pipeline_fastdeploy_stable_diffusion_inpaint import (
        FastDeployStableDiffusionInpaintPipeline,
    )
    from .pipeline_fastdeploy_stable_diffusion_inpaint_legacy import (
        FastDeployStableDiffusionInpaintPipelineLegacy,
    )
    from .pipeline_fastdeploy_stable_diffusion_mega import (
        FastDeployStableDiffusionMegaPipeline,
    )
