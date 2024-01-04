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

from ..utils import (
    OptionalDependencyNotAvailable,
    is_einops_available,
    is_fastdeploy_available,
    is_paddle_available,
    is_paddlenlp_available,
)
from .animatediff import AnimateDiffPipeline
from .consistency_models import ConsistencyModelPipeline
from .controlnet import StableDiffusionControlNetPipeline
from .latent_consistency_models import (
    LatentConsistencyModelImg2ImgPipeline,
    LatentConsistencyModelPipeline,
)
from .lvdm import LVDMTextToVideoPipeline, LVDMUncondPipeline
from .pipeline_utils import DiffusionPipeline
from .stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput
from .stable_diffusion_xl import (
    StableDiffusionXLPipeline,
    StableDiffusionXLPipelineOutput,
)
from .stable_video_diffusion import (
    StableVideoDiffusionPipeline,
    StableVideoDiffusionPipelineOutput,
)

try:
    if not is_fastdeploy_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_fastdeploy_objects import *  # noqa F403
else:
    from .controlnet import FastDeployStableDiffusionControlNetPipeline
    from .fastdeploy_utils import (
        FastDeployDiffusionPipelineMixin,
        FastDeployRuntimeModel,
    )
    from .stable_diffusion import (
        FastDeployCycleDiffusionPipeline,
        FastDeployStableDiffusionImg2ImgPipeline,
        FastDeployStableDiffusionInpaintPipeline,
        FastDeployStableDiffusionInpaintPipelineLegacy,
        FastDeployStableDiffusionMegaPipeline,
        FastDeployStableDiffusionPipeline,
    )
    from .stable_diffusion_xl import (
        FastDeployStableDiffusionXLImg2ImgPipeline,
        FastDeployStableDiffusionXLInpaintPipeline,
        FastDeployStableDiffusionXLMegaPipeline,
        FastDeployStableDiffusionXLPipeline,
    )

try:
    if not (is_paddle_available() and is_paddlenlp_available() and is_einops_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_paddle_and_paddlenlp_and_einops_objects import *  # noqa F403
else:
    from .unidiffuser import (
        UniDiffuserModel,
        UniDiffuserPipeline,
        UniDiffuserTextDecoder,
    )
