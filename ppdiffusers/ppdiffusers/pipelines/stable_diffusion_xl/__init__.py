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
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL

from ...utils import (
    BaseOutput,
    OptionalDependencyNotAvailable,
    is_paddle_available,
    is_paddlenlp_available,
)


@dataclass
class StableDiffusionXLPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


try:
    if not (is_paddlenlp_available() and is_paddle_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_paddle_and_paddlenlp_objects import *
else:
    from .pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
    from .pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline
    from .pipeline_stable_diffusion_xl_inpaint import StableDiffusionXLInpaintPipeline
    from .pipeline_stable_diffusion_xl_instruct_pix2pix import (
        StableDiffusionXLInstructPix2PixPipeline,
    )
    from .pipeline_fastdeploy_stable_diffusion_xl import FastDeployStableDiffusionXLPipeline
    from .pipeline_fastdeploy_stable_diffusion_xl_img2img import FastDeployStableDiffusionXLImg2ImgPipeline
    from .pipeline_fastdeploy_stable_diffusion_xl_inpaint import FastDeployStableDiffusionXLInpaintPipeline
    from .pipeline_fastdeploy_stable_diffusion_xl_mega import FastDeployStableDiffusionXLMegaPipeline