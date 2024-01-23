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

from typing import TYPE_CHECKING

from ...utils import (
    PPDIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_paddle_available,
    is_paddlenlp_available,
)

_dummy_objects = {}
_import_structure = {
    "timesteps": [
        "fast27_timesteps",
        "smart100_timesteps",
        "smart185_timesteps",
        "smart27_timesteps",
        "smart50_timesteps",
        "super100_timesteps",
        "super27_timesteps",
        "super40_timesteps",
    ]
}

try:
    if not (is_paddlenlp_available() and is_paddle_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_paddle_and_paddlenlp_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_paddle_and_paddlenlp_objects))
else:
    _import_structure["pipeline_if"] = ["IFPipeline"]
    _import_structure["pipeline_if_img2img"] = ["IFImg2ImgPipeline"]
    _import_structure["pipeline_if_img2img_superresolution"] = ["IFImg2ImgSuperResolutionPipeline"]
    _import_structure["pipeline_if_inpainting"] = ["IFInpaintingPipeline"]
    _import_structure["pipeline_if_inpainting_superresolution"] = ["IFInpaintingSuperResolutionPipeline"]
    _import_structure["pipeline_if_superresolution"] = ["IFSuperResolutionPipeline"]
    _import_structure["pipeline_output"] = ["IFPipelineOutput"]
    _import_structure["safety_checker"] = ["IFSafetyChecker"]
    _import_structure["watermark"] = ["IFWatermarker"]


if TYPE_CHECKING or PPDIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_paddlenlp_available() and is_paddle_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_paddle_and_paddlenlp_objects import *
    else:
        from .pipeline_if import IFPipeline
        from .pipeline_if_img2img import IFImg2ImgPipeline
        from .pipeline_if_img2img_superresolution import (
            IFImg2ImgSuperResolutionPipeline,
        )
        from .pipeline_if_inpainting import IFInpaintingPipeline
        from .pipeline_if_inpainting_superresolution import (
            IFInpaintingSuperResolutionPipeline,
        )
        from .pipeline_if_superresolution import IFSuperResolutionPipeline
        from .pipeline_output import IFPipelineOutput
        from .safety_checker import IFSafetyChecker
        from .timesteps import (
            fast27_timesteps,
            smart27_timesteps,
            smart50_timesteps,
            smart100_timesteps,
            smart185_timesteps,
            super27_timesteps,
            super40_timesteps,
            super100_timesteps,
        )
        from .watermark import IFWatermarker

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
