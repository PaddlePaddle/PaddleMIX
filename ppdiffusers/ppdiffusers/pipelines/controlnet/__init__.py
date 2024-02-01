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

from typing import TYPE_CHECKING

from ...utils import (
    PPDIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_fastdeploy_available,
    is_paddle_available,
    is_paddlenlp_available,
)

_dummy_objects = {}
_import_structure = {}

try:
    if not (is_paddlenlp_available() and is_paddle_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_paddle_and_paddlenlp_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_paddle_and_paddlenlp_objects))
else:
    _import_structure["multicontrolnet"] = ["MultiControlNetModel"]
    _import_structure["pipeline_controlnet"] = ["StableDiffusionControlNetPipeline"]
    _import_structure["pipeline_controlnet_blip_diffusion"] = ["BlipDiffusionControlNetPipeline"]
    _import_structure["pipeline_controlnet_img2img"] = ["StableDiffusionControlNetImg2ImgPipeline"]
    _import_structure["pipeline_controlnet_inpaint"] = ["StableDiffusionControlNetInpaintPipeline"]
    _import_structure["pipeline_controlnet_inpaint_sd_xl"] = ["StableDiffusionXLControlNetInpaintPipeline"]
    _import_structure["pipeline_controlnet_sd_xl"] = ["StableDiffusionXLControlNetPipeline"]
    _import_structure["pipeline_controlnet_sd_xl_img2img"] = ["StableDiffusionXLControlNetImg2ImgPipeline"]
try:
    if not (is_paddle_available() and is_paddlenlp_available() and is_fastdeploy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_fastdeploy_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_fastdeploy_objects))
else:
    _import_structure["pipeline_fastdeploy_stable_diffusion_controlnet"] = [
        "FastDeployStableDiffusionControlNetPipeline"
    ]

_import_structure["pipeline_paddleinfer_stable_diffusion_controlnet"] = [
    "PaddleInferStableDiffusionControlNetPipeline",
]


if TYPE_CHECKING or PPDIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_paddlenlp_available() and is_paddle_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_paddle_and_paddlenlp_objects import *
    else:
        from .multicontrolnet import MultiControlNetModel
        from .pipeline_controlnet import StableDiffusionControlNetPipeline
        from .pipeline_controlnet_blip_diffusion import BlipDiffusionControlNetPipeline
        from .pipeline_controlnet_img2img import (
            StableDiffusionControlNetImg2ImgPipeline,
        )
        from .pipeline_controlnet_inpaint import (
            StableDiffusionControlNetInpaintPipeline,
        )
        from .pipeline_controlnet_inpaint_sd_xl import (
            StableDiffusionXLControlNetInpaintPipeline,
        )
        from .pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline
        from .pipeline_controlnet_sd_xl_img2img import (
            StableDiffusionXLControlNetImg2ImgPipeline,
        )

    try:
        if not (is_paddle_available() and is_paddlenlp_available() and is_fastdeploy_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_fastdeploy_objects import *  # noqa F403
    else:
        from .pipeline_fastdeploy_stable_diffusion_controlnet import (
            FastDeployStableDiffusionControlNetPipeline,
        )

    from .pipeline_paddleinfer_stable_diffusion_controlnet import (
        PaddleInferStableDiffusionControlNetPipeline,
    )

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
