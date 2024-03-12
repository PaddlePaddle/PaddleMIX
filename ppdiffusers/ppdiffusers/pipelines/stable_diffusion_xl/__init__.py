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
_additional_imports = {}
_import_structure = {"pipeline_output": ["StableDiffusionXLPipelineOutput"]}

try:
    if not (is_paddlenlp_available() and is_paddle_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_paddle_and_paddlenlp_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_paddle_and_paddlenlp_objects))
else:
    _import_structure["pipeline_stable_diffusion_xl"] = ["StableDiffusionXLPipeline"]
    _import_structure["pipeline_stable_diffusion_xl_img2img"] = ["StableDiffusionXLImg2ImgPipeline"]
    _import_structure["pipeline_stable_diffusion_xl_inpaint"] = ["StableDiffusionXLInpaintPipeline"]
    _import_structure["pipeline_stable_diffusion_xl_instruct_pix2pix"] = ["StableDiffusionXLInstructPix2PixPipeline"]
    # paddleinfer
    _import_structure["pipeline_paddleinfer_stable_diffusion_xl"] = ["PaddleInferStableDiffusionXLPipeline"]
    _import_structure["pipeline_paddleinfer_stable_diffusion_xl_img2img"] = [
        "PaddleInferStableDiffusionXLImg2ImgPipeline"
    ]
    _import_structure["pipeline_paddleinfer_stable_diffusion_xl_inpaint"] = [
        "PaddleInferStableDiffusionXLInpaintPipeline"
    ]
    _import_structure["pipeline_paddleinfer_stable_diffusion_xl_pix2pix"] = [
        "PaddleInferStableDiffusionXLPix2PixPipeline"
    ]
    _import_structure["pipeline_paddleinfer_stable_diffusion_xl_mega"] = ["PaddleInferStableDiffusionXLMegaPipeline"]
    _import_structure["pipeline_paddleinfer_stable_diffusion_xl_instruct_pix2pix"] = [
        "PaddleInferStableDiffusionXLInstructPix2PixPipeline"
    ]


# fastdeploy
try:
    if not (is_paddle_available() and is_paddlenlp_available() and is_fastdeploy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_fastdeploy_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_fastdeploy_objects))
else:
    _import_structure["pipeline_fastdeploy_stable_diffusion_xl"] = ["FastDeployStableDiffusionXLPipeline"]
    _import_structure["pipeline_fastdeploy_stable_diffusion_xl_img2img"] = [
        "FastDeployStableDiffusionXLImg2ImgPipeline"
    ]
    _import_structure["pipeline_fastdeploy_stable_diffusion_xl_inpaint"] = [
        "FastDeployStableDiffusionXLInpaintPipeline"
    ]
    _import_structure["pipeline_fastdeploy_stable_diffusion_xl_mega"] = ["FastDeployStableDiffusionXLMegaPipeline"]
    _import_structure["pipeline_fastdeploy_stable_diffusion_xl_instruct_pix2pix"] = [
        "FastDeployStableDiffusionXLInstructPix2PixPipeline"
    ]

if TYPE_CHECKING or PPDIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_paddlenlp_available() and is_paddle_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_paddle_and_paddlenlp_objects import *  # noqa F403
    else:
        from .pipeline_output import StableDiffusionXLPipelineOutput
        from .pipeline_paddleinfer_stable_diffusion_xl import (
            PaddleInferStableDiffusionXLPipeline,
        )
        from .pipeline_paddleinfer_stable_diffusion_xl_img2img import (
            PaddleInferStableDiffusionXLImg2ImgPipeline,
        )
        from .pipeline_paddleinfer_stable_diffusion_xl_inpaint import (
            PaddleInferStableDiffusionXLInpaintPipeline,
        )
        from .pipeline_paddleinfer_stable_diffusion_xl_instruct_pix2pix import (
            PaddleInferStableDiffusionXLInstructPix2PixPipeline,
        )
        from .pipeline_paddleinfer_stable_diffusion_xl_mega import (
            PaddleInferStableDiffusionXLMegaPipeline,
        )
        from .pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
        from .pipeline_stable_diffusion_xl_img2img import (
            StableDiffusionXLImg2ImgPipeline,
        )
        from .pipeline_stable_diffusion_xl_inpaint import (
            StableDiffusionXLInpaintPipeline,
        )
        from .pipeline_stable_diffusion_xl_instruct_pix2pix import (
            StableDiffusionXLInstructPix2PixPipeline,
        )

    try:
        if not (is_paddle_available() and is_paddlenlp_available() and is_fastdeploy_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_fastdeploy_objects import *
    else:
        from .pipeline_fastdeploy_stable_diffusion_xl import (
            FastDeployStableDiffusionXLPipeline,
        )
        from .pipeline_fastdeploy_stable_diffusion_xl_img2img import (
            FastDeployStableDiffusionXLImg2ImgPipeline,
        )
        from .pipeline_fastdeploy_stable_diffusion_xl_inpaint import (
            FastDeployStableDiffusionXLInpaintPipeline,
        )
        from .pipeline_fastdeploy_stable_diffusion_xl_instruct_pix2pix import (
            FastDeployStableDiffusionXLInstructPix2PixPipeline,
        )
        from .pipeline_fastdeploy_stable_diffusion_xl_mega import (
            FastDeployStableDiffusionXLMegaPipeline,
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
    for name, value in _additional_imports.items():
        setattr(sys.modules[__name__], name, value)
