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
    is_fastdeploy_available,
    is_k_diffusion_available,
    is_k_diffusion_version,
    is_paddle_available,
    is_paddlenlp_available,
    is_paddlenlp_version,
)

_dummy_objects = {}
_additional_imports = {}
_import_structure = {"pipeline_output": ["StableDiffusionPipelineOutput"]}

try:
    if not (is_paddlenlp_available() and is_paddle_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_paddle_and_paddlenlp_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_paddle_and_paddlenlp_objects))
else:
    _import_structure["clip_image_project_model"] = ["CLIPImageProjection"]
    _import_structure["pipeline_cycle_diffusion"] = ["CycleDiffusionPipeline"]
    _import_structure["pipeline_stable_diffusion"] = ["StableDiffusionPipeline"]
    _import_structure["pipeline_stable_diffusion_attend_and_excite"] = ["StableDiffusionAttendAndExcitePipeline"]
    _import_structure["pipeline_stable_diffusion_gligen"] = ["StableDiffusionGLIGENPipeline"]
    _import_structure["pipeline_stable_diffusion_gligen"] = ["StableDiffusionGLIGENPipeline"]
    _import_structure["pipeline_stable_diffusion_gligen_text_image"] = ["StableDiffusionGLIGENTextImagePipeline"]
    _import_structure["pipeline_stable_diffusion_img2img"] = ["StableDiffusionImg2ImgPipeline"]
    _import_structure["pipeline_stable_diffusion_inpaint"] = ["StableDiffusionInpaintPipeline"]
    _import_structure["pipeline_stable_diffusion_inpaint_legacy"] = ["StableDiffusionInpaintPipelineLegacy"]
    _import_structure["pipeline_stable_diffusion_instruct_pix2pix"] = ["StableDiffusionInstructPix2PixPipeline"]
    _import_structure["pipeline_stable_diffusion_latent_upscale"] = ["StableDiffusionLatentUpscalePipeline"]
    _import_structure["pipeline_stable_diffusion_ldm3d"] = ["StableDiffusionLDM3DPipeline"]
    _import_structure["pipeline_stable_diffusion_model_editing"] = ["StableDiffusionModelEditingPipeline"]
    _import_structure["pipeline_stable_diffusion_panorama"] = ["StableDiffusionPanoramaPipeline"]
    _import_structure["pipeline_stable_diffusion_paradigms"] = ["StableDiffusionParadigmsPipeline"]
    _import_structure["pipeline_stable_diffusion_sag"] = ["StableDiffusionSAGPipeline"]
    _import_structure["pipeline_stable_diffusion_upscale"] = ["StableDiffusionUpscalePipeline"]
    _import_structure["pipeline_stable_unclip"] = ["StableUnCLIPPipeline"]
    _import_structure["pipeline_stable_unclip_img2img"] = ["StableUnCLIPImg2ImgPipeline"]
    _import_structure["safety_checker"] = ["StableDiffusionSafetyChecker"]
    _import_structure["stable_unclip_image_normalizer"] = ["StableUnCLIPImageNormalizer"]
    _import_structure["pipeline_paddleinfer_cycle_diffusion"] = ["PaddleInferCycleDiffusionPipeline"]
    _import_structure["pipeline_paddleinfer_stable_diffusion"] = ["PaddleInferStableDiffusionPipeline"]
    _import_structure["pipeline_paddleinfer_stable_diffusion_img2img"] = ["PaddleInferStableDiffusionImg2ImgPipeline"]
    _import_structure["pipeline_paddleinfer_stable_diffusion_inpaint"] = ["PaddleInferStableDiffusionInpaintPipeline"]
    _import_structure["pipeline_paddleinfer_stable_diffusion_inpaint_legacy"] = [
        "PaddleInferStableDiffusionInpaintPipelineLegacy"
    ]
    _import_structure["pipeline_paddleinfer_stable_diffusion_mega"] = ["PaddleInferStableDiffusionMegaPipeline"]

try:
    if not (is_paddlenlp_available() and is_paddle_available() and is_paddlenlp_version(">=", "2.6.0")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_paddle_and_paddlenlp_objects import (
        StableDiffusionImageVariationPipeline,
    )

    _dummy_objects.update({"StableDiffusionImageVariationPipeline": StableDiffusionImageVariationPipeline})
else:
    _import_structure["pipeline_stable_diffusion_image_variation"] = ["StableDiffusionImageVariationPipeline"]
try:
    if not (is_paddlenlp_available() and is_paddle_available() and is_paddlenlp_version(">=", "2.6.0")):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_paddle_and_paddlenlp_objects import (
        StableDiffusionDepth2ImgPipeline,
        StableDiffusionDiffEditPipeline,
        StableDiffusionPix2PixZeroPipeline,
    )

    _dummy_objects.update(
        {
            "StableDiffusionDepth2ImgPipeline": StableDiffusionDepth2ImgPipeline,
            "StableDiffusionDiffEditPipeline": StableDiffusionDiffEditPipeline,
            "StableDiffusionPix2PixZeroPipeline": StableDiffusionPix2PixZeroPipeline,
        }
    )
else:
    _import_structure["pipeline_stable_diffusion_depth2img"] = ["StableDiffusionDepth2ImgPipeline"]
    _import_structure["pipeline_stable_diffusion_diffedit"] = ["StableDiffusionDiffEditPipeline"]
    _import_structure["pipeline_stable_diffusion_pix2pix_zero"] = ["StableDiffusionPix2PixZeroPipeline"]
try:
    if not (
        is_paddle_available()
        and is_paddlenlp_available()
        and is_k_diffusion_available()
        and is_k_diffusion_version(">=", "0.0.12")
    ):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_paddle_and_paddlenlp_and_k_diffusion_objects

    _dummy_objects.update(get_objects_from_module(dummy_paddle_and_paddlenlp_and_k_diffusion_objects))
else:
    _import_structure["pipeline_stable_diffusion_k_diffusion"] = ["StableDiffusionKDiffusionPipeline"]
try:
    if not (is_paddlenlp_available() and is_fastdeploy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_fastdeploy_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_fastdeploy_objects))
else:
    _import_structure["pipeline_fastdeploy_stable_diffusion"] = ["FastDeployStableDiffusionPipeline"]
    _import_structure["pipeline_fastdeploy_stable_diffusion_img2img"] = ["FastDeployStableDiffusionImg2ImgPipeline"]
    _import_structure["pipeline_fastdeploy_stable_diffusion_inpaint"] = ["FastDeployStableDiffusionInpaintPipeline"]
    _import_structure["pipeline_fastdeploy_stable_diffusion_inpaint_legacy"] = [
        "FastDeployStableDiffusionInpaintPipelineLegacy"
    ]
    # new add
    _import_structure["pipeline_fastdeploy_stable_diffusion_mega"] = ["FastDeployStableDiffusionMegaPipeline"]
    _import_structure["pipeline_fastdeploy_cycle_diffusion"] = ["FastDeployCycleDiffusionPipeline"]
    _import_structure["pipeline_fastdeploy_stable_diffusion_upscale"] = ["FastDeployStableDiffusionUpscalePipeline"]


if TYPE_CHECKING or PPDIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_paddlenlp_available() and is_paddle_available()):
            raise OptionalDependencyNotAvailable()

    except OptionalDependencyNotAvailable:
        from ...utils.dummy_paddle_and_paddlenlp_objects import *

    else:
        from .clip_image_project_model import CLIPImageProjection
        from .pipeline_cycle_diffusion import CycleDiffusionPipeline
        from .pipeline_output import StableDiffusionPipelineOutput

        # paddleinfer
        from .pipeline_paddleinfer_cycle_diffusion import (
            PaddleInferCycleDiffusionPipeline,
        )
        from .pipeline_paddleinfer_stable_diffusion import (
            PaddleInferStableDiffusionPipeline,
        )
        from .pipeline_paddleinfer_stable_diffusion_img2img import (
            PaddleInferStableDiffusionImg2ImgPipeline,
        )
        from .pipeline_paddleinfer_stable_diffusion_inpaint import (
            PaddleInferStableDiffusionInpaintPipeline,
        )
        from .pipeline_paddleinfer_stable_diffusion_inpaint_legacy import (
            PaddleInferStableDiffusionInpaintPipelineLegacy,
        )
        from .pipeline_paddleinfer_stable_diffusion_mega import (
            PaddleInferStableDiffusionMegaPipeline,
        )
        from .pipeline_stable_diffusion import StableDiffusionPipeline
        from .pipeline_stable_diffusion_attend_and_excite import (
            StableDiffusionAttendAndExcitePipeline,
        )
        from .pipeline_stable_diffusion_gligen import StableDiffusionGLIGENPipeline
        from .pipeline_stable_diffusion_gligen_text_image import (
            StableDiffusionGLIGENTextImagePipeline,
        )
        from .pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
        from .pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
        from .pipeline_stable_diffusion_inpaint_legacy import (
            StableDiffusionInpaintPipelineLegacy,
        )
        from .pipeline_stable_diffusion_instruct_pix2pix import (
            StableDiffusionInstructPix2PixPipeline,
        )
        from .pipeline_stable_diffusion_latent_upscale import (
            StableDiffusionLatentUpscalePipeline,
        )
        from .pipeline_stable_diffusion_ldm3d import StableDiffusionLDM3DPipeline
        from .pipeline_stable_diffusion_model_editing import (
            StableDiffusionModelEditingPipeline,
        )
        from .pipeline_stable_diffusion_panorama import StableDiffusionPanoramaPipeline
        from .pipeline_stable_diffusion_paradigms import (
            StableDiffusionParadigmsPipeline,
        )
        from .pipeline_stable_diffusion_sag import StableDiffusionSAGPipeline
        from .pipeline_stable_diffusion_upscale import StableDiffusionUpscalePipeline
        from .pipeline_stable_unclip import StableUnCLIPPipeline
        from .pipeline_stable_unclip_img2img import StableUnCLIPImg2ImgPipeline
        from .safety_checker import StableDiffusionSafetyChecker
        from .stable_unclip_image_normalizer import StableUnCLIPImageNormalizer

    try:
        if not (is_paddlenlp_available() and is_paddle_available() and is_paddlenlp_version(">=", "2.6.0")):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_paddle_and_paddlenlp_objects import (
            StableDiffusionImageVariationPipeline,
        )
    else:
        from .pipeline_stable_diffusion_image_variation import (
            StableDiffusionImageVariationPipeline,
        )

    try:
        if not (is_paddlenlp_available() and is_paddle_available() and is_paddlenlp_version(">=", "2.6.0")):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_paddle_and_paddlenlp_objects import (
            StableDiffusionDepth2ImgPipeline,
            StableDiffusionDiffEditPipeline,
            StableDiffusionPix2PixZeroPipeline,
        )
    else:
        from .pipeline_stable_diffusion_depth2img import (
            StableDiffusionDepth2ImgPipeline,
        )
        from .pipeline_stable_diffusion_diffedit import StableDiffusionDiffEditPipeline
        from .pipeline_stable_diffusion_pix2pix_zero import (
            StableDiffusionPix2PixZeroPipeline,
        )

    try:
        if not (
            is_paddle_available()
            and is_paddlenlp_available()
            and is_k_diffusion_available()
            and is_k_diffusion_version(">=", "0.0.12")
        ):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_paddle_and_paddlenlp_and_k_diffusion_objects import *
    else:
        from .pipeline_stable_diffusion_k_diffusion import (
            StableDiffusionKDiffusionPipeline,
        )

    try:
        if not (is_paddlenlp_available() and is_fastdeploy_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_fastdeploy_objects import *
    else:
        from .pipeline_fastdeploy_cycle_diffusion import (
            FastDeployCycleDiffusionPipeline,
        )
        from .pipeline_fastdeploy_stable_diffusion import (
            FastDeployStableDiffusionPipeline,
        )
        from .pipeline_fastdeploy_stable_diffusion_img2img import (
            FastDeployStableDiffusionImg2ImgPipeline,
        )
        from .pipeline_fastdeploy_stable_diffusion_inpaint import (
            FastDeployStableDiffusionInpaintPipeline,
        )
        from .pipeline_fastdeploy_stable_diffusion_inpaint_legacy import (
            FastDeployStableDiffusionInpaintPipelineLegacy,
        )

        # new add
        from .pipeline_fastdeploy_stable_diffusion_mega import (
            FastDeployStableDiffusionMegaPipeline,
        )
        from .pipeline_fastdeploy_stable_diffusion_upscale import (
            FastDeployStableDiffusionUpscalePipeline,
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
