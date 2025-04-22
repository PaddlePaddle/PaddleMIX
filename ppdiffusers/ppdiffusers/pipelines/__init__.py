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

from ..utils import (
    PPDIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_fastdeploy_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_note_seq_available,
    is_paddle_available,
    is_paddlenlp_available,
)

# These modules contain pipelines from multiple libraries/frameworks
_dummy_objects = {}
_import_structure = {
    "controlnet": [],
    "controlnet_sd3": [],
    "latent_diffusion": [],
    "stable_diffusion": [],
    "stable_diffusion_xl": [],
}

try:
    if not is_paddle_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_paddle_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_paddle_objects))
else:
    _import_structure["auto_pipeline"] = [
        "AutoPipelineForImage2Image",
        "AutoPipelineForInpainting",
        "AutoPipelineForText2Image",
    ]
    _import_structure["consistency_models"] = ["ConsistencyModelPipeline"]
    _import_structure["dance_diffusion"] = ["DanceDiffusionPipeline"]
    _import_structure["ddim"] = ["DDIMPipeline"]
    _import_structure["ddpm"] = ["DDPMPipeline"]
    _import_structure["dit"] = ["DiTPipeline"]
    _import_structure["latent_diffusion"].extend(["LDMSuperResolutionPipeline"])
    _import_structure["latent_diffusion"].extend(["LDMTextToImageUViTPipeline"])
    _import_structure["latent_diffusion"].extend(["LDMTextToImageLargeDiTPipeline"])
    _import_structure["latent_diffusion_uncond"] = ["LDMPipeline"]
    _import_structure["pipeline_utils"] = [
        "AudioPipelineOutput",
        "DiffusionPipeline",
        "ImagePipelineOutput",
    ]
    _import_structure["pndm"] = ["PNDMPipeline"]
    _import_structure["repaint"] = ["RePaintPipeline"]
    _import_structure["score_sde_ve"] = ["ScoreSdeVePipeline"]
    _import_structure["stochastic_karras_ve"] = ["KarrasVePipeline"]
    # new add
    _import_structure["img_to_video"] = ["ImgToVideoSDPipelineOutput", "ImgToVideoSDPipeline"]
    _import_structure["video_to_video"] = ["VideoToVideoModelscopePipelineOutput", "VideoToVideoModelscopePipeline"]

try:
    if not (is_paddle_available() and is_librosa_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_paddle_and_librosa_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_paddle_and_librosa_objects))
else:
    _import_structure["audio_diffusion"] = ["AudioDiffusionPipeline", "Mel"]
try:
    if not (is_paddle_available() and is_paddlenlp_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_paddle_and_paddlenlp_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_paddle_and_paddlenlp_objects))
else:
    _import_structure["alt_diffusion"] = [
        "AltDiffusionImg2ImgPipeline",
        "AltDiffusionPipeline",
    ]
    _import_structure["animatediff"] = ["AnimateDiffPipeline"]
    _import_structure["audioldm"] = ["AudioLDMPipeline"]
    _import_structure["audioldm2"] = [
        "AudioLDM2Pipeline",
        "AudioLDM2ProjectionModel",
        "AudioLDM2UNet2DConditionModel",
    ]
    _import_structure["blip_diffusion"] = ["BlipDiffusionPipeline"]
    _import_structure["cogvideo"] = [
        "CogVideoXPipeline",
        "CogVideoXVCtrlPipeline",
        "CogVideoXVCtrlImageToVideoPipeline",
    ]
    _import_structure["wan"] = [
        "WanPipeline",
        "WanImageToVideoPipeline",
    ]
    _import_structure["controlnet"].extend(
        [
            "BlipDiffusionControlNetPipeline",
            "StableDiffusionControlNetImg2ImgPipeline",
            "StableDiffusionControlNetInpaintPipeline",
            "StableDiffusionControlNetPipeline",
            "StableDiffusionXLControlNetImg2ImgPipeline",
            "StableDiffusionXLControlNetInpaintPipeline",
            "StableDiffusionXLControlNetPipeline",
            "PaddleInferStableDiffusionControlNetPipeline",
        ]
    )
    _import_structure["controlnet_sd3"].extend(
        [
            "StableDiffusion3ControlNetPipeline",
            "StableDiffusion3ControlNetInpaintingPipeline",
        ]
    )
    _import_structure["deepfloyd_if"] = [
        "IFImg2ImgPipeline",
        "IFImg2ImgSuperResolutionPipeline",
        "IFInpaintingPipeline",
        "IFInpaintingSuperResolutionPipeline",
        "IFPipeline",
        "IFSuperResolutionPipeline",
    ]
    _import_structure["flux"] = [
        "FluxControlPipeline",
        "FluxControlInpaintPipeline",
        "FluxControlImg2ImgPipeline",
        "FluxImg2ImgPipeline",
        "FluxInpaintPipeline",
        "FluxPipeline",
        "FluxFillPipeline",
        "FluxPriorReduxPipeline",
        "ReduxImageEncoder",
    ]
    _import_structure["kandinsky"] = [
        "KandinskyCombinedPipeline",
        "KandinskyImg2ImgCombinedPipeline",
        "KandinskyImg2ImgPipeline",
        "KandinskyInpaintCombinedPipeline",
        "KandinskyInpaintPipeline",
        "KandinskyPipeline",
        "KandinskyPriorPipeline",
    ]
    _import_structure["kandinsky2_2"] = [
        "KandinskyV22CombinedPipeline",
        "KandinskyV22ControlnetImg2ImgPipeline",
        "KandinskyV22ControlnetPipeline",
        "KandinskyV22Img2ImgCombinedPipeline",
        "KandinskyV22Img2ImgPipeline",
        "KandinskyV22InpaintCombinedPipeline",
        "KandinskyV22InpaintPipeline",
        "KandinskyV22Pipeline",
        "KandinskyV22PriorEmb2EmbPipeline",
        "KandinskyV22PriorPipeline",
    ]
    _import_structure["kandinsky3"] = [
        "Kandinsky3Img2ImgPipeline",
        "Kandinsky3Pipeline",
    ]
    _import_structure["latent_consistency_models"] = [
        "LatentConsistencyModelImg2ImgPipeline",
        "LatentConsistencyModelPipeline",
    ]
    _import_structure["latent_diffusion"].extend(["LDMTextToImagePipeline"])
    _import_structure["latent_diffusion"].extend(["LDMTextToImageUViTPipeline"])
    _import_structure["latent_diffusion"].extend(["LDMTextToImageLargeDiTPipeline"])
    _import_structure["musicldm"] = ["MusicLDMPipeline"]
    _import_structure["paint_by_example"] = ["PaintByExamplePipeline"]
    _import_structure["pixart_alpha"] = ["PixArtAlphaPipeline"]
    _import_structure["semantic_stable_diffusion"] = ["SemanticStableDiffusionPipeline"]
    _import_structure["shap_e"] = ["ShapEImg2ImgPipeline", "ShapEPipeline"]
    _import_structure["stable_diffusion"].extend(
        [
            "CLIPImageProjection",
            "CycleDiffusionPipeline",
            "StableDiffusionAttendAndExcitePipeline",
            "StableDiffusionDepth2ImgPipeline",
            "StableDiffusionDiffEditPipeline",
            "StableDiffusionGLIGENPipeline",
            "StableDiffusionGLIGENPipeline",
            "StableDiffusionGLIGENTextImagePipeline",
            "StableDiffusionImageVariationPipeline",
            "StableDiffusionImg2ImgPipeline",
            "StableDiffusionInpaintPipeline",
            "StableDiffusionInpaintPipelineLegacy",
            "StableDiffusionInstructPix2PixPipeline",
            "StableDiffusionLatentUpscalePipeline",
            "StableDiffusionLDM3DPipeline",
            "StableDiffusionModelEditingPipeline",
            "StableDiffusionPanoramaPipeline",
            "StableDiffusionParadigmsPipeline",
            "StableDiffusionPipeline",
            "StableDiffusionPix2PixZeroPipeline",
            "StableDiffusionSAGPipeline",
            "StableDiffusionUpscalePipeline",
            "StableUnCLIPImg2ImgPipeline",
            "StableUnCLIPPipeline",
            "StableDiffusionSafetyChecker",
            "PaddleInferCycleDiffusionPipeline",
            "PaddleInferStableDiffusionPipeline",
            "PaddleInferStableDiffusionImg2ImgPipeline",
            "PaddleInferStableDiffusionInpaintPipeline",
            "PaddleInferStableDiffusionInpaintPipelineLegacy",
            "PaddleInferStableDiffusionMegaPipeline",
        ]
    )
    _import_structure["stable_diffusion_3"] = ["StableDiffusion3Pipeline", "StableDiffusion3Img2ImgPipeline"]
    _import_structure["stable_diffusion_safe"] = ["StableDiffusionPipelineSafe"]
    _import_structure["stable_video_diffusion"] = [
        "StableVideoDiffusionPipeline",
        "PaddleInferStableVideoDiffusionPipeline",
    ]
    _import_structure["stable_diffusion_xl"].extend(
        [
            "StableDiffusionXLImg2ImgPipeline",
            "StableDiffusionXLInpaintPipeline",
            "StableDiffusionXLInstructPix2PixPipeline",
            "StableDiffusionXLPipeline",
            "PaddleInferStableDiffusionXLImg2ImgPipeline",
            "PaddleInferStableDiffusionXLInpaintPipeline",
            "PaddleInferStableDiffusionXLMegaPipeline",
            "PaddleInferStableDiffusionXLPipeline",
            "PaddleInferStableDiffusionXLInstructPix2PixPipeline",
        ]
    )
    _import_structure["t2i_adapter"] = [
        "StableDiffusionAdapterPipeline",
        "StableDiffusionXLAdapterPipeline",
    ]
    _import_structure["text_to_video_synthesis"] = [
        "TextToVideoSDPipeline",
        "TextToVideoZeroPipeline",
        "TextToVideoZeroSDXLPipeline",
        "VideoToVideoSDPipeline",
    ]
    _import_structure["unclip"] = ["UnCLIPImageVariationPipeline", "UnCLIPPipeline"]
    _import_structure["unidiffuser"] = [
        "ImageTextPipelineOutput",
        "UniDiffuserModel",
        "UniDiffuserPipeline",
        "UniDiffuserTextDecoder",
    ]
    _import_structure["versatile_diffusion"] = [
        "VersatileDiffusionDualGuidedPipeline",
        "VersatileDiffusionImageVariationPipeline",
        "VersatileDiffusionPipeline",
        "VersatileDiffusionTextToImagePipeline",
    ]
    _import_structure["vq_diffusion"] = ["VQDiffusionPipeline"]
    _import_structure["wuerstchen"] = [
        "WuerstchenCombinedPipeline",
        "WuerstchenDecoderPipeline",
        "WuerstchenPriorPipeline",
    ]
    _import_structure["lvdm"] = [
        "LVDMTextToVideoPipeline",
        "LVDMUncondPipeline",
        "VideoPipelineOutput",
    ]
    _import_structure["hunyuan_video"] = [
        "HunyuanVideoPipeline",
        "HunyuanVideoPipelineOutput",
    ]
try:
    if not is_fastdeploy_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_fastdeploy_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_fastdeploy_objects))
else:
    _import_structure["fastdeploy_utils"] = ["FastDeployRuntimeModel", "FastDeployDiffusionPipelineMixin"]
    _import_structure["fastdeployxl_utils"] = ["FastDeployDiffusionXLPipelineMixin"]

try:
    if not (is_paddle_available() and is_paddlenlp_available() and is_fastdeploy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_paddle_and_paddlenlp_and_fastdeploy_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_paddle_and_paddlenlp_and_fastdeploy_objects))
else:
    _import_structure["stable_diffusion"].extend(
        [
            "FastDeployStableDiffusionImg2ImgPipeline",
            "FastDeployStableDiffusionInpaintPipeline",
            "FastDeployStableDiffusionInpaintPipelineLegacy",
            "FastDeployStableDiffusionPipeline",
            "FastDeployStableDiffusionMegaPipeline",
            "FastDeployCycleDiffusionPipeline",
            "FastDeployStableDiffusionUpscalePipeline",
        ]
    )
    # new add
    _import_structure["stable_diffusion_xl"].extend(
        [
            "FastDeployStableDiffusionXLPipeline",
            "FastDeployStableDiffusionXLImg2ImgPipeline",
            "FastDeployStableDiffusionXLInpaintPipeline",
            "FastDeployStableDiffusionXLMegaPipeline",
            "FastDeployStableDiffusionXLInstructPix2PixPipeline",
        ]
    )

    # new add
    _import_structure["controlnet"].extend(
        [
            "FastDeployStableDiffusionControlNetPipeline",
        ]
    )
try:
    if not (is_paddle_available() and is_paddlenlp_available() and is_k_diffusion_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_paddle_and_paddlenlp_and_k_diffusion_objects

    _dummy_objects.update(get_objects_from_module(dummy_paddle_and_paddlenlp_and_k_diffusion_objects))
else:
    _import_structure["stable_diffusion"].extend(["StableDiffusionKDiffusionPipeline"])

try:
    if not (is_paddlenlp_available() and is_paddle_available() and is_note_seq_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils import dummy_paddle_and_paddlenlp_and_note_seq_objects  # noqa F403

    _dummy_objects.update(get_objects_from_module(dummy_paddle_and_paddlenlp_and_note_seq_objects))
else:
    _import_structure["spectrogram_diffusion"] = [
        "MidiProcessor",
        "SpectrogramDiffusionPipeline",
    ]

if TYPE_CHECKING or PPDIFFUSERS_SLOW_IMPORT:
    try:
        if not is_paddle_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_paddle_objects import *  # noqa F403

    else:
        from .auto_pipeline import (
            AutoPipelineForImage2Image,
            AutoPipelineForInpainting,
            AutoPipelineForText2Image,
        )
        from .consistency_models import ConsistencyModelPipeline
        from .dance_diffusion import DanceDiffusionPipeline
        from .ddim import DDIMPipeline
        from .ddpm import DDPMPipeline
        from .dit import DiTPipeline
        from .latent_diffusion import LDMSuperResolutionPipeline
        from .latent_diffusion_uncond import LDMPipeline
        from .pipeline_utils import (
            AudioPipelineOutput,
            DiffusionPipeline,
            ImagePipelineOutput,
        )
        from .pndm import PNDMPipeline
        from .repaint import RePaintPipeline
        from .score_sde_ve import ScoreSdeVePipeline
        from .stochastic_karras_ve import KarrasVePipeline

    try:
        if not (is_paddle_available() and is_librosa_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_paddle_and_librosa_objects import *
    else:
        from .audio_diffusion import AudioDiffusionPipeline, Mel

    try:
        if not (is_paddle_available() and is_paddlenlp_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ..utils.dummy_paddle_and_paddlenlp_objects import *
    else:
        from .alt_diffusion import AltDiffusionImg2ImgPipeline, AltDiffusionPipeline
        from .animatediff import AnimateDiffPipeline
        from .audioldm import AudioLDMPipeline
        from .audioldm2 import (
            AudioLDM2Pipeline,
            AudioLDM2ProjectionModel,
            AudioLDM2UNet2DConditionModel,
        )
        from .blip_diffusion import BlipDiffusionPipeline
        from .cogvideo import (
            CogVideoXPipeline,
            CogVideoXVCtrlImageToVideoPipeline,
            CogVideoXVCtrlPipeline,
        )
        from .controlnet import (
            BlipDiffusionControlNetPipeline,
            PaddleInferStableDiffusionControlNetPipeline,
            StableDiffusionControlNetImg2ImgPipeline,
            StableDiffusionControlNetInpaintPipeline,
            StableDiffusionControlNetPipeline,
            StableDiffusionXLControlNetImg2ImgPipeline,
            StableDiffusionXLControlNetInpaintPipeline,
            StableDiffusionXLControlNetPipeline,
        )
        from .controlnet_sd3 import (
            StableDiffusion3ControlNetInpaintingPipeline,
            StableDiffusion3ControlNetPipeline,
        )
        from .deepfloyd_if import (
            IFImg2ImgPipeline,
            IFImg2ImgSuperResolutionPipeline,
            IFInpaintingPipeline,
            IFInpaintingSuperResolutionPipeline,
            IFPipeline,
            IFSuperResolutionPipeline,
        )
        from .flux import (
            FluxControlImg2ImgPipeline,
            FluxControlInpaintPipeline,
            FluxControlPipeline,
            FluxFillPipeline,
            FluxImg2ImgPipeline,
            FluxInpaintPipeline,
            FluxPipeline,
            FluxPriorReduxPipeline,
            ReduxImageEncoder,
        )
        from .hunyuan_video import HunyuanVideoPipeline, HunyuanVideoPipelineOutput
        from .img_to_video import ImgToVideoSDPipeline, ImgToVideoSDPipelineOutput
        from .kandinsky import (
            KandinskyCombinedPipeline,
            KandinskyImg2ImgCombinedPipeline,
            KandinskyImg2ImgPipeline,
            KandinskyInpaintCombinedPipeline,
            KandinskyInpaintPipeline,
            KandinskyPipeline,
            KandinskyPriorPipeline,
        )
        from .kandinsky2_2 import (
            KandinskyV22CombinedPipeline,
            KandinskyV22ControlnetImg2ImgPipeline,
            KandinskyV22ControlnetPipeline,
            KandinskyV22Img2ImgCombinedPipeline,
            KandinskyV22Img2ImgPipeline,
            KandinskyV22InpaintCombinedPipeline,
            KandinskyV22InpaintPipeline,
            KandinskyV22Pipeline,
            KandinskyV22PriorEmb2EmbPipeline,
            KandinskyV22PriorPipeline,
        )
        from .kandinsky3 import Kandinsky3Img2ImgPipeline, Kandinsky3Pipeline
        from .latent_consistency_models import (
            LatentConsistencyModelImg2ImgPipeline,
            LatentConsistencyModelPipeline,
        )
        from .latent_diffusion import (
            LDMTextToImageLargeDiTPipeline,
            LDMTextToImagePipeline,
            LDMTextToImageUViTPipeline,
        )

        # new add
        from .lvdm import (
            LVDMTextToVideoPipeline,
            LVDMUncondPipeline,
            VideoPipelineOutput,
        )
        from .musicldm import MusicLDMPipeline
        from .paint_by_example import PaintByExamplePipeline
        from .pixart_alpha import PixArtAlphaPipeline
        from .semantic_stable_diffusion import SemanticStableDiffusionPipeline
        from .shap_e import ShapEImg2ImgPipeline, ShapEPipeline
        from .stable_diffusion import (
            CLIPImageProjection,
            CycleDiffusionPipeline,
            PaddleInferCycleDiffusionPipeline,
            PaddleInferStableDiffusionImg2ImgPipeline,
            PaddleInferStableDiffusionInpaintPipeline,
            PaddleInferStableDiffusionInpaintPipelineLegacy,
            PaddleInferStableDiffusionMegaPipeline,
            PaddleInferStableDiffusionPipeline,
            StableDiffusionAttendAndExcitePipeline,
            StableDiffusionDepth2ImgPipeline,
            StableDiffusionDiffEditPipeline,
            StableDiffusionGLIGENPipeline,
            StableDiffusionGLIGENTextImagePipeline,
            StableDiffusionImageVariationPipeline,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionInpaintPipeline,
            StableDiffusionInpaintPipelineLegacy,
            StableDiffusionInstructPix2PixPipeline,
            StableDiffusionLatentUpscalePipeline,
            StableDiffusionLDM3DPipeline,
            StableDiffusionModelEditingPipeline,
            StableDiffusionPanoramaPipeline,
            StableDiffusionParadigmsPipeline,
            StableDiffusionPipeline,
            StableDiffusionPix2PixZeroPipeline,
            StableDiffusionSafetyChecker,
            StableDiffusionSAGPipeline,
            StableDiffusionUpscalePipeline,
            StableUnCLIPImg2ImgPipeline,
            StableUnCLIPPipeline,
        )
        from .stable_diffusion_3 import (
            StableDiffusion3Img2ImgPipeline,
            StableDiffusion3Pipeline,
        )
        from .stable_diffusion_safe import StableDiffusionPipelineSafe
        from .stable_diffusion_xl import (
            PaddleInferStableDiffusionXLImg2ImgPipeline,
            PaddleInferStableDiffusionXLInpaintPipeline,
            PaddleInferStableDiffusionXLInstructPix2PixPipeline,
            PaddleInferStableDiffusionXLMegaPipeline,
            PaddleInferStableDiffusionXLPipeline,
            StableDiffusionXLImg2ImgPipeline,
            StableDiffusionXLInpaintPipeline,
            StableDiffusionXLInstructPix2PixPipeline,
            StableDiffusionXLPipeline,
        )
        from .stable_video_diffusion import (
            PaddleInferStableVideoDiffusionPipeline,
            StableVideoDiffusionPipeline,
            StableVideoDiffusionPipelineOutput,
        )
        from .t2i_adapter import (
            StableDiffusionAdapterPipeline,
            StableDiffusionXLAdapterPipeline,
        )
        from .text_to_video_synthesis import (
            TextToVideoSDPipeline,
            TextToVideoZeroPipeline,
            TextToVideoZeroSDXLPipeline,
            VideoToVideoSDPipeline,
        )
        from .unclip import UnCLIPImageVariationPipeline, UnCLIPPipeline
        from .unidiffuser import (
            ImageTextPipelineOutput,
            UniDiffuserModel,
            UniDiffuserPipeline,
            UniDiffuserTextDecoder,
        )
        from .versatile_diffusion import (
            VersatileDiffusionDualGuidedPipeline,
            VersatileDiffusionImageVariationPipeline,
            VersatileDiffusionPipeline,
            VersatileDiffusionTextToImagePipeline,
        )
        from .video_to_video import (
            VideoToVideoModelscopePipeline,
            VideoToVideoModelscopePipelineOutput,
        )
        from .vq_diffusion import VQDiffusionPipeline
        from .wan import WanImageToVideoPipeline, WanPipeline
        from .wuerstchen import (
            WuerstchenCombinedPipeline,
            WuerstchenDecoderPipeline,
            WuerstchenPriorPipeline,
        )

        try:
            if not is_fastdeploy_available():
                raise OptionalDependencyNotAvailable()
        except OptionalDependencyNotAvailable:
            from ..utils.dummy_fastdeploy_objects import *  # noqa F403

        else:
            from .fastdeploy_utils import (
                FastDeployDiffusionPipelineMixin,
                FastDeployRuntimeModel,
            )
            from .fastdeployxl_utils import FastDeployDiffusionXLPipelineMixin

        try:
            if not (is_paddle_available() and is_paddlenlp_available() and is_fastdeploy_available()):
                raise OptionalDependencyNotAvailable()
        except OptionalDependencyNotAvailable:
            from ..utils.dummy_paddle_and_paddlenlp_and_fastdeploy_objects import *
        else:
            # new add
            from .controlnet import FastDeployStableDiffusionControlNetPipeline
            from .stable_diffusion import (
                FastDeployCycleDiffusionPipeline,
                FastDeployStableDiffusionImg2ImgPipeline,
                FastDeployStableDiffusionInpaintPipeline,
                FastDeployStableDiffusionInpaintPipelineLegacy,
                FastDeployStableDiffusionMegaPipeline,
                FastDeployStableDiffusionPipeline,
                FastDeployStableDiffusionUpscalePipeline,
            )

            # xl
            from .stable_diffusion_xl import (
                FastDeployStableDiffusionXLImg2ImgPipeline,
                FastDeployStableDiffusionXLInpaintPipeline,
                FastDeployStableDiffusionXLInstructPix2PixPipeline,
                FastDeployStableDiffusionXLPipeline,
            )

        try:
            if not (is_paddle_available() and is_paddlenlp_available() and is_k_diffusion_available()):
                raise OptionalDependencyNotAvailable()
        except OptionalDependencyNotAvailable:
            from ..utils.dummy_paddle_and_paddlenlp_and_k_diffusion_objects import *
        else:
            from .stable_diffusion import StableDiffusionKDiffusionPipeline

        try:
            if not (is_paddlenlp_available() and is_paddle_available() and is_note_seq_available()):
                raise OptionalDependencyNotAvailable()
        except OptionalDependencyNotAvailable:
            from ..utils.dummy_paddle_and_paddlenlp_and_note_seq_objects import *  # noqa F403

        else:
            from .spectrogram_diffusion import (
                MidiProcessor,
                SpectrogramDiffusionPipeline,
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
