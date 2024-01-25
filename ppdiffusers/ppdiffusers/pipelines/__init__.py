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

from ..utils import (
    OptionalDependencyNotAvailable,
    is_einops_available,
    is_fastdeploy_available,
    is_k_diffusion_available,
    is_librosa_available,
    is_note_seq_available,
    is_paddle_available,
    is_paddlenlp_available,
)

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
        TextPipelineOutput,
    )
    from .pndm import PNDMPipeline
    from .repaint import RePaintPipeline
    from .score_sde_ve import ScoreSdeVePipeline
    from .stochastic_karras_ve import KarrasVePipeline

try:
    if not (is_paddle_available() and is_librosa_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_paddle_and_librosa_objects import *  # noqa F403
else:
    from .audio_diffusion import AudioDiffusionPipeline, Mel

try:
    if not (is_paddle_available() and is_paddlenlp_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_paddle_and_paddlenlp_objects import *  # noqa F403
else:
    from .alt_diffusion import AltDiffusionImg2ImgPipeline, AltDiffusionPipeline
    from .audioldm import AudioLDMPipeline
    from .controlnet import (
        StableDiffusionControlNetImg2ImgPipeline,
        StableDiffusionControlNetInpaintPipeline,
        StableDiffusionControlNetPipeline,
        StableDiffusionXLControlNetPipeline,
    )
    from .deepfloyd_if import (
        IFImg2ImgPipeline,
        IFImg2ImgSuperResolutionPipeline,
        IFInpaintingPipeline,
        IFInpaintingSuperResolutionPipeline,
        IFPipeline,
        IFSuperResolutionPipeline,
    )
    from .img_to_video import ImgToVideoSDPipeline
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
    from .latent_diffusion import LDMTextToImagePipeline
    from .lvdm import LVDMTextToVideoPipeline, LVDMUncondPipeline
    from .paint_by_example import PaintByExamplePipeline
    from .semantic_stable_diffusion import SemanticStableDiffusionPipeline
    from .shap_e import ShapEImg2ImgPipeline, ShapEPipeline
    from .stable_diffusion import (
        CycleDiffusionPipeline,
        StableDiffusionAttendAndExcitePipeline,
        StableDiffusionDepth2ImgPipeline,
        StableDiffusionDiffEditPipeline,
        StableDiffusionImageVariationPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInpaintPipelineLegacy,
        StableDiffusionInstructPix2PixPipeline,
        StableDiffusionLatentUpscalePipeline,
        StableDiffusionLDM3DPipeline,
        StableDiffusionMegaPipeline,
        StableDiffusionModelEditingPipeline,
        StableDiffusionPanoramaPipeline,
        StableDiffusionParadigmsPipeline,
        StableDiffusionPipeline,
        StableDiffusionPipelineAllinOne,
        StableDiffusionPix2PixZeroPipeline,
        StableDiffusionSAGPipeline,
        StableDiffusionUpscalePipeline,
        StableUnCLIPImg2ImgPipeline,
        StableUnCLIPPipeline,
    )
    from .stable_diffusion_safe import StableDiffusionPipelineSafe
    from .stable_diffusion_xl import (
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLInstructPix2PixPipeline,
        StableDiffusionXLPipeline,
    )
    from .t2i_adapter import StableDiffusionAdapterPipeline
    from .text_to_video_synthesis import (
        TextToVideoSDPipeline,
        TextToVideoZeroPipeline,
        VideoToVideoSDPipeline,
    )
    from .unclip import UnCLIPImageVariationPipeline, UnCLIPPipeline
    from .versatile_diffusion import (
        VersatileDiffusionDualGuidedPipeline,
        VersatileDiffusionImageVariationPipeline,
        VersatileDiffusionPipeline,
        VersatileDiffusionTextToImagePipeline,
    )
    from .video_to_video import VideoToVideoModelscopePipeline
    from .vq_diffusion import VQDiffusionPipeline

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

try:
    if not (is_paddle_available() and is_paddlenlp_available() and is_fastdeploy_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_paddle_and_paddlenlp_and_fastdeploy_objects import *  # noqa F403
else:
    from .controlnet import FastDeployStableDiffusionControlNetPipeline
    from .stable_diffusion import (
        FastDeployCycleDiffusionPipeline,
        FastDeployStableDiffusionImageVariationPipeline,
        FastDeployStableDiffusionImg2ImgPipeline,
        FastDeployStableDiffusionInpaintPipeline,
        FastDeployStableDiffusionInpaintPipelineLegacy,
        FastDeployStableDiffusionMegaPipeline,
        FastDeployStableDiffusionPipeline,
        FastDeployStableDiffusionUpscalePipeline,
    )
    from .stable_diffusion_xl import (
        FastDeployStableDiffusionXLPipeline,
        FastDeployStableDiffusionXLImg2ImgPipeline,
        FastDeployStableDiffusionXLInpaintPipeline,
        FastDeployStableDiffusionXLMegaPipeline,
    )

try:
    if not (is_paddle_available() and is_paddlenlp_available() and is_k_diffusion_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_paddle_and_paddlenlp_and_k_diffusion_objects import *  # noqa F403
else:
    from .stable_diffusion import StableDiffusionKDiffusionPipeline

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

try:
    if not (is_paddle_available() and is_paddlenlp_available() and is_note_seq_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ..utils.dummy_paddle_and_paddlenlp_and_note_seq_objects import *  # noqa F403
else:
    from .spectrogram_diffusion import MidiProcessor, SpectrogramDiffusionPipeline
