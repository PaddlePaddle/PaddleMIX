# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from ppdiffusers.pipelines.paddleinfer_utils import (
    PaddleInferDiffusionPipelineMixin,
    PaddleInferRuntimeModel,
)
from ppdiffusers.pipelines.pipeline_utils import DiffusionPipeline
from ppdiffusers.schedulers import EulerDiscreteScheduler
from ppdiffusers.transformers import CLIPImageProcessor
from ppdiffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PaddleInferStableVideoDiffusionPipelineHousing(
    DiffusionPipeline,
    PaddleInferDiffusionPipelineMixin,
):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae_encoder ([`PaddleInferRuntimeModel`]):
            Variational Auto-Encoder (VAE) Model to encode images to latent representations.
        vae_decoder ([`PaddleInferRuntimeModel`]):
            Variational Auto-Encoder (VAE) Model to decode images from latent representations.
        image_encoder ([`PaddleInferRuntimeModel`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`PaddleInferRuntimeModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
    """

    def __init__(
        self,
        vae_encoder: PaddleInferRuntimeModel,
        vae_decoder: PaddleInferRuntimeModel,
        image_encoder: PaddleInferRuntimeModel,
        unet: PaddleInferRuntimeModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()
        self.register_modules(
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
