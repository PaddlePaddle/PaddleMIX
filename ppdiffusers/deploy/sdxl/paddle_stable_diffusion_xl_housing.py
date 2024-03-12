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

from ppdiffusers.pipelines.paddleinfer_xl_utils import (
    PaddleInferDiffusionXLPipelineMixin,
    PaddleInferRuntimeModel,
)
from ppdiffusers.pipelines.pipeline_utils import DiffusionPipeline
from ppdiffusers.schedulers import KarrasDiffusionSchedulers
from ppdiffusers.transformers import CLIPTokenizer
from ppdiffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PaddleInferStableDiffusionXLPipelineHousing(DiffusionPipeline, PaddleInferDiffusionXLPipelineMixin):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving etc.)
    Args:
        vae_encoder ([`PaddleInferRuntimeModel`]):
            Variational Auto-Encoder (VAE) Model to encode images to latent representations.
        vae_decoder ([`PaddleInferRuntimeModel`]):
            Variational Auto-Encoder (VAE) Model to decode images from latent representations.
        text_encoder ([`PaddleInferRuntimeModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([` PaddleInferRuntimeModel`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`PaddleInferRuntimeModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """
    _optional_components = [
        "vae_encoder",
    ]

    def __init__(
        self,
        vae_encoder: PaddleInferRuntimeModel,
        vae_decoder: PaddleInferRuntimeModel,
        text_encoder: PaddleInferRuntimeModel,
        text_encoder_2: PaddleInferRuntimeModel,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: PaddleInferRuntimeModel,
        scheduler: KarrasDiffusionSchedulers,
        force_zeros_for_empty_prompt: bool = True,
    ):
        super().__init__()
        self.register_modules(
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
        )
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.post_init()
