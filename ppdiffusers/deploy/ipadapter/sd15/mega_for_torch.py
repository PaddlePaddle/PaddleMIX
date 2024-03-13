# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import PIL.Image
import torch
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
)
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import logging

from ppdiffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class StableDiffusionMegaPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular xxxx, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], [`PNDMScheduler`], [`EulerDiscreteScheduler`], [`EulerAncestralDiscreteScheduler`]
            or [`DPMSolverMultistepScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]

    def __call__(self, *args, **kwargs):
        return self.text2img(*args, **kwargs)

    def text2img(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):

        expected_components = inspect.signature(StableDiffusionPipeline.__init__).parameters.keys()
        components = {name: component for name, component in self.components.items() if name in expected_components}
        temp_pipeline = StableDiffusionPipeline(
            **components, requires_safety_checker=self.config.requires_safety_checker
        )
        output = temp_pipeline(
            prompt=prompt,
            height=height,
            width=width,
            ip_adapter_image=ip_adapter_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            cross_attention_kwargs=cross_attention_kwargs,
        )
        return output

    def img2img(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.Tensor, PIL.Image.Image] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        expected_components = inspect.signature(StableDiffusionImg2ImgPipeline.__init__).parameters.keys()
        components = {name: component for name, component in self.components.items() if name in expected_components}
        temp_pipeline = StableDiffusionImg2ImgPipeline(
            **components, requires_safety_checker=self.config.requires_safety_checker
        )
        output = temp_pipeline(
            prompt=prompt,
            image=image,
            ip_adapter_image=ip_adapter_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds=prompt_embeds,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs,
        )

        return output

    def inpaint_legacy(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.Tensor, PIL.Image.Image] = None,
        mask_image: Union[torch.Tensor, PIL.Image.Image] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        add_predicted_noise: Optional[bool] = False,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        expected_components = inspect.signature(StableDiffusionInpaintPipeline.__init__).parameters.keys()
        components = {name: component for name, component in self.components.items() if name in expected_components}
        temp_pipeline = StableDiffusionInpaintPipeline(
            **components, requires_safety_checker=self.config.requires_safety_checker
        )
        output = temp_pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            ip_adapter_image=ip_adapter_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            add_predicted_noise=add_predicted_noise,
            eta=eta,
            generator=generator,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
        )

        return output

    def change_scheduler(self, scheduler_type="ddim"):
        self.orginal_scheduler_config = self.scheduler.config
        scheduler_type = scheduler_type.lower()
        if scheduler_type == "pndm":
            scheduler = PNDMScheduler.from_config(self.orginal_scheduler_config, skip_prk_steps=True)
        elif scheduler_type == "lms":
            scheduler = LMSDiscreteScheduler.from_config(self.orginal_scheduler_config)
        elif scheduler_type == "heun":
            scheduler = HeunDiscreteScheduler.from_config(self.orginal_scheduler_config)
        elif scheduler_type == "euler":
            scheduler = EulerDiscreteScheduler.from_config(self.orginal_scheduler_config)
        elif scheduler_type == "euler-ancestral":
            scheduler = EulerAncestralDiscreteScheduler.from_config(self.orginal_scheduler_config)
        elif scheduler_type == "dpm-multi":
            scheduler = DPMSolverMultistepScheduler.from_config(self.orginal_scheduler_config)
        elif scheduler_type == "dpm-single":
            scheduler = DPMSolverSinglestepScheduler.from_config(self.orginal_scheduler_config)
        elif scheduler_type == "kdpm2-ancestral":
            scheduler = KDPM2AncestralDiscreteScheduler.from_config(self.orginal_scheduler_config)
        elif scheduler_type == "kdpm2":
            scheduler = KDPM2DiscreteScheduler.from_config(self.orginal_scheduler_config)
        elif scheduler_type == "unipc-multi":
            scheduler = UniPCMultistepScheduler.from_config(self.orginal_scheduler_config)
        elif scheduler_type == "ddim":
            scheduler = DDIMScheduler.from_config(
                self.orginal_scheduler_config,
                steps_offset=1,
                clip_sample=False,
                set_alpha_to_one=False,
            )
        elif scheduler_type == "ddpm":
            scheduler = DDPMScheduler.from_config(
                self.orginal_scheduler_config,
            )
        elif scheduler_type == "deis-multi":
            scheduler = DEISMultistepScheduler.from_config(
                self.orginal_scheduler_config,
            )
        else:
            raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")
        return scheduler
