# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Any, Callable, Dict, List, Optional, Union

import paddle
from fkd_class import FKD
from rewards import get_reward_function

import ppdiffusers
from ppdiffusers import DiffusionPipeline, StableDiffusionPipeline
from ppdiffusers.image_processor import PipelineImageInput
from ppdiffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    TextualInversionLoaderMixin,
)
from ppdiffusers.models import AutoencoderKL, UNet2DConditionModel
from ppdiffusers.pipelines.stable_diffusion.pipeline_output import (
    StableDiffusionPipelineOutput,
)
from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
    retrieve_timesteps,
)
from ppdiffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from ppdiffusers.schedulers import KarrasDiffusionSchedulers
from ppdiffusers.transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from ppdiffusers.utils import deprecate

logger = ppdiffusers.utils.logging.get_logger(__name__)


class FKDStableDiffusion(
    StableDiffusionPipeline,
    DiffusionPipeline,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
    FromSingleFileMixin,
):
    """
    Pipeline for text-to-image generation using Stable Diffusion.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection = None,
        requires_safety_checker: bool = True,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )

    @paddle.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        fkd_args: Optional[Dict[str, Any]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[paddle.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        """
        The call function to the pipeline for generation.
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = tuple(prompt_embeds.shape)[0]

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = paddle.concat(x=[negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None:
            image_embeds, negative_image_embeds = self.encode_image(ip_adapter_image, num_images_per_prompt)
            if self.do_classifier_free_guidance:
                image_embeds = paddle.concat([negative_image_embeds, image_embeds])

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = paddle.to_tensor(data=self.guidance_scale - 1).tile(
                repeat_times=batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).cast(dtype=latents.dtype)

        # reward_fn
        def postprocess_and_apply_reward_fn(x):
            imagesx = self.image_processor.postprocess(x, output_type=output_type)
            imagesx = [image for image in imagesx]
            rewards = get_reward_function(
                fkd_args["guidance_reward_fn"],
                images=imagesx,
                prompts=prompt,
                metric_to_chase=fkd_args.get("metric_to_chase", None),
            )
            return paddle.to_tensor(data=rewards)

        print("Args:", fkd_args)
        if fkd_args is not None and fkd_args["use_smc"]:
            fkd = FKD(
                latent_to_decode_fn=lambda x: latent_to_decode(model=self, output_type=output_type, latents=x),
                reward_fn=postprocess_and_apply_reward_fn,
                **fkd_args,
            )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self._interrupt:
                    continue
                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat(x=[latents] * 2) if self.do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(chunks=2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )

                step_dict = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)
                latents = step_dict["prev_sample"]
                x0_preds = step_dict["pred_original_sample"]

                if fkd_args is not None and fkd_args["use_smc"]:

                    latents, _ = fkd.resample(sampling_idx=i, latents=latents, x0_preds=x0_preds)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor,
                return_dict=False,
                generator=generator,
            )[0]
            has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None
        if has_nsfw_concept is None:
            do_denormalize = [True] * tuple(image.shape)[0]
        else:
            do_denormalize = [(not has_nsfw) for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        self.maybe_free_model_hooks()
        if not return_dict:
            return image, has_nsfw_concept
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


def latent_to_decode(*, model, output_type, latents):
    if not output_type == "latent":

        if latents.dtype != model.vae.dtype:
            model.vae = model.vae.to(latents.dtype)

        has_latents_mean = hasattr(model.vae.config, "latents_mean") and model.vae.config.latents_mean is not None
        has_latents_std = hasattr(model.vae.config, "latents_std") and model.vae.config.latents_std is not None

        if has_latents_mean and has_latents_std:
            latents_mean = (
                paddle.to_tensor(data=model.vae.config.latents_mean).reshape([1, 4, 1, 1]).cast(latents.dtype)
            )
            latents_std = paddle.to_tensor(data=model.vae.config.latents_std).reshape([1, 4, 1, 1]).cast(latents.dtype)
            latents = latents * latents_std / model.vae.config.scaling_factor + latents_mean
        else:
            latents = latents / model.vae.config.scaling_factor
        image = model.vae.decode(latents, return_dict=False)[0]
    else:
        image = latents
    return image
