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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import paddle
from fkd_class import FKD
from rewards import get_reward_function

from ppdiffusers import DiffusionPipeline, StableDiffusionXLPipeline
from ppdiffusers.image_processor import PipelineImageInput
from ppdiffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from ppdiffusers.models import AutoencoderKL, UNet2DConditionModel
from ppdiffusers.pipelines.stable_diffusion_xl.pipeline_output import (
    StableDiffusionXLPipelineOutput,
)
from ppdiffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    rescale_noise_cfg,
    retrieve_timesteps,
)
from ppdiffusers.schedulers import KarrasDiffusionSchedulers
from ppdiffusers.transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from ppdiffusers.utils import deprecate


class FKDStableDiffusionXL(
    StableDiffusionXLPipeline,
    DiffusionPipeline,
    FromSingleFileMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
):
    """
    Pipeline for text-to-image generation using Stable Diffusion XL.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt,
            add_watermarker=add_watermarker,
        )

    @paddle.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        fkd_args=None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        pooled_prompt_embeds: Optional[paddle.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[paddle.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[paddle.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        """
        Function invoked when calling the pipeline for generation.
        """
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = tuple(prompt_embeds.shape)[0]

        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, timesteps)

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

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = paddle.concat(x=[negative_prompt_embeds, prompt_embeds], axis=0)
            add_text_embeds = paddle.concat(x=[negative_pooled_prompt_embeds, add_text_embeds], axis=0)
            add_time_ids = paddle.concat(x=[negative_add_time_ids, add_time_ids], axis=0)

        add_time_ids = add_time_ids.tile(repeat_times=[batch_size * num_images_per_prompt, 1])

        if ip_adapter_image is not None:
            image_embeds, negative_image_embeds = self.encode_image(ip_adapter_image, num_images_per_prompt)
            if self.do_classifier_free_guidance:
                image_embeds = paddle.concat([negative_image_embeds, image_embeds])

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        if num_warmup_steps < 0:
            num_warmup_steps = 0

        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - self.denoising_end * self.scheduler.config.num_train_timesteps
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = paddle.to_tensor([self.guidance_scale - 1]).tile(
                repeat_times=batch_size * num_images_per_prompt
            )
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).cast(dtype=latents.dtype)

        self._num_timesteps = len(timesteps)

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

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                latent_model_input = paddle.concat(x=[latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                }
                if ip_adapter_image is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
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
                latents_dtype = latents.dtype
                step_dict = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)

                latents = step_dict["prev_sample"]
                x0_preds = step_dict["pred_original_sample"]

                if fkd_args is not None and fkd_args["use_smc"]:
                    latents, current_pop_images = fkd.resample(sampling_idx=i, latents=latents, x0_preds=x0_preds)

                if latents.dtype != latents_dtype:
                    latents = latents.cast(latents_dtype)
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)
                if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            needs_upcasting = self.vae.dtype in [paddle.float16, "float16"] and self.vae.config.force_upcast
            if needs_upcasting:
                self.upcast_vae()
                latents = latents.cast(next(iter(self.vae.post_quant_conv.parameters())).dtype)

            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    paddle.to_tensor(data=self.vae.config.latents_mean).reshape([1, 4, 1, 1]).cast(latents.dtype)
                )
                latents_std = (
                    paddle.to_tensor(data=self.vae.config.latents_std).reshape([1, 4, 1, 1]).cast(latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            if needs_upcasting:
                self.vae.to(dtype=paddle.float16)
        else:
            image = latents
        if not output_type == "latent":
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)
            image = self.image_processor.postprocess(image, output_type=output_type)
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


def latent_to_decode(*, model, output_type, latents):
    if not output_type == "latent":
        needs_upcasting = model.vae.dtype in [paddle.float16, "float16"] and model.vae.config.force_upcast
        if needs_upcasting:
            model.upcast_vae()
            latents = latents.cast(next(iter(model.vae.post_quant_conv.parameters())).dtype)
        elif latents.dtype != model.vae.dtype:
            model.vae = model.vae.cast(latents.dtype)

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
        if needs_upcasting:
            model.vae.to(dtype="float16")
    else:
        image = latents
    return image
