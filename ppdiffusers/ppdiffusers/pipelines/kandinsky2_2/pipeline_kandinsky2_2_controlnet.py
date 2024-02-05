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

from typing import Callable, List, Optional, Union

import paddle

from ...models import UNet2DConditionModel, VQModel
from ...schedulers import DDPMScheduler
from ...utils import logging
from ...utils.paddle_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import paddle
        >>> import numpy as np

        >>> from ppdiffusers import KandinskyV22PriorPipeline, KandinskyV22ControlnetPipeline
        >>> from ppdiffusers.transformers import pipeline
        >>> from ppdiffusers.utils import load_image


        >>> def make_hint(image, depth_estimator):
        ...     image = depth_estimator(image)["depth"]
        ...     image = np.array(image)
        ...     image = image[:, :, None]
        ...     image = np.concatenate([image, image, image], axis=2)
        ...     detected_map = paddle.to_tensor(image).cast("float32") / 255.0
        ...     hint = detected_map.permute(2, 0, 1)
        ...     return hint


        >>> depth_estimator = pipeline("depth-estimation")

        >>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-prior", paddle_dtype=paddle.float16
        ... )

        >>> pipe = KandinskyV22ControlnetPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-controlnet-depth", paddle_dtype=paddle.float16
        ... )


        >>> img = load_image(
        ...     "https://hf-mirror.com/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/cat.png"
        ... ).resize((768, 768))

        >>> hint = make_hint(img, depth_estimator).unsqueeze(0).cast("float16")

        >>> prompt = "A robot, 4k photo"
        >>> negative_prior_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

        >>> generator = paddle.Generator().manual_seed(43)

        >>> image_emb, zero_image_emb = pipe_prior(
        ...     prompt=prompt, negative_prompt=negative_prior_prompt, generator=generator
        ... ).to_tuple()

        >>> images = pipe(
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=zero_image_emb,
        ...     hint=hint,
        ...     num_inference_steps=50,
        ...     generator=generator,
        ...     height=768,
        ...     width=768,
        ... ).images

        >>> images[0].save("robot_cat.png")
        ```
"""


# Copied from ppdiffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2.downscale_height_and_width
def downscale_height_and_width(height, width, scale_factor=8):
    new_height = height // scale_factor**2
    if height % scale_factor**2 != 0:
        new_height += 1
    new_width = width // scale_factor**2
    if width % scale_factor**2 != 0:
        new_width += 1
    return new_height * scale_factor, new_width * scale_factor


class KandinskyV22ControlnetPipeline(DiffusionPipeline):
    """
    Pipeline for text-to-image generation using Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `unet` to generate image latents.
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        movq ([`VQModel`]):
            MoVQ Decoder to generate the image from the latents.
    """

    model_cpu_offload_seq = "unet->movq"

    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        movq: VQModel,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
        )
        self.movq_scale_factor = 2 ** (len(self.movq.config.block_out_channels) - 1)

    # Copied from ppdiffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents
    def prepare_latents(self, shape, dtype, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)
        else:
            if latents.shape != list(shape):
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {list(shape)}")
            latents = latents.cast(dtype)

        latents = latents * scheduler.init_noise_sigma
        return latents

    @paddle.no_grad()
    def __call__(
        self,
        image_embeds: Union[paddle.Tensor, List[paddle.Tensor]],
        negative_image_embeds: Union[paddle.Tensor, List[paddle.Tensor]],
        hint: paddle.Tensor,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: int = 1,
        return_dict: bool = True,
    ):
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            hint (`paddle.Tensor`):
                The controlnet condition.
            image_embeds (`paddle.Tensor` or `List[paddle.Tensor]`):
                The clip image embeddings for text prompt, that will be used to condition the image generation.
            negative_image_embeds (`paddle.Tensor` or `List[paddle.Tensor]`):
                The clip image embeddings for negative text prompt, will be used to condition the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`paddle.Generator` or `List[paddle.Generator]`, *optional*):
                One or a list of [paddle generator(s)] to make generation deterministic.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pd"` (`paddle.Tensor`).
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: paddle.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`
        """

        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(image_embeds, list):
            image_embeds = paddle.concat(image_embeds, axis=0)
        if isinstance(negative_image_embeds, list):
            negative_image_embeds = paddle.concat(negative_image_embeds, axis=0)
        if isinstance(hint, list):
            hint = paddle.concat(hint, axis=0)

        batch_size = image_embeds.shape[0] * num_images_per_prompt

        if do_classifier_free_guidance:
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, axis=0)
            negative_image_embeds = negative_image_embeds.repeat_interleave(num_images_per_prompt, axis=0)
            hint = hint.repeat_interleave(num_images_per_prompt, axis=0)

            image_embeds = paddle.concat([negative_image_embeds, image_embeds], axis=0).cast(dtype=self.unet.dtype)
            hint = paddle.concat([hint, hint], axis=0).cast(dtype=self.unet.dtype)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps

        num_channels_latents = self.movq.config.latent_channels

        height, width = downscale_height_and_width(height, width, self.movq_scale_factor)

        # create initial latent
        latents = self.prepare_latents(
            [batch_size, num_channels_latents, height, width],
            image_embeds.dtype,
            generator,
            latents,
            self.scheduler,
        )

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents

            added_cond_kwargs = {"image_embeds": image_embeds, "hint": hint}
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred, variance_pred = noise_pred.split(
                    [latents.shape[1], noise_pred.shape[1] - latents.shape[1]], axis=1
                )
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                _, variance_pred_text = variance_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = paddle.concat([noise_pred, variance_pred_text], axis=1)

            if not (
                hasattr(self.scheduler.config, "variance_type")
                and self.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred, _ = noise_pred.split([latents.shape[1], noise_pred.shape[1] - latents.shape[1]], axis=1)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
                generator=generator,
            )[0]

            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(self.scheduler, "order", 1)
                callback(step_idx, t, latents)
        # post-processing
        image = self.movq.decode(latents, force_not_quantize=True)["sample"]

        # Offload all models

        if output_type not in ["pd", "np", "pil"]:
            raise ValueError(f"Only the output types `pd`, `pil` and `np` are supported not output_type={output_type}")

        if output_type in ["np", "pil"]:
            image = image * 0.5 + 0.5
            image = image.clip(0, 1)
            image = image.transpose([0, 2, 3, 1]).cast("float32").cpu().numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
