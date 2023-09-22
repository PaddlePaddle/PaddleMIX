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
from copy import deepcopy
from typing import Callable, List, Optional, Union

import numpy as np
import paddle
import PIL
from PIL import Image

from ...models import UNet2DConditionModel, VQModel
from ...schedulers import DDPMScheduler
from ...utils import logging, randn_tensor, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from ppdiffusers import KandinskyV22InpaintPipeline, KandinskyV22PriorPipeline
        >>> from ppdiffusers.utils import load_image
        >>> import paddle
        >>> import numpy as np

        >>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-prior", paddle_dtype=paddle.float16
        ... )

        >>> prompt = "a hat"
        >>> image_emb, zero_image_emb = pipe_prior(prompt, return_dict=False)

        >>> pipe = KandinskyV22InpaintPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-decoder-inpaint", paddle_dtype=paddle.float16
        ... )

        >>> init_image = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/cat.png"
        ... )

        >>> mask = np.zeros((768, 768), dtype=np.float32)
        >>> mask[:250, 250:-250] = 1

        >>> out = pipe(
        ...     image=init_image,
        ...     mask_image=mask,
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=zero_image_emb,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=50,
        ... )

        >>> image = out.images[0]
        >>> image.save("cat_with_hat.png")
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


# Copied from ppdiffusers.pipelines.kandinsky.pipeline_kandinsky_inpaint.prepare_mask
def prepare_mask(masks):
    prepared_masks = []
    for mask in masks:
        old_mask = deepcopy(mask)
        for i in range(mask.shape[1]):
            for j in range(mask.shape[2]):
                if old_mask[0][i][j] == 1:
                    continue
                if i != 0:
                    mask[:, (i - 1), (j)] = 0
                if j != 0:
                    mask[:, (i), (j - 1)] = 0
                if i != 0 and j != 0:
                    mask[:, (i - 1), (j - 1)] = 0
                if i != mask.shape[1] - 1:
                    mask[:, (i + 1), (j)] = 0
                if j != mask.shape[2] - 1:
                    mask[:, (i), (j + 1)] = 0
                if i != mask.shape[1] - 1 and j != mask.shape[2] - 1:
                    mask[:, (i + 1), (j + 1)] = 0
        prepared_masks.append(mask)
    return paddle.stack(x=prepared_masks, axis=0)


def prepare_mask_and_masked_image(image, mask, height, width):
    """
    Prepares a pair (mask, image) to be consumed by the Kandinsky inpaint pipeline. This means that those inputs will
    be converted to ``paddle.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for
    the ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``paddle.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``paddle.float32`` too.

    Args:
        image (Union[np.array, PIL.Image, paddle.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``paddle.Tensor`` or a ``batch x channels x height x width`` ``paddle.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``paddle.Tensor`` or a ``batch x 1 x height x width`` ``paddle.Tensor``.
        height (`int`, *optional*, defaults to 512):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to 512):
            The width in pixels of the generated image.


    Raises:
        ValueError: ``paddle.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``paddle.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``paddle.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        tuple[paddle.Tensor]: The pair (mask, image) as ``paddle.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """
    if image is None:
        raise ValueError("`image` input cannot be undefined.")
    if mask is None:
        raise ValueError("`mask_image` input cannot be undefined.")
    if isinstance(image, paddle.Tensor):
        if not isinstance(mask, paddle.Tensor):
            raise TypeError(f"`image` is a paddle.Tensor but `mask` (type: {type(mask)} is not")
        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(axis=0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(axis=0).unsqueeze(axis=0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(axis=0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(axis=1)
        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.cast(dtype="float32")
    elif isinstance(mask, paddle.Tensor):
        raise TypeError(f"`mask` is a paddle.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=Image.BICUBIC, reducing_gap=1) for i in image]
            image = [np.array(i.convert("RGB"))[(None), :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[(None), :] for i in image], axis=0)
        image = image.transpose(0, 3, 1, 2)
        image = paddle.to_tensor(data=image).cast(dtype="float32") / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]
        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
            mask = np.concatenate([np.array(m.convert("L"))[(None), (None), :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[(None), (None), :] for m in mask], axis=0)
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = paddle.to_tensor(data=mask)
    mask = 1 - mask
    return mask, image


class KandinskyV22InpaintPipeline(DiffusionPipeline):
    """
    Pipeline for text-guided image inpainting using Kandinsky2.1

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

    def __init__(self, unet: UNet2DConditionModel, scheduler: DDPMScheduler, movq: VQModel):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, movq=movq)
        self.movq_scale_factor = 2 ** (len(self.movq.config.block_out_channels) - 1)
        self._warn_has_been_called = False

    # Copied from ppdiffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents
    def prepare_latents(self, shape, dtype, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
        latents = latents * scheduler.init_noise_sigma
        return latents

    @paddle.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image_embeds: Union[paddle.Tensor, List[paddle.Tensor]],
        image: Union[paddle.Tensor, PIL.Image.Image],
        mask_image: Union[paddle.Tensor, PIL.Image.Image, np.ndarray],
        negative_image_embeds: Union[paddle.Tensor, List[paddle.Tensor]],
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
            image_embeds (`paddle.Tensor `List[paddle.Tensor]`):
                The clip image embeddings for text prompt, that will be used to condition the image generation.
            image (`PIL.Image.Image`):
                `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
                be masked out with `mask_image` and repainted according to `prompt`.
            mask_image (`np.array`):
                Tensor representing an image batch, to mask `image`. White pixels in the mask will be repainted, while
                black pixels will be preserved. If `mask_image` is a PIL image, it will be converted to a single
                channel (luminance) before use. If it's a tensor, it should contain one color channel (L) instead of 3,
                so the expected shape would be `(B, H, W, 1)`.
            negative_image_embeds (`paddle.Tensor` or `List[paddle.Tensor]`):
                The clip image embeddings for negative text prompt, will be used to condition the image generation.
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
                One or a list of paddle generator(s).
                to make generation deterministic.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between: `"pil"` (`PIL.Image.Image`), `"np"`
                (`np.array`) or `"pt"` (`paddle.Tensor`).
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
            image_embeds = paddle.concat(x=image_embeds, axis=0)
        batch_size = image_embeds.shape[0] * num_images_per_prompt
        if isinstance(negative_image_embeds, list):
            negative_image_embeds = paddle.concat(x=negative_image_embeds, axis=0)
        if do_classifier_free_guidance:
            image_embeds = image_embeds.repeat_interleave(repeats=num_images_per_prompt, axis=0)
            negative_image_embeds = negative_image_embeds.repeat_interleave(repeats=num_images_per_prompt, axis=0)
            negative_image_embeds = negative_image_embeds.cast(self.unet.dtype)
            image_embeds = image_embeds.cast(self.unet.dtype)
            image_embeds = paddle.concat(x=[negative_image_embeds, image_embeds], axis=0).cast(dtype=self.unet.dtype)
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps
        mask_image, image = prepare_mask_and_masked_image(image, mask_image, height, width)

        # preprocess image and mask
        image = image.cast(dtype=image_embeds.dtype)
        image = self.movq.encode(image)["latents"]
        mask_image = mask_image.cast(dtype=image_embeds.dtype)
        image_shape = tuple(image.shape[-2:])
        mask_image = paddle.nn.functional.interpolate(x=mask_image, size=image_shape, mode="nearest")
        mask_image = prepare_mask(mask_image)
        masked_image = image * mask_image
        mask_image = mask_image.repeat_interleave(repeats=num_images_per_prompt, axis=0)
        masked_image = masked_image.repeat_interleave(repeats=num_images_per_prompt, axis=0)
        if do_classifier_free_guidance:
            mask_image = mask_image.tile(repeat_times=[2, 1, 1, 1])
            masked_image = masked_image.tile(repeat_times=[2, 1, 1, 1])
        num_channels_latents = self.movq.config.latent_channels
        height, width = downscale_height_and_width(height, width, self.movq_scale_factor)

        # create initial latent
        latents = self.prepare_latents(
            (batch_size, num_channels_latents, height, width), image_embeds.dtype, generator, latents, self.scheduler
        )
        noise = paddle.clone(x=latents)
        for i, t in enumerate(self.progress_bar(timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = paddle.concat(x=[latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = paddle.concat(x=[latent_model_input, masked_image, mask_image], axis=1)
            added_cond_kwargs = {"image_embeds": image_embeds}
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred, variance_pred = noise_pred.split(noise_pred.shape[1] // latents.shape[1], axis=1)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(chunks=2)
                _, variance_pred_text = variance_pred.chunk(chunks=2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                noise_pred = paddle.concat(x=[noise_pred, variance_pred_text], axis=1)
            if not (
                hasattr(self.scheduler.config, "variance_type")
                and self.scheduler.config.variance_type in ["learned", "learned_range"]
            ):
                noise_pred, _ = noise_pred.split(noise_pred.shape[1] // latents.shape[1], axis=1)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, generator=generator)[0]
            init_latents_proper = image[:1]
            init_mask = mask_image[:1]
            if i < len(timesteps_tensor) - 1:
                noise_timestep = timesteps_tensor[i + 1]
                init_latents_proper = self.scheduler.add_noise(
                    init_latents_proper, noise, paddle.to_tensor(data=[noise_timestep])
                )
            latents = init_mask * init_latents_proper + (1 - init_mask) * latents
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # post-processing
        latents = mask_image[:1] * image[:1] + (1 - mask_image[:1]) * latents
        image = self.movq.decode(latents, force_not_quantize=True)["sample"]

        if output_type not in ["pd", "np", "pil"]:
            raise ValueError(f"Only the output types `pt`, `pil` and `np` are supported not output_type={output_type}")
        if output_type in ["np", "pil"]:
            image = image * 0.5 + 0.5
            image = image.clip(min=0, max=1)
            image = image.cpu().transpose(perm=[0, 2, 3, 1]).astype(dtype="float32").numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)
