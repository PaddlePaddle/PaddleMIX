from ppdiffusers import PixArtAlphaPipeline
from .tgate_utils import register_forward,tgate_scheduler
import html
import inspect
import re
import urllib.parse as ul
from typing import Callable, List, Optional, Tuple, Union
from ppdiffusers.pipelines.pipeline_utils import ImagePipelineOutput
import paddle
import paddle.nn.functional as F
from ppdiffusers.utils import (
    BACKENDS_MAPPING,
    deprecate,
    is_bs4_available,
    is_ftfy_available,
    logging,
    replace_example_docstring,
)
from ppdiffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
    ASPECT_RATIO_1024_BIN,
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_256_BIN,
    EXAMPLE_DOC_STRING,
    retrieve_timesteps
    )

from types import MethodType


@paddle.no_grad()
@replace_example_docstring(EXAMPLE_DOC_STRING)
def tgate(
    self,
    prompt: Union[str, List[str]] = None,
    negative_prompt: str = "",
    num_inference_steps: int = 20,
    timesteps: List[int] = None,
    sigmas: List[float] = None,
    guidance_scale: float = 4.5,
    num_images_per_prompt: Optional[int] = 1,
    height: Optional[int] = None,
    width: Optional[int] = None,
    eta: float = 0.0,
    generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
    latents: Optional[paddle.Tensor] = None,
    prompt_embeds: Optional[paddle.Tensor] = None,
    prompt_attention_mask: Optional[paddle.Tensor] = None,
    negative_prompt_embeds: Optional[paddle.Tensor] = None,
    negative_prompt_attention_mask: Optional[paddle.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
    callback_steps: int = 1,
    clean_caption: bool = True,
    use_resolution_binning: bool = True,
    max_sequence_length: int = 120,
    gate_step: int = 10,
    sp_interval: int = 5,
    fi_interval: int = 1,
    warm_up: int = 2,
    lcm: bool = False,
    **kwargs,
) -> Union[ImagePipelineOutput, Tuple]:
    """
    Function invoked when calling the pipeline for generation.

    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        num_inference_steps (`int`, *optional*, defaults to 100):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        timesteps (`List[int]`, *optional*):
            Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
            in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
            passed will be used. Must be in descending order.
        sigmas (`List[float]`, *optional*):
            Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
            their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
            will be used.
        guidance_scale (`float`, *optional*, defaults to 4.5):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        height (`int`, *optional*, defaults to self.unet.config.sample_size):
            The height in pixels of the generated image.
        width (`int`, *optional*, defaults to self.unet.config.sample_size):
            The width in pixels of the generated image.
        eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
            [`schedulers.DDIMScheduler`], will be ignored for others.
        generator (`paddle.Generator` or `List[paddle.Generator]`, *optional*):
            One or a list of [paddle generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`paddle.Tensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`paddle.Tensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        prompt_attention_mask (`paddle.Tensor`, *optional*): Pre-generated attention mask for text embeddings.
        negative_prompt_embeds (`paddle.Tensor`, *optional*):
            Pre-generated negative text embeddings. For PixArt-Alpha this negative prompt should be "". If not
            provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
        negative_prompt_attention_mask (`paddle.Tensor`, *optional*):
            Pre-generated attention mask for negative text embeddings.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
        callback (`Callable`, *optional*):
            A function that will be called every `callback_steps` steps during inference. The function will be
            called with the following arguments: `callback(step: int, timestep: int, latents: paddle.Tensor)`.
        callback_steps (`int`, *optional*, defaults to 1):
            The frequency at which the `callback` function will be called. If not specified, the callback will be
            called at every step.
        clean_caption (`bool`, *optional*, defaults to `True`):
            Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
            be installed. If the dependencies are not installed, the embeddings will be created from the raw
            prompt.
        use_resolution_binning (`bool` defaults to `True`):
            If set to `True`, the requested height and width are first mapped to the closest resolutions using
            `ASPECT_RATIO_1024_BIN`. After the produced latents are decoded into images, they are resized back to
            the requested resolution. Useful for generating non-square images.
        max_sequence_length (`int` defaults to 120): Maximum sequence length to use with the `prompt`.
        gate_step (`int` defaults to 10): The time step to stop calculating the cross attention.
        sp_interval (`int` defaults to 5): The time-step interval to cache self attention before gate_step (Semantics-Planning Phase).
        fi_interval (`int` defaults to 1): The time-step interval to cache self attention after gate_step (Fidelity-Improving Phase).
        warm_up (`int` defaults to 2): The time step to warm up the model inference.
        lcm (`bool` defaults to False): If the latent consistency model is used as the transformer.
    Examples:

    Returns:
        [`~pipelines.ImagePipelineOutput`] or `tuple`:
            If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
            returned where the first element is a list with the generated images
    """
    if "mask_feature" in kwargs:
        deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
        deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)
    # 1. Check inputs. Raise error if not correct
    height = height or self.transformer.config.sample_size * self.vae_scale_factor
    width = width or self.transformer.config.sample_size * self.vae_scale_factor
    if use_resolution_binning:
        if self.transformer.config.sample_size == 128:
            aspect_ratio_bin = ASPECT_RATIO_1024_BIN
        elif self.transformer.config.sample_size == 64:
            aspect_ratio_bin = ASPECT_RATIO_512_BIN
        elif self.transformer.config.sample_size == 32:
            aspect_ratio_bin = ASPECT_RATIO_256_BIN
        else:
            raise ValueError("Invalid sample size")
        orig_height, orig_width = height, width
        height, width = self.image_processor.classify_height_width_bin(height, width, ratios=aspect_ratio_bin)

    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt,
        callback_steps,
        prompt_embeds,
        negative_prompt_embeds,
        prompt_attention_mask,
        negative_prompt_attention_mask,
    )

    # 2. Default height and width to transformer
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = self.encode_prompt(
        prompt,
        do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        clean_caption=clean_caption,
        max_sequence_length=max_sequence_length,
    )
    if do_classifier_free_guidance:
        cfg_prompt_embeds = paddle.concat([negative_prompt_embeds, prompt_embeds], axis=0)
        cfg_prompt_attention_mask = paddle.concat([negative_prompt_attention_mask, prompt_attention_mask], axis=0)

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler, num_inference_steps, timesteps, sigmas
    )

    # 5. Prepare latents.
    latent_channels = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        latent_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

    # 6.1 Prepare micro-conditions.
    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
    negative_cond_kwargs = {"resolution": None, "aspect_ratio": None}
    if self.transformer.config.sample_size == 128:
        resolution = paddle.to_tensor([height, width]).tile([batch_size * num_images_per_prompt, 1])
        aspect_ratio = paddle.to_tensor([float(height / width)]).tile([batch_size * num_images_per_prompt, 1])
        resolution = resolution.to(dtype=prompt_embeds.dtype, device=device)
        aspect_ratio = aspect_ratio.to(dtype=prompt_embeds.dtype, device=device)

        negative_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}
        if do_classifier_free_guidance:
            resolution = paddle.concat([resolution, resolution], axis=0)
            aspect_ratio = paddle.concat([aspect_ratio, aspect_ratio], axis=0)

        added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

    # 7. Denoising loop
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    register_forward(self.transformer, 
        'Attention',
        ca_kward = {
            'cache': False,
            'reuse': False,
        },
        sa_kward = {
            'cache': False,
            'reuse': False,
        },
        keep_shape=True
        )
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if do_classifier_free_guidance and i < gate_step:
                latent_model_input = paddle.concat([latents] * 2)
                prompt_embeds = cfg_prompt_embeds
                prompt_attention_mask = cfg_prompt_attention_mask
            else:
                latent_model_input = latents
                prompt_embeds = negative_prompt_embeds if do_classifier_free_guidance else prompt_embeds
                prompt_attention_mask = negative_prompt_attention_mask if do_classifier_free_guidance else prompt_attention_mask
                added_cond_kwargs = negative_cond_kwargs

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            current_timestep = t
            if not paddle.is_tensor(current_timestep):
                if isinstance(current_timestep, float):
                    dtype = paddle.float64
                else:
                    dtype = paddle.int64
                current_timestep = paddle.to_tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
            elif len(current_timestep.shape) == 0:
                current_timestep = current_timestep[None]
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            current_timestep = current_timestep.expand(latent_model_input.shape[0])

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                ca_kwards,sa_kwards,keep_shape=tgate_scheduler(
                    cur_step=i-num_warmup_steps, 
                    gate_step=gate_step,
                    sp_interval=sp_interval,
                    fi_interval=fi_interval,
                    warm_up=warm_up
                )
                keep_shape = keep_shape if not lcm else lcm
                register_forward(self.transformer, 
                    'Attention',
                    ca_kward=ca_kwards,
                    sa_kward=sa_kwards,
                    keep_shape=keep_shape
                    )
            # predict noise model_output
            noise_pred = self.transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=current_timestep,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance and (i-num_warmup_steps) < gate_step:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            else:
                noise_pred = noise_pred

            # compute previous image: x_t -> x_t-1
            if num_inference_steps == 1:
                # For DMD one step sampling: https://arxiv.org/abs/2311.18828
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).pred_original_sample
            else:
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

    if not output_type == "latent":
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        if use_resolution_binning:
            image = self.image_processor.resize_and_crop_tensor(image, orig_width, orig_height)
    else:
        image = latents

    if not output_type == "latent":
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return ImagePipelineOutput(images=image)


def TgatePixArtAlphaLoader(pipe, **kwargs):
    pipe.tgate = MethodType(tgate,pipe)
    return pipe
