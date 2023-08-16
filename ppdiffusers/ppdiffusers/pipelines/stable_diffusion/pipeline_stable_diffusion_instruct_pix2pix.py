import paddle
import inspect
import warnings
from typing import Callable, List, Optional, Union
import numpy as np
import PIL
from ...image_processor import VaeImageProcessor
from ...loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import PIL_INTERPOLATION, deprecate, is_accelerate_available, is_accelerate_version, logging, randn_tensor
from ..pipeline_utils import DiffusionPipeline
from . import StableDiffusionPipelineOutput
from .safety_checker import StableDiffusionSafetyChecker
from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
logger = logging.get_logger(__name__)


def preprocess(image):
    warnings.warn(
        'The preprocess method is deprecated and will be removed in a future version. Please use VaeImageProcessor.preprocess instead',
        FutureWarning)
    if isinstance(image, paddle.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]
    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = (x - x % 8 for x in (w, h))
        image = [
            np.array(i.resize(
                (w, h), resample=PIL_INTERPOLATION['lanczos']))[None, :]
            for i in image
        ]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = paddle.to_tensor(data=image)
    elif isinstance(image[0], paddle.Tensor):
        image = paddle.concat(x=image, axis=0)
    return image


class StableDiffusionInstructPix2PixPipeline(
        DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin):
    """
    Pipeline for pixel-level image editing by following text instructions (based on Stable Diffusion).

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """
    _optional_components = ['safety_checker', 'feature_extractor']

    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: KarrasDiffusionSchedulers,
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: CLIPImageProcessor,
                 requires_safety_checker: bool=True):
        super().__init__()
        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f'You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered results in services or applications open to the public. Both the diffusers team and Hugging Face strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling it only for use-cases that involve analyzing network behavior or auditing its results. For more information, please have a look at https://github.com/huggingface/diffusers/pull/254 .'
            )
        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor)
        self.vae_scale_factor = 2**(len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    @paddle.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]]=None,
            image: Union[paddle.Tensor, PIL.Image.Image, np.ndarray, List[
                paddle.Tensor], List[PIL.Image.Image], List[np.ndarray]]=None,
            num_inference_steps: int=100,
            guidance_scale: float=7.5,
            image_guidance_scale: float=1.5,
            negative_prompt: Optional[Union[str, List[str]]]=None,
            num_images_per_prompt: Optional[int]=1,
            eta: float=0.0,
            generator: Optional[Union[torch.Generator, List[
                torch.Generator]]]=None,
            latents: Optional[paddle.Tensor]=None,
            prompt_embeds: Optional[paddle.Tensor]=None,
            negative_prompt_embeds: Optional[paddle.Tensor]=None,
            output_type: Optional[str]='pil',
            return_dict: bool=True,
            callback: Optional[Callable[[int, int, paddle.Tensor], None]]=None,
            callback_steps: int=1):
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.FloatTensor` `np.ndarray`, `PIL.Image.Image`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image` or tensor representing an image batch to be repainted according to `prompt`. Can also accept
                image latents as `image`, but if passing latents directly it is not encoded again.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            image_guidance_scale (`float`, *optional*, defaults to 1.5):
                Push the generated image towards the inital `image`. Image guidance scale is enabled by setting
                `image_guidance_scale > 1`. Higher image guidance scale encourages generated images that are closely
                linked to the source `image`, usually at the expense of lower image quality. This pipeline requires a
                value of at least `1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.

        Examples:

        ```py
        >>> import PIL
        >>> import requests
        >>> import torch
        >>> from io import BytesIO

        >>> from diffusers import StableDiffusionInstructPix2PixPipeline


        >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


        >>> img_url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"

        >>> image = download_image(img_url).resize((512, 512))

        >>> pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        ...     "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "make the mountains snowy"
        >>> image = pipe(prompt=prompt, image=image).images[0]
        ```

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        self.check_inputs(prompt, callback_steps, negative_prompt,
                          prompt_embeds, negative_prompt_embeds)
        if image is None:
            raise ValueError('`image` input cannot be undefined.')
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = (guidance_scale > 1.0 and
                                       image_guidance_scale >= 1.0)
        scheduler_is_in_sigma_space = hasattr(self.scheduler, 'sigmas')
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds)
        image = self.image_processor.preprocess(image)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        image_latents = self.prepare_image_latents(
            image, batch_size, num_images_per_prompt, prompt_embeds.dtype,
            device, do_classifier_free_guidance, generator)
        height, width = image_latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents, height,
            width, prompt_embeds.dtype, device, generator, latents)
        num_channels_image = image_latents.shape[1]
        if (num_channels_latents + num_channels_image !=
                self.unet.config.in_channels):
            raise ValueError(
                f'Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} + `num_channels_image`: {num_channels_image}  = {num_channels_latents + num_channels_image}. Please verify the config of `pipeline.unet` or your `image` input.'
            )
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        num_warmup_steps = len(
            timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = paddle.concat(
                    x=[latents] * 3) if do_classifier_free_guidance else latents
                scaled_latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)
                scaled_latent_model_input = paddle.concat(
                    x=[scaled_latent_model_input, image_latents], axis=1)
                noise_pred = self.unet(
                    scaled_latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False)[0]
                if scheduler_is_in_sigma_space:
                    step_index = (
                        self.scheduler.timesteps == t).nonzero()[0].item()
                    sigma = self.scheduler.sigmas[step_index]
                    noise_pred = latent_model_input - sigma * noise_pred
                if do_classifier_free_guidance:
                    (noise_pred_text, noise_pred_image,
                     noise_pred_uncond) = noise_pred.chunk(chunks=3)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_image
                    ) + image_guidance_scale * (noise_pred_image -
                                                noise_pred_uncond)
                if scheduler_is_in_sigma_space:
                    noise_pred = (noise_pred - latents) / -sigma
                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    **extra_step_kwargs,
                    return_dict=False)[0]
                if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (
                        i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        if not output_type == 'latent':
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(
                image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [(not has_nsfw) for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize)
        if hasattr(
                self,
                'final_offload_hook') and self.final_offload_hook is not None:
            self.final_offload_hook.offload()
        if not return_dict:
            return image, has_nsfw_concept
        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept)

    def enable_model_cpu_offload(self, gpu_id=0):
        """
        Offload all models to CPU to reduce memory usage with a low impact on performance. Moves one whole model at a
        time to the GPU when its `forward` method is called, and the model remains in GPU until the next model runs.
        Memory savings are lower than using `enable_sequential_cpu_offload`, but performance is much better due to the
        iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version('>=',
                                                               '0.17.0.dev0'):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError(
                '`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.'
            )
        device = str(f'cuda:{gpu_id}').replace('cuda', 'gpu')
        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(
                cpu_offloaded_model, device, prev_module_hook=hook)
        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(
                self.safety_checker, device, prev_module_hook=hook)
        self.final_offload_hook = hook

    def _encode_prompt(self,
                       prompt,
                       device,
                       num_images_per_prompt,
                       do_classifier_free_guidance,
                       negative_prompt=None,
                       prompt_embeds: Optional[paddle.Tensor]=None,
                       negative_prompt_embeds: Optional[paddle.Tensor]=None):
        """
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)
            text_inputs = self.tokenizer(
                prompt,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt')
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding='longest', return_tensors='pt').input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                    -1] and not paddle.equal_all(
                        x=text_input_ids, y=untruncated_ids).item():
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1:-1])
                logger.warning(
                    f'The following part of your input was truncated because CLIP can only handle sequences up to {self.tokenizer.model_max_length} tokens: {removed_text}'
                )
            if hasattr(self.text_encoder.config, 'use_attention_mask'
                       ) and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None
            prompt_embeds = self.text_encoder(
                text_input_ids.to(device), attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype,
                                         device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.tile(
            repeat_times=[1, num_images_per_prompt, 1])
        prompt_embeds = prompt_embeds.reshape(
            [bs_embed * num_images_per_prompt, seq_len, -1])
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [''] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}.'
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches the batch size of `prompt`.'
                )
            else:
                uncond_tokens = negative_prompt
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens,
                                                          self.tokenizer)
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding='max_length',
                max_length=max_length,
                truncation=True,
                return_tensors='pt')
            if hasattr(self.text_encoder.config, 'use_attention_mask'
                       ) and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None
            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask)
            negative_prompt_embeds = negative_prompt_embeds[0]
        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.tile(
                repeat_times=[1, num_images_per_prompt, 1])
            negative_prompt_embeds = negative_prompt_embeds.reshape(
                [batch_size * num_images_per_prompt, seq_len, -1])
            prompt_embeds = paddle.concat(x=[
                prompt_embeds, negative_prompt_embeds, negative_prompt_embeds
            ])
        return prompt_embeds

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is None:
            has_nsfw_concept = None
        else:
            if paddle.is_tensor(x=image):
                feature_extractor_input = self.image_processor.postprocess(
                    image, output_type='pil')
            else:
                feature_extractor_input = self.image_processor.numpy_to_pil(
                    image)
            safety_checker_input = self.feature_extractor(
                feature_extractor_input, return_tensors='pt').to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.to(dtype))
        return image, has_nsfw_concept

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = 'eta' in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs['eta'] = eta
        accepts_generator = 'generator' in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs['generator'] = generator
        return extra_step_kwargs

    def decode_latents(self, latents):
        warnings.warn(
            'The decode_latents method is deprecated and will be removed in a future version. Please use VaeImageProcessor instead',
            FutureWarning)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clip(min=0, max=1)
        image = image.cpu().transpose(perm=[0, 2, 3, 1]).astype(
            dtype='float32').numpy()
        return image

    def check_inputs(self,
                     prompt,
                     callback_steps,
                     negative_prompt=None,
                     prompt_embeds=None,
                     negative_prompt_embeds=None):
        if callback_steps is None or callback_steps is not None and (
                not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.'
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f'Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to only forward one of the two.'
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                'Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined.'
            )
        elif prompt is not None and (not isinstance(prompt, str) and
                                     not isinstance(prompt, list)):
            raise ValueError(
                f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f'Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to only forward one of the two.'
            )
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    f'`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds` {negative_prompt_embeds.shape}.'
                )

    def prepare_latents(self,
                        batch_size,
                        num_channels_latents,
                        height,
                        width,
                        dtype,
                        device,
                        generator,
                        latents=None):
        shape = (batch_size, num_channels_latents, height //
                 self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
            )
        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_image_latents(self,
                              image,
                              batch_size,
                              num_images_per_prompt,
                              dtype,
                              device,
                              do_classifier_free_guidance,
                              generator=None):
        if not isinstance(image, (paddle.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f'`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}'
            )
        image = image.to(device=device, dtype=dtype)
        batch_size = batch_size * num_images_per_prompt
        if image.shape[1] == 4:
            image_latents = image
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.'
                )
            if isinstance(generator, list):
                image_latents = [
                    self.vae.encode(image[i:i + 1]).latent_dist.mode()
                    for i in range(batch_size)
                ]
                image_latents = paddle.concat(x=image_latents, axis=0)
            else:
                image_latents = self.vae.encode(image).latent_dist.mode()
        if batch_size > image_latents.shape[
                0] and batch_size % image_latents.shape[0] == 0:
            deprecation_message = (
                f'You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial images (`image`). Initial images are now duplicating to match the number of text prompts. Note that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update your script to pass as many initial images as text prompts to suppress this warning.'
            )
            deprecate(
                'len(prompt) != len(image)',
                '1.0.0',
                deprecation_message,
                standard_warn=False)
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = paddle.concat(
                x=[image_latents] * additional_image_per_prompt, axis=0)
        elif batch_size > image_latents.shape[
                0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f'Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts.'
            )
        else:
            image_latents = paddle.concat(x=[image_latents], axis=0)
        if do_classifier_free_guidance:
            uncond_image_latents = paddle.zeros_like(x=image_latents)
            image_latents = paddle.concat(
                x=[image_latents, image_latents, uncond_image_latents], axis=0)
        return image_latents
