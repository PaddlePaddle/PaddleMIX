import paddle
import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import PIL
from ...image_processor import VaeImageProcessor
from ...loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from ...models import AutoencoderKL, UNet2DConditionModel
from ...models.attention_processor import AttnProcessor2_0, LoRAAttnProcessor2_0, LoRAXFormersAttnProcessor, XFormersAttnProcessor
from ...schedulers import DDPMScheduler, KarrasDiffusionSchedulers
from ...utils import deprecate, is_accelerate_available, is_accelerate_version, logging, randn_tensor
from ..pipeline_utils import DiffusionPipeline
from . import StableDiffusionPipelineOutput
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
        w, h = (x - x % 64 for x in (w, h))
        image = [np.array(i.resize((w, h)))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = paddle.to_tensor(data=image)
    elif isinstance(image[0], paddle.Tensor):
        image = paddle.concat(x=image, axis=0)
    return image


class StableDiffusionUpscalePipeline(
        DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin):
    """
    Pipeline for text-guided image super-resolution using Stable Diffusion 2.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        low_res_scheduler ([`SchedulerMixin`]):
            A scheduler used to add initial noise to the low resolution conditioning image. It must be an instance of
            [`DDPMScheduler`].
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """
    _optional_components = [
        'watermarker', 'safety_checker', 'feature_extractor'
    ]

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: transformers.CLIPTextModel,
            tokenizer: transformers.CLIPTokenizer,
            unet: UNet2DConditionModel,
            low_res_scheduler: DDPMScheduler,
            scheduler: KarrasDiffusionSchedulers,
            safety_checker: Optional[Any]=None,
            feature_extractor: Optional[transformers.CLIPImageProcessor]=None,
            watermarker: Optional[Any]=None,
            max_noise_level: int=350):
        super().__init__()
        if hasattr(vae, 'config'):
            is_vae_scaling_factor_set_to_0_08333 = hasattr(
                vae.config,
                'scaling_factor') and vae.config.scaling_factor == 0.08333
            if not is_vae_scaling_factor_set_to_0_08333:
                deprecation_message = (
                    f"The configuration file of the vae does not contain `scaling_factor` or it is set to {vae.config.scaling_factor}, which seems highly unlikely. If your checkpoint is a fine-tuned version of `stabilityai/stable-diffusion-x4-upscaler` you should change 'scaling_factor' to 0.08333 Please make sure to update the config accordingly, as not doing so might lead to incorrect results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull Request for the `vae/config.json` file"
                )
                deprecate(
                    'wrong scaling_factor',
                    '1.0.0',
                    deprecation_message,
                    standard_warn=False)
                vae.register_to_config(scaling_factor=0.08333)
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            low_res_scheduler=low_res_scheduler,
            scheduler=scheduler,
            safety_checker=safety_checker,
            watermarker=watermarker,
            feature_extractor=feature_extractor)
        self.vae_scale_factor = 2**(len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, resample='bicubic')
        self.register_to_config(max_noise_level=max_noise_level)

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
        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                _, hook = cpu_offload_with_hook(
                    cpu_offloaded_model, device, prev_module_hook=hook)
        self.final_offload_hook = hook

    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors='pt').to(device)
            image, nsfw_detected, watermark_detected = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.to(dtype=dtype))
        else:
            nsfw_detected = None
            watermark_detected = None
            if hasattr(
                    self,
                    'unet_offload_hook') and self.unet_offload_hook is not None:
                self.unet_offload_hook.offload()
        return image, nsfw_detected, watermark_detected

    def _encode_prompt(self,
                       prompt,
                       device,
                       num_images_per_prompt,
                       do_classifier_free_guidance,
                       negative_prompt=None,
                       prompt_embeds: Optional[paddle.Tensor]=None,
                       negative_prompt_embeds: Optional[paddle.Tensor]=None,
                       lora_scale: Optional[float]=None):
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
            negative_prompt (`str` or `List[str]`, *optional*):
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
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale
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
            elif prompt is not None and type(prompt) is not type(
                    negative_prompt):
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
            prompt_embeds = paddle.concat(
                x=[negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

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
                     image,
                     noise_level,
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
        if not isinstance(image, paddle.Tensor) and not isinstance(
                image, PIL.Image.Image) and not isinstance(
                    image, np.ndarray) and not isinstance(image, list):
            raise ValueError(
                f'`image` has to be of type `torch.Tensor`, `np.ndarray`, `PIL.Image.Image` or `list` but is {type(image)}'
            )
        if isinstance(image, list) or isinstance(
                image, paddle.Tensor) or isinstance(image, np.ndarray):
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]
            if isinstance(image, list):
                image_batch_size = len(image)
            else:
                image_batch_size = image.shape[0]
            if batch_size != image_batch_size:
                raise ValueError(
                    f'`prompt` has batch size {batch_size} and `image` has batch size {image_batch_size}. Please make sure that passed `prompt` matches the batch size of `image`.'
                )
        if noise_level > self.config.max_noise_level:
            raise ValueError(
                f'`noise_level` has to be <= {self.config.max_noise_level} but is {noise_level}'
            )
        if callback_steps is None or callback_steps is not None and (
                not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f'`callback_steps` has to be a positive integer but is {callback_steps} of type {type(callback_steps)}.'
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
        shape = batch_size, num_channels_latents, height, width
        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(
                    f'Unexpected latents shape, got {latents.shape}, expected {shape}'
                )
            latents = latents.to(device)
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype='float32')
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (AttnProcessor2_0, XFormersAttnProcessor, LoRAXFormersAttnProcessor,
             LoRAAttnProcessor2_0))
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    @paddle.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]]=None,
            image: Union[paddle.Tensor, PIL.Image.Image, np.ndarray, List[
                paddle.Tensor], List[PIL.Image.Image], List[np.ndarray]]=None,
            num_inference_steps: int=75,
            guidance_scale: float=9.0,
            noise_level: int=20,
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
            callback_steps: int=1,
            cross_attention_kwargs: Optional[Dict[str, Any]]=None):
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image` or tensor representing an image batch to be upscaled.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
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
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:
        ```py
        >>> import requests
        >>> from PIL import Image
        >>> from io import BytesIO
        >>> from diffusers import StableDiffusionUpscalePipeline
        >>> import torch

        >>> # load model and scheduler
        >>> model_id = "stabilityai/stable-diffusion-x4-upscaler"
        >>> pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        ...     model_id, revision="fp16", torch_dtype=torch.float16
        ... )
        >>> pipeline = pipeline.to("cuda")

        >>> # let's download an  image
        >>> url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
        >>> response = requests.get(url)
        >>> low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
        >>> low_res_img = low_res_img.resize((128, 128))
        >>> prompt = "a white cat"

        >>> upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
        >>> upscaled_image.save("upsampled_cat.png")
        ```

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        self.check_inputs(prompt, image, noise_level, callback_steps,
                          negative_prompt, prompt_embeds,
                          negative_prompt_embeds)
        if image is None:
            raise ValueError('`image` input cannot be undefined.')
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        text_encoder_lora_scale = cross_attention_kwargs.get(
            'scale', None) if cross_attention_kwargs is not None else None
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale)
        image = self.image_processor.preprocess(image)
        image = image.to(dtype=prompt_embeds.dtype, device=device)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        noise_level = paddle.to_tensor(
            data=[noise_level], dtype='int64', place=device)
        noise = randn_tensor(
            image.shape,
            generator=generator,
            device=device,
            dtype=prompt_embeds.dtype)
        image = self.low_res_scheduler.add_noise(image, noise, noise_level)
        batch_multiplier = 2 if do_classifier_free_guidance else 1
        image = paddle.concat(x=[image] * batch_multiplier *
                              num_images_per_prompt)
        noise_level = paddle.concat(x=[noise_level] * image.shape[0])
        height, width = image.shape[2:]
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents, height,
            width, prompt_embeds.dtype, device, generator, latents)
        num_channels_image = image.shape[1]
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
                    x=[latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)
                latent_model_input = paddle.concat(
                    x=[latent_model_input, image], axis=1)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=noise_level,
                    return_dict=False)[0]
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(
                        chunks=2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond)
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
        if self.vae.dtype == 'float16' and self.vae.config.force_upcast:
            self.upcast_vae()
            latents = latents.to(
                next(iter(self.vae.post_quant_conv.parameters())).dtype)
        if not output_type == 'latent':
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept, _ = self.run_safety_checker(
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
        if output_type == 'pil' and self.watermarker is not None:
            image = self.watermarker.apply_watermark(image)
        if hasattr(
                self,
                'final_offload_hook') and self.final_offload_hook is not None:
            self.final_offload_hook.offload()
        if not return_dict:
            return image, has_nsfw_concept
        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept)
