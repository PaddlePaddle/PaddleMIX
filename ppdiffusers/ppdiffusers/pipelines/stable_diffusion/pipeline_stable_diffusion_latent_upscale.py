import paddle
import warnings
from typing import Callable, List, Optional, Union
import numpy as np
import PIL
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL, UNet2DConditionModel
from ...schedulers import EulerDiscreteScheduler
from ...utils import logging, randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from paddlenlp.transformers import CLIPTextModel, CLIPTokenizer
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


class StableDiffusionLatentUpscalePipeline(DiffusionPipeline):
    """
    Pipeline for upscaling Stable Diffusion output image resolution by a factor of 2.

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
        scheduler ([`SchedulerMixin`]):
            A [`EulerDiscreteScheduler`] to be used in combination with `unet` to denoise the encoded image latents.
    """

    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: EulerDiscreteScheduler):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler)
        self.vae_scale_factor = 2**(len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, resample='bicubic')

    def _encode_prompt(self, prompt, device, do_classifier_free_guidance,
                       negative_prompt):
        """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `list(int)`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        text_inputs = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_length=True,
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
        text_encoder_out = self.text_encoder(
            text_input_ids.to(device), output_hidden_states=True)
        text_embeddings = text_encoder_out.hidden_states[-1]
        text_pooler_out = text_encoder_out.pooler_output
        if do_classifier_free_guidance:
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
            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding='max_length',
                max_length=max_length,
                truncation=True,
                return_length=True,
                return_tensors='pt')
            uncond_encoder_out = self.text_encoder(
                uncond_input.input_ids.to(device), output_hidden_states=True)
            uncond_embeddings = uncond_encoder_out.hidden_states[-1]
            uncond_pooler_out = uncond_encoder_out.pooler_output
            text_embeddings = paddle.concat(
                x=[uncond_embeddings, text_embeddings])
            text_pooler_out = paddle.concat(
                x=[uncond_pooler_out, text_pooler_out])
        return text_embeddings, text_pooler_out

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

    def check_inputs(self, prompt, image, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(
                f'`prompt` has to be of type `str` or `list` but is {type(prompt)}'
            )
        if not isinstance(image, paddle.Tensor) and not isinstance(
                image, PIL.Image.Image) and not isinstance(image, list):
            raise ValueError(
                f'`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or `list` but is {type(image)}'
            )
        if isinstance(image, list) or isinstance(image, paddle.Tensor):
            if isinstance(prompt, str):
                batch_size = 1
            else:
                batch_size = len(prompt)
            if isinstance(image, list):
                image_batch_size = len(image)
            else:
                image_batch_size = image.shape[0] if image.ndim == 4 else 1
            if batch_size != image_batch_size:
                raise ValueError(
                    f'`prompt` has batch size {batch_size} and `image` has batch size {image_batch_size}. Please make sure that passed `prompt` matches the batch size of `image`.'
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

    @paddle.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            image: Union[paddle.Tensor, PIL.Image.Image, np.ndarray, List[
                paddle.Tensor], List[PIL.Image.Image], List[np.ndarray]]=None,
            num_inference_steps: int=75,
            guidance_scale: float=9.0,
            negative_prompt: Optional[Union[str, List[str]]]=None,
            generator: Optional[Union[torch.Generator, List[
                torch.Generator]]]=None,
            latents: Optional[paddle.Tensor]=None,
            output_type: Optional[str]='pil',
            return_dict: bool=True,
            callback: Optional[Callable[[int, int, paddle.Tensor], None]]=None,
            callback_steps: int=1):
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image upscaling.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image` or tensor representing an image batch to be upscaled. If it's a tensor, it can be either a
                latent output from a Stable Diffusion model or an image tensor in the range `[-1, 1]`. It is considered
                a `latent` if `image.shape[1]` is `4`; otherwise, it is considered to be an image representation and
                encoded using this pipeline's `vae` encoder.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
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
        >>> from diffusers import StableDiffusionLatentUpscalePipeline, StableDiffusionPipeline
        >>> import torch


        >>> pipeline = StableDiffusionPipeline.from_pretrained(
        ...     "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
        ... )
        >>> pipeline.to("cuda")

        >>> model_id = "stabilityai/sd-x2-latent-upscaler"
        >>> upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        >>> upscaler.to("cuda")

        >>> prompt = "a photo of an astronaut high resolution, unreal engine, ultra realistic"
        >>> generator = torch.manual_seed(33)

        >>> low_res_latents = pipeline(prompt, generator=generator, output_type="latent").images

        >>> with torch.no_grad():
        ...     image = pipeline.decode_latents(low_res_latents)
        >>> image = pipeline.numpy_to_pil(image)[0]

        >>> image.save("../images/a1.png")

        >>> upscaled_image = upscaler(
        ...     prompt=prompt,
        ...     image=low_res_latents,
        ...     num_inference_steps=20,
        ...     guidance_scale=0,
        ...     generator=generator,
        ... ).images[0]

        >>> upscaled_image.save("../images/a2.png")
        ```

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images.
        """
        self.check_inputs(prompt, image, callback_steps)
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        if guidance_scale == 0:
            prompt = [''] * batch_size
        text_embeddings, text_pooler_out = self._encode_prompt(
            prompt, device, do_classifier_free_guidance, negative_prompt)
        image = self.image_processor.preprocess(image)
        image = image.to(dtype=text_embeddings.dtype, device=device)
        if image.shape[1] == 3:
            image = self.vae.encode(image).latent_dist.sample(
            ) * self.vae.config.scaling_factor
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        batch_multiplier = 2 if do_classifier_free_guidance else 1
        image = image[None, :] if image.ndim == 3 else image
        image = paddle.concat(x=[image] * batch_multiplier)
        noise_level = paddle.to_tensor(
            data=[0.0], dtype='float32', place=device)
        noise_level = paddle.concat(x=[noise_level] * image.shape[0])
        inv_noise_level = (noise_level**2 + 1)**-0.5
        image_cond = paddle.nn.functional.interpolate(
            x=image, scale_factor=2,
            mode='nearest') * inv_noise_level[:, None, None, None]
        image_cond = image_cond.to(text_embeddings.dtype)
        noise_level_embed = paddle.concat(
            x=[
                paddle.ones(
                    shape=[text_pooler_out.shape[0], 64],
                    dtype=text_pooler_out.dtype), paddle.zeros(
                        shape=[text_pooler_out.shape[0], 64],
                        dtype=text_pooler_out.dtype)
            ],
            axis=1)
        timestep_condition = paddle.concat(
            x=[noise_level_embed, text_pooler_out], axis=1)
        height, width = image.shape[2:]
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size, num_channels_latents, height * 2, width * 2,
            text_embeddings.dtype, device, generator, latents)
        num_channels_image = image.shape[1]
        if (num_channels_latents + num_channels_image !=
                self.unet.config.in_channels):
            raise ValueError(
                f'Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} + `num_channels_image`: {num_channels_image}  = {num_channels_latents + num_channels_image}. Please verify the config of `pipeline.unet` or your `image` input.'
            )
        num_warmup_steps = 0
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                sigma = self.scheduler.sigmas[i]
                latent_model_input = paddle.concat(
                    x=[latents] * 2) if do_classifier_free_guidance else latents
                scaled_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)
                scaled_model_input = paddle.concat(
                    x=[scaled_model_input, image_cond], axis=1)
                timestep = paddle.log(x=sigma) * 0.25
                noise_pred = self.unet(
                    scaled_model_input,
                    timestep,
                    encoder_hidden_states=text_embeddings,
                    timestep_cond=timestep_condition).sample
                noise_pred = noise_pred[:, :-1]
                inv_sigma = 1 / (sigma**2 + 1)
                noise_pred = (
                    inv_sigma * latent_model_input +
                    self.scheduler.scale_model_input(sigma, t) * noise_pred)
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(
                        chunks=2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t,
                                              latents).prev_sample
                if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (
                        i + 1) % self.scheduler.order == 0:
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        if not output_type == 'latent':
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
        image = self.image_processor.postprocess(image, output_type=output_type)
        if not return_dict:
            return image,
        return ImagePipelineOutput(images=image)
