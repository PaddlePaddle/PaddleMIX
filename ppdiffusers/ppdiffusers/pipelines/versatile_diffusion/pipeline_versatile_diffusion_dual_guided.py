import paddle
import inspect
import warnings
from typing import Callable, List, Optional, Tuple, Union
import numpy as np
import PIL
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL, DualTransformer2DModel, Transformer2DModel, UNet2DConditionModel
from ...schedulers import KarrasDiffusionSchedulers
from ...utils import logging, randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from .modeling_text_unet import UNetFlatConditionModel

from paddlenlp.transformers import (
    CLIPImageProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection, )
logger = logging.get_logger(__name__)


class VersatileDiffusionDualGuidedPipeline(DiffusionPipeline):
    """
    Pipeline for image-text dual-guided generation using Versatile Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        vqvae ([`VQModel`]):
            Vector-quantized (VQ) model to encode and decode images to and from latent representations.
        bert ([`LDMBertModel`]):
            Text-encoder model based on [`~transformers.BERT`].
        tokenizer ([`~transformers.BertTokenizer`]):
            A `BertTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """
    tokenizer: transformers.CLIPTokenizer
    image_feature_extractor: transformers.CLIPImageProcessor
    text_encoder: transformers.CLIPTextModelWithProjection
    image_encoder: transformers.CLIPVisionModelWithProjection
    image_unet: UNet2DConditionModel
    text_unet: UNetFlatConditionModel
    vae: AutoencoderKL
    scheduler: KarrasDiffusionSchedulers
    _optional_components = ['text_unet']

    def __init__(self,
                 tokenizer: transformers.CLIPTokenizer,
                 image_feature_extractor: transformers.CLIPImageProcessor,
                 text_encoder: transformers.CLIPTextModelWithProjection,
                 image_encoder: transformers.CLIPVisionModelWithProjection,
                 image_unet: UNet2DConditionModel,
                 text_unet: UNetFlatConditionModel,
                 vae: AutoencoderKL,
                 scheduler: KarrasDiffusionSchedulers):
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer,
            image_feature_extractor=image_feature_extractor,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            image_unet=image_unet,
            text_unet=text_unet,
            vae=vae,
            scheduler=scheduler)
        self.vae_scale_factor = 2**(len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor)
        if self.text_unet is not None and (
                'dual_cross_attention' not in self.image_unet.config or
                not self.image_unet.config.dual_cross_attention):
            self._convert_to_dual_attention()

    def remove_unused_weights(self):
        self.register_modules(text_unet=None)

    def _convert_to_dual_attention(self):
        """
        Replace image_unet's `Transformer2DModel` blocks with `DualTransformer2DModel` that contains transformer blocks
        from both `image_unet` and `text_unet`
        """
        for name, module in self.image_unet.named_modules():
            if isinstance(module, Transformer2DModel):
                parent_name, index = name.rsplit('.', 1)
                index = int(index)
                image_transformer = self.image_unet.get_submodule(parent_name)[
                    index]
                text_transformer = self.text_unet.get_submodule(parent_name)[
                    index]
                config = image_transformer.config
                dual_transformer = DualTransformer2DModel(
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=config.attention_head_dim,
                    in_channels=config.in_channels,
                    num_layers=config.num_layers,
                    dropout=config.dropout,
                    norm_num_groups=config.norm_num_groups,
                    cross_attention_dim=config.cross_attention_dim,
                    attention_bias=config.attention_bias,
                    sample_size=config.sample_size,
                    num_vector_embeds=config.num_vector_embeds,
                    activation_fn=config.activation_fn,
                    num_embeds_ada_norm=config.num_embeds_ada_norm)
                dual_transformer.transformers[0] = image_transformer
                dual_transformer.transformers[1] = text_transformer
                self.image_unet.get_submodule(parent_name)[
                    index] = dual_transformer
                self.image_unet.register_to_config(dual_cross_attention=True)

    def _revert_dual_attention(self):
        """
        Revert the image_unet `DualTransformer2DModel` blocks back to `Transformer2DModel` with image_unet weights Call
        this function if you reuse `image_unet` in another pipeline, e.g. `VersatileDiffusionPipeline`
        """
        for name, module in self.image_unet.named_modules():
            if isinstance(module, DualTransformer2DModel):
                parent_name, index = name.rsplit('.', 1)
                index = int(index)
                self.image_unet.get_submodule(parent_name)[
                    index] = module.transformers[0]
        self.image_unet.register_to_config(dual_cross_attention=False)

    def _encode_text_prompt(self, prompt, device, num_images_per_prompt,
                            do_classifier_free_guidance):
        """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
        """

        def normalize_embeddings(encoder_output):
            embeds = self.text_encoder.text_projection(
                encoder_output.last_hidden_state)
            embeds_pooled = encoder_output.text_embeds
            embeds = embeds / paddle.linalg.norm(
                x=embeds_pooled.unsqueeze(axis=1), axis=-1, keepdim=True)
            return embeds

        batch_size = len(prompt)
        text_inputs = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt')
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding='max_length', return_tensors='pt').input_ids
        if not paddle.equal_all(x=text_input_ids, y=untruncated_ids).item():
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
        prompt_embeds = normalize_embeddings(prompt_embeds)
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.tile(
            repeat_times=[1, num_images_per_prompt, 1])
        prompt_embeds = prompt_embeds.reshape(
            [bs_embed * num_images_per_prompt, seq_len, -1])
        if do_classifier_free_guidance:
            uncond_tokens = [''] * batch_size
            max_length = text_input_ids.shape[-1]
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
            negative_prompt_embeds = normalize_embeddings(
                negative_prompt_embeds)
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.tile(
                repeat_times=[1, num_images_per_prompt, 1])
            negative_prompt_embeds = negative_prompt_embeds.reshape(
                [batch_size * num_images_per_prompt, seq_len, -1])
            prompt_embeds = paddle.concat(
                x=[negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    def _encode_image_prompt(self, prompt, device, num_images_per_prompt,
                             do_classifier_free_guidance):
        """
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
        """

        def normalize_embeddings(encoder_output):
            embeds = self.image_encoder.vision_model.post_layernorm(
                encoder_output.last_hidden_state)
            embeds = self.image_encoder.visual_projection(embeds)
            embeds_pooled = embeds[:, 0:1]
            embeds = embeds / paddle.linalg.norm(
                x=embeds_pooled, axis=-1, keepdim=True)
            return embeds

        batch_size = len(prompt) if isinstance(prompt, list) else 1
        image_input = self.image_feature_extractor(
            images=prompt, return_tensors='pt')
        pixel_values = image_input.pixel_values.to(device).to(
            self.image_encoder.dtype)
        image_embeddings = self.image_encoder(pixel_values)
        image_embeddings = normalize_embeddings(image_embeddings)
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.tile(
            repeat_times=[1, num_images_per_prompt, 1])
        image_embeddings = image_embeddings.reshape(
            [bs_embed * num_images_per_prompt, seq_len, -1])
        if do_classifier_free_guidance:
            uncond_images = [np.zeros((512, 512, 3)) + 0.5] * batch_size
            uncond_images = self.image_feature_extractor(
                images=uncond_images, return_tensors='pt')
            pixel_values = uncond_images.pixel_values.to(device).to(
                self.image_encoder.dtype)
            negative_prompt_embeds = self.image_encoder(pixel_values)
            negative_prompt_embeds = normalize_embeddings(
                negative_prompt_embeds)
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.tile(
                repeat_times=[1, num_images_per_prompt, 1])
            negative_prompt_embeds = negative_prompt_embeds.reshape(
                [batch_size * num_images_per_prompt, seq_len, -1])
            image_embeddings = paddle.concat(
                x=[negative_prompt_embeds, image_embeddings])
        return image_embeddings

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

    def check_inputs(self, prompt, image, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(
                prompt, PIL.Image.Image) and not isinstance(prompt, list):
            raise ValueError(
                f'`prompt` has to be of type `str` `PIL.Image` or `list` but is {type(prompt)}'
            )
        if not isinstance(image, str) and not isinstance(
                image, PIL.Image.Image) and not isinstance(image, list):
            raise ValueError(
                f'`image` has to be of type `str` `PIL.Image` or `list` but is {type(image)}'
            )
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f'`height` and `width` have to be divisible by 8 but are {height} and {width}.'
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

    def set_transformer_params(self,
                               mix_ratio: float=0.5,
                               condition_types: Tuple=('text', 'image')):
        for name, module in self.image_unet.named_modules():
            if isinstance(module, DualTransformer2DModel):
                module.mix_ratio = mix_ratio
                for i, type in enumerate(condition_types):
                    if type == 'text':
                        module.condition_lengths[
                            i] = self.text_encoder.config.max_position_embeddings
                        module.transformer_index_for_condition[i] = 1
                    else:
                        module.condition_lengths[i] = 257
                        module.transformer_index_for_condition[i] = 0

    @paddle.no_grad()
    def __call__(
            self,
            prompt: Union[PIL.Image.Image, List[PIL.Image.Image]],
            image: Union[str, List[str]],
            text_to_image_strength: float=0.5,
            height: Optional[int]=None,
            width: Optional[int]=None,
            num_inference_steps: int=50,
            guidance_scale: float=7.5,
            num_images_per_prompt: Optional[int]=1,
            eta: float=0.0,
            generator: Optional[Union[torch.Generator, List[
                torch.Generator]]]=None,
            latents: Optional[paddle.Tensor]=None,
            output_type: Optional[str]='pil',
            return_dict: bool=True,
            callback: Optional[Callable[[int, int, paddle.Tensor], None]]=None,
            callback_steps: int=1,
            **kwargs):
        """
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation.
            height (`int`, *optional*, defaults to `self.image_unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.image_unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
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
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.

        Examples:

        ```py
        >>> from diffusers import VersatileDiffusionDualGuidedPipeline
        >>> import torch
        >>> import requests
        >>> from io import BytesIO
        >>> from PIL import Image

        >>> # let's download an initial image
        >>> url = "https://huggingface.co/datasets/diffusers/images/resolve/main/benz.jpg"

        >>> response = requests.get(url)
        >>> image = Image.open(BytesIO(response.content)).convert("RGB")
        >>> text = "a red car in the sun"

        >>> pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(
        ...     "shi-labs/versatile-diffusion", torch_dtype=torch.float16
        ... )
        >>> pipe.remove_unused_weights()
        >>> pipe = pipe.to("cuda")

        >>> generator = torch.Generator(device="cuda").manual_seed(0)
        >>> text_to_image_strength = 0.75

        >>> image = pipe(
        ...     prompt=text, image=image, text_to_image_strength=text_to_image_strength, generator=generator
        ... ).images[0]
        >>> image.save("./car_variation.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images.
        """
        height = (height or
                  self.image_unet.config.sample_size * self.vae_scale_factor)
        width = (width or
                 self.image_unet.config.sample_size * self.vae_scale_factor)
        self.check_inputs(prompt, image, height, width, callback_steps)
        prompt = [prompt] if not isinstance(prompt, list) else prompt
        image = [image] if not isinstance(image, list) else image
        batch_size = len(prompt)
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds = self._encode_text_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance)
        image_embeddings = self._encode_image_prompt(
            image, device, num_images_per_prompt, do_classifier_free_guidance)
        dual_prompt_embeddings = paddle.concat(
            x=[prompt_embeds, image_embeddings], axis=1)
        prompt_types = 'text', 'image'
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_channels_latents = self.image_unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt, num_channels_latents, height,
            width, dual_prompt_embeddings.dtype, device, generator, latents)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        self.set_transformer_params(text_to_image_strength, prompt_types)
        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = paddle.concat(
                x=[latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t)
            noise_pred = self.image_unet(
                latent_model_input,
                t,
                encoder_hidden_states=dual_prompt_embeddings).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(chunks=2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents,
                                          **extra_step_kwargs).prev_sample
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
