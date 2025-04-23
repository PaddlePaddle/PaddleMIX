# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
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
import os
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

import paddle
from  ppdiffusers.transformers import ( # T5TokenizerFast,
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5Tokenizer
)

from ...image_processor import PipelineImageInput, VaeImageProcessor
from ...loaders import FromSingleFileMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from ...models.autoencoder_kl import AutoencoderKL
from ...models.controlnet_flux import FluxControlNetModel, FluxMultiControlNetModel
from ...models.transformer_flux import FluxTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging, replace_example_docstring
from ...utils.paddle_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import FluxPipelineOutput

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from controlnet_aux import CannyDetector
        >>> from diffusers import FluxControlNetPipeline
        >>> from diffusers.utils import load_image

        >>> pipe = FluxControlNetPipeline.from_pretrained(
        ...     "black-forest-labs/FLUX.1-Canny-dev", torch_dtype=torch.bfloat16
        ... ).to("cuda")

        >>> prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."
        >>> control_image = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png"
        ... )

        >>> processor = CannyDetector()
        >>> control_image = processor(
        ...     control_image, low_threshold=50, high_threshold=200, detect_resolution=1024, image_resolution=1024
        ... )

        >>> image = pipe(
        ...     prompt=prompt,
        ...     control_image=control_image,
        ...     height=1024,
        ...     width=1024,
        ...     num_inference_steps=50,
        ...     guidance_scale=30.0,
        ... ).images[0]
        >>> image.save("output.png")
        ```
"""

try:
    # paddle.incubate.jit.inference is available in paddle develop but not in paddle 3.0beta, so we add a try except.
    from paddle.incubate.jit import is_inference_mode
except:

    def is_inference_mode(func):
        return False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_latents(
    encoder_output: paddle.Tensor, generator: Optional[int] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[paddle.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps



class FluxControlNetPipeline(
    DiffusionPipeline,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
):
    """
    Flux Control Pipeline for text-to-image generation.

    Args:
        transformer (`FluxTransformer2DModel`): The transformer model for denoising.
        scheduler (`FlowMatchEulerDiscreteScheduler`): The scheduler to use for diffusion steps.
        vae (`AutoencoderKL`): Variational Auto-Encoder for encoding/decoding images.
        text_encoder (`CLIPTextModel`): CLIP model for text encoding.
        tokenizer (`CLIPTokenizer`): Tokenizer for the CLIP text model.
        text_encoder_2 (`T5EncoderModel`): T5-based encoder for text processing.
        tokenizer_2 (`T5TokenizerFast`): Tokenizer for the T5 text model.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "control_image"]


    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5Tokenizer,
        transformer: FluxTransformer2DModel,
        controlnet: Union[
            FluxControlNetModel, List[FluxControlNetModel],Tuple[FluxControlNetModel], FluxMultiControlNetModel
        ],
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            controlnet=controlnet,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.vae_latent_channels = self.vae.config.latent_channels if getattr(self, "vae", None) else 16
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2, vae_latent_channels=self.vae_latent_channels
        )
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        dtype: Optional[paddle.dtype] = None,
    ):
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pd",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pd").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not paddle.equal_all(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids, output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.astype(dtype=dtype)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.tile([1, num_images_per_prompt, 1])
        prompt_embeds = prompt_embeds.reshape([batch_size * num_images_per_prompt, seq_len, -1])

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
    ):

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pd",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pd").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not paddle.equal_all(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids, output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.astype(dtype=self.text_encoder.dtype)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.tile([1, num_images_per_prompt])
        prompt_embeds = prompt_embeds.reshape([batch_size * num_images_per_prompt, -1])

        return prompt_embeds
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[paddle.Tensor] = None,
        pooled_prompt_embeds: Optional[paddle.Tensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        # TODO
        # if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
        #     self._lora_scale = lora_scale

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = paddle.zeros([prompt_embeds.shape[1], 3]).astype(dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids
    
    def encode_image(self, image, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, paddle.Tensor):
            image = self.feature_extractor(image, return_tensors="pd").pixel_values

        image = image.astype(dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, axis=0)
        return image_embeds

    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, num_images_per_prompt
    ):
        image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.transformer.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.transformer.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.transformer.encoder_hid_proj.image_projection_layers
            ):
                single_image_embeds = self.encode_image(single_ip_adapter_image, 1)

                image_embeds.append(single_image_embeds[None, :])
        else:
            for single_image_embeds in ip_adapter_image_embeds:
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for i, single_image_embeds in enumerate(image_embeds):
            single_image_embeds = paddle.concat([single_image_embeds] * num_images_per_prompt, axis=0)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds
    

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")
        
        
    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, dtype):
        latent_image_ids = paddle.zeros([height, width, 3], dtype=dtype)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + paddle.arange(height, dtype=dtype)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + paddle.arange(width, dtype=dtype)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            [latent_image_id_height * latent_image_id_width, latent_image_id_channels]
        )

        return latent_image_ids.astype(dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.reshape([batch_size, num_channels_latents, height // 2, 2, width // 2, 2])
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape([batch_size, (height // 2) * (width // 2), num_channels_latents * 4])

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.reshape([batch_size, height // 2, width // 2, channels // 4, 2, 2])
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape([batch_size, channels // (2 * 2), height, width])

        return latents
    
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    # def prepare_latents(
    #     self,
    #     batch_size,
    #     num_channels_latents,
    #     height,
    #     width,
    #     dtype,
    #     generator,
    #     latents=None,
    # ):
    #     """
    #     Prepares the latent tensor for diffusion.
    #     """
    #     height = 2 * (height // (self.vae_scale_factor * 2))
    #     width = 2 * (width // (self.vae_scale_factor * 2))

    #     shape = (batch_size, num_channels_latents, height, width)

    #     if latents is not None:
    #         latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, dtype)
    #         return latents.astype(dtype), latent_image_ids
        
    #     if isinstance(generator, list) and len(generator) != batch_size:
    #         raise ValueError(
    #             f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
    #             f" size of {batch_size}. Make sure the batch size matches the length of the generators."
    #         )
        
        
    #     latents = randn_tensor(shape, generator=generator, dtype=dtype)
    #     latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

    #     latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, dtype)

    #     return latents, latent_image_ids
    
    
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        generator,
        latents=None,
    ):
        """
        Prepares the latent tensor for diffusion.
        """
        print(f"DEBUG: Input parameters - batch_size: {batch_size}, num_channels_latents: {num_channels_latents}")
        print(f"DEBUG: Original height/width: {height}/{width}, vae_scale_factor: {self.vae_scale_factor}")
        
        height = 2 * (height // (self.vae_scale_factor * 2))
        width = 2 * (width // (self.vae_scale_factor * 2))
        print(f"DEBUG: Adjusted height/width: {height}/{width}")

        shape = (batch_size, num_channels_latents, height, width)
        print(f"DEBUG: Target shape for latents: {shape}")

        if latents is not None:
            print(f"DEBUG: Using provided latents with shape: {latents.shape}")
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, dtype)
            return latents.astype(dtype), latent_image_ids
        
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        
        # 添加这个函数来检查randn_tensor的实现
        print(f"DEBUG: About to call randn_tensor with shape={shape}, dtype={dtype}")
        print(f"DEBUG: randn_tensor implementation: {randn_tensor.__module__}.{randn_tensor.__name__}")
        
        latents = randn_tensor(shape, generator=generator, dtype=dtype)
        print(f"DEBUG: Generated latents - shape: {latents.shape}, min: {latents.min().item():.4f}, max: {latents.max().item():.4f}, mean: {latents.mean().item():.4f}")
        
        # 保存原始latents副本
        latents_before_pack = latents.clone()
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        print(f"DEBUG: After packing - shape: {latents.shape}, min: {latents.min().item():.4f}, max: {latents.max().item():.4f}, mean: {latents.mean().item():.4f}")
        print(f"DEBUG: Packing changes: shape {latents_before_pack.shape} -> {latents.shape}")

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, dtype)
        print(f"DEBUG: Generated latent_image_ids - shape: {latent_image_ids.shape}")

        return latents, latent_image_ids
    
    

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if isinstance(image, paddle.Tensor):
            pass
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, axis=0)

        image = image.cast(dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = paddle.concat([image] * 2)

        return image
    
    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt
    

    @paddle.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        control_image: PipelineImageInput = None,
        control_mode: Optional[Union[int, List[int]]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        pooled_prompt_embeds: Optional[paddle.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[paddle.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[paddle.Tensor]] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            control_image (`paddle.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[paddle.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[paddle.Tensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `paddle.Tensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be accepted
                as an image. The dimensions of the output image defaults to `image`'s dimensions. If height and/or
                width are passed, `image` is resized accordingly. If multiple ControlNets are specified in `init`,
                images must be passed as a list such that each element of the list can be correctly batched for input
                to a single ControlNet.
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            control_mode (`int` or `List[int]`,, *optional*, defaults to None):
                The control mode when applying ControlNet-Union.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`paddle.Generator` or `List[paddle.Generator]`, *optional*):
                One or a list of paddle generators to make generation deterministic.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[paddle.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            negative_ip_adapter_image:
                (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            negative_ip_adapter_image_embeds (`List[paddle.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """
        

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        print(f"DEBUG: Initial dimensions - height: {height}, width: {width}")

        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
            print(f"DEBUG: Adjusted control_guidance_start: {control_guidance_start} to match end: {control_guidance_end}")
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
            print(f"DEBUG: Adjusted control_guidance_end: {control_guidance_end} to match start: {control_guidance_start}")
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(self.controlnet.nets) if hasattr(self.controlnet, 'nets') else 1
            print(f"DEBUG: Calculated mult: {mult}, controlnet type: {type(self.controlnet).__name__}")
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )
            print(f"DEBUG: Final guidance ranges - start: {control_guidance_start}, end: {control_guidance_end}")

        # 1. Check inputs. Raise error if not correct
        print("DEBUG: Checking inputs...")
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
        print("DEBUG: Input check passed")

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        print(f"DEBUG: Config - guidance_scale: {guidance_scale}, true_cfg_scale: {true_cfg_scale}")

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        print(f"DEBUG: Determined batch_size: {batch_size}, num_images_per_prompt: {num_images_per_prompt}")

        dtype = self.transformer.dtype
        print(f"DEBUG: Using dtype: {dtype}")

        # 3. Prepare text embeddings
        print("DEBUG: Preparing text embeddings...")
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        print(f"DEBUG: LoRA scale: {lora_scale}")
        do_true_cfg = true_cfg_scale > 1 and negative_prompt is not None
        print(f"DEBUG: do_true_cfg: {do_true_cfg}, true_cfg_scale: {true_cfg_scale}")

        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        print(f"DEBUG: Prompt encoding stats - shape: {prompt_embeds.shape}, min: {prompt_embeds.min().item():.4f}, max: {prompt_embeds.max().item():.4f}, mean: {prompt_embeds.mean().item():.4f}")
        if pooled_prompt_embeds is not None:
            print(f"DEBUG: Pooled embeds stats - shape: {pooled_prompt_embeds.shape}, min: {pooled_prompt_embeds.min().item():.4f}, max: {pooled_prompt_embeds.max().item():.4f}, mean: {pooled_prompt_embeds.mean().item():.4f}")

        if do_true_cfg:
            print("DEBUG: Encoding negative prompt...")
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                _,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
            print(f"DEBUG: Negative prompt stats - shape: {negative_prompt_embeds.shape}, min: {negative_prompt_embeds.min().item():.4f}, max: {negative_prompt_embeds.max().item():.4f}, mean: {negative_prompt_embeds.mean().item():.4f}")
            if negative_pooled_prompt_embeds is not None:
                print(f"DEBUG: Negative pooled stats - shape: {negative_pooled_prompt_embeds.shape}, min: {negative_pooled_prompt_embeds.min().item():.4f}, max: {negative_pooled_prompt_embeds.max().item():.4f}, mean: {negative_pooled_prompt_embeds.mean().item():.4f}")

        # 3. Prepare control image
        print("DEBUG: Preparing control image...")
        num_channels_latents = self.transformer.config.in_channels // 4
        print(f"DEBUG: num_channels_latents: {num_channels_latents}")

        print(f"DEBUG: controlnet type: {type(self.controlnet).__name__}")
        if isinstance(self.controlnet, FluxControlNetModel):
            print("DEBUG: Using FluxControlNetModel")
            control_image = self.prepare_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                dtype=self.vae.dtype,
            )
            print(f"DEBUG: Prepared control image - shape: {control_image.shape}, min: {control_image.min().item():.4f}, max: {control_image.max().item():.4f}, mean: {control_image.mean().item():.4f}")
            height, width = control_image.shape[-2:]
            print(f"DEBUG: Updated dimensions - height: {height}, width: {width}")

            # xlab controlnet has a input_hint_block and instantx controlnet does not
            controlnet_blocks_repeat = False if self.controlnet.input_hint_block is None else True
            print(f"DEBUG: controlnet_blocks_repeat: {controlnet_blocks_repeat}, input_hint_block exists: {self.controlnet.input_hint_block is not None}")
            
            if self.controlnet.input_hint_block is None:
                print("DEBUG: No input_hint_block, encoding control image with VAE...")
                # vae encode
                vae_output = self.vae.encode(control_image)
                control_image_before = control_image
                control_image = retrieve_latents(vae_output, generator=generator)
                print(f"DEBUG: VAE encode - input mean: {control_image_before.mean().item():.4f}, output mean: {control_image.mean().item():.4f}")
                control_image_before = control_image.clone()
                control_image = (control_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                print(f"DEBUG: Latent processing - before min/max/mean: {control_image_before.min().item():.4f}/{control_image_before.max().item():.4f}/{control_image_before.mean().item():.4f}")
                print(f"DEBUG: Latent processing - after min/max/mean: {control_image.min().item():.4f}/{control_image.max().item():.4f}/{control_image.mean().item():.4f}")
                print(f"DEBUG: Used shift_factor: {self.vae.config.shift_factor}, scaling_factor: {self.vae.config.scaling_factor}")

                # pack
                height_control_image, width_control_image = control_image.shape[2:]
                print(f"DEBUG: Control image dimensions before packing: {height_control_image}x{width_control_image}")
                control_image_before = control_image.clone()
                control_image = self._pack_latents(
                    control_image,
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height_control_image,
                    width_control_image,
                )
                print(f"DEBUG: Packing - before shape: {control_image_before.shape}, after shape: {control_image.shape}")
                print(f"DEBUG: Packing - before/after mean: {control_image_before.mean().item():.4f}/{control_image.mean().item():.4f}")

            # Here we ensure that `control_mode` has the same length as the control_image.
            if control_mode is not None:
                print(f"DEBUG: Setting control_mode: {control_mode}")
                if not isinstance(control_mode, int):
                    raise ValueError(" For `FluxControlNet`, `control_mode` should be an `int` or `None`")
                control_mode = paddle.to_tensor(control_mode, dtype=paddle.int64)
                control_mode = control_mode.reshape([-1, 1]).expand([control_image.shape[0], 1])
                print(f"DEBUG: Final control_mode: shape={control_mode.shape}, values={control_mode.tolist()[:5]}{'...' if control_mode.shape[0] > 5 else ''}")

        elif isinstance(self.controlnet, FluxMultiControlNetModel):
            print("DEBUG: Using FluxMultiControlNetModel with {0} nets".format(len(self.controlnet.nets)))
            control_images = []
            # xlab controlnet has a input_hint_block and instantx controlnet does not
            controlnet_blocks_repeat = False if self.controlnet.nets[0].input_hint_block is None else True
            print(f"DEBUG: controlnet_blocks_repeat: {controlnet_blocks_repeat}")
            
            print(f"DEBUG: Processing {len(control_image)} control images")
            for i, control_image_ in enumerate(control_image):
                print(f"DEBUG: Processing control image {i}...")
                control_image_ = self.prepare_image(
                    image=control_image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    dtype=self.vae.dtype,
                )
                print(f"DEBUG: Control image {i} - shape: {control_image_.shape}, min: {control_image_.min().item():.4f}, max: {control_image_.max().item():.4f}, mean: {control_image_.mean().item():.4f}")
                height, width = control_image_.shape[-2:]

                if self.controlnet.nets[0].input_hint_block is None:
                    print(f"DEBUG: Encoding control image {i} with VAE...")
                    # vae encode
                    vae_output = self.vae.encode(control_image_)
                    control_image_before = control_image_
                    control_image_ = retrieve_latents(vae_output, generator=generator)
                    print(f"DEBUG: VAE encode {i} - input mean: {control_image_before.mean().item():.4f}, output mean: {control_image_.mean().item():.4f}")
                    
                    control_image_before = control_image_.clone()
                    control_image_ = (control_image_ - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                    print(f"DEBUG: Latent processing {i} - before min/max/mean: {control_image_before.min().item():.4f}/{control_image_before.max().item():.4f}/{control_image_before.mean().item():.4f}")
                    print(f"DEBUG: Latent processing {i} - after min/max/mean: {control_image_.min().item():.4f}/{control_image_.max().item():.4f}/{control_image_.mean().item():.4f}")

                    # pack
                    height_control_image, width_control_image = control_image_.shape[2:]
                    control_image_before = control_image_.clone()
                    control_image_ = self._pack_latents(
                        control_image_,
                        batch_size * num_images_per_prompt,
                        num_channels_latents,
                        height_control_image,
                        width_control_image,
                    )
                    print(f"DEBUG: Packing {i} - before shape: {control_image_before.shape}, after shape: {control_image_.shape}")
                    print(f"DEBUG: Packing {i} - before/after mean: {control_image_before.mean().item():.4f}/{control_image_.mean().item():.4f}")
                    
                control_images.append(control_image_)

            control_image = control_images
            print(f"DEBUG: Final control_image is a list of {len(control_image)} tensors")

            # Here we ensure that `control_mode` has the same length as the control_image.
            print(f"DEBUG: Original control_mode: {control_mode}")
            if isinstance(control_mode, list) and len(control_mode) != len(control_image):
                raise ValueError(
                    "For Multi-ControlNet, `control_mode` must be a list of the same "
                    + " length as the number of controlnets (control images) specified"
                )
            if not isinstance(control_mode, list):
                control_mode = [control_mode] * len(control_image)
                print(f"DEBUG: Extended control_mode to list: {control_mode}")
            
            # set control mode
            control_modes = []
            for i, cmode in enumerate(control_mode):
                print(f"DEBUG: Processing control mode {i}: {cmode}")
                if cmode is None:
                    cmode = -1
                    print(f"DEBUG: Set None control mode to -1")
                control_mode_tensor = paddle.to_tensor(cmode).expand([control_images[0].shape[0]]).astype(paddle.int64)
                print(f"DEBUG: Control mode {i}: shape={control_mode_tensor.shape}, values={control_mode_tensor.tolist()[:5]}{'...' if control_mode_tensor.shape[0] > 5 else ''}")
                control_modes.append(control_mode_tensor)
            control_mode = control_modes

        # 4. Prepare latent variables
        print("DEBUG: Preparing latent variables...")
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )
        print(f"DEBUG: Prepared latents - shape: {latents.shape}, min: {latents.min().item():.4f}, max: {latents.max().item():.4f}, mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}")
        if latent_image_ids is not None:
            print(f"DEBUG: latent_image_ids - shape: {latent_image_ids.shape}, values: {latent_image_ids.tolist()[:10]}{'...' if latent_image_ids.shape[0] > 10 else ''}")

        # 5. Prepare timesteps
        print("DEBUG: Preparing timesteps...")
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        print(f"DEBUG: Sigmas range: [{sigmas[0]:.4f}, ..., {sigmas[-1]:.4f}], length: {len(sigmas)}")

        image_seq_len = latents.shape[1]
        print(f"DEBUG: image_seq_len: {image_seq_len}")

        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        print(f"DEBUG: Calculated mu: {mu:.4f}")

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            mu=mu,
        )
        print(f"DEBUG: Retrieved {len(timesteps)} timesteps, num_inference_steps: {num_inference_steps}")
        print(f"DEBUG: Timesteps range: [{timesteps[0].item():.2f}, {timesteps[min(5, len(timesteps)-1)].item():.2f}, ..., {timesteps[-1].item():.2f}]")

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        print(f"DEBUG: num_warmup_steps: {num_warmup_steps}, total timesteps: {self._num_timesteps}")

        # 6. Create tensor stating which controlnets to keep
        print("DEBUG: Creating controlnet_keep tensor...")
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(self.controlnet, FluxControlNetModel) else keeps)

        print(f"DEBUG: controlnet_keep samples - start: {controlnet_keep[0]}, middle: {controlnet_keep[len(controlnet_keep)//2]}, end: {controlnet_keep[-1]}")

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            print("DEBUG: Creating negative_ip_adapter_image as zeros")
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            print("DEBUG: Creating ip_adapter_image as zeros")
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}
            print("DEBUG: Created empty joint_attention_kwargs")
        else:
            print(f"DEBUG: joint_attention_kwargs keys: {list(self.joint_attention_kwargs.keys())}")

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            print("DEBUG: Preparing IP adapter image embeds...")
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                batch_size * num_images_per_prompt,
            )
            if isinstance(image_embeds, list):
                print(f"DEBUG: image_embeds is a list of {len(image_embeds)} tensors")
                for i, embed in enumerate(image_embeds):
                    if hasattr(embed, 'shape'):
                        print(f"DEBUG: image_embeds[{i}] - shape: {embed.shape}, min: {embed.min().item():.4f}, max: {embed.max().item():.4f}, mean: {embed.mean().item():.4f}")
            else:
                print(f"DEBUG: image_embeds - shape: {image_embeds.shape}, min: {image_embeds.min().item():.4f}, max: {image_embeds.max().item():.4f}, mean: {image_embeds.mean().item():.4f}")
            
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            print("DEBUG: Preparing negative IP adapter image embeds...")
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                batch_size * num_images_per_prompt,
            )
            if isinstance(negative_image_embeds, list):
                print(f"DEBUG: negative_image_embeds is a list of {len(negative_image_embeds)} tensors")
                for i, embed in enumerate(negative_image_embeds):
                    if hasattr(embed, 'shape'):
                        print(f"DEBUG: negative_image_embeds[{i}] - shape: {embed.shape}, min: {embed.min().item():.4f}, max: {embed.max().item():.4f}, mean: {embed.mean().item():.4f}")
            else:
                print(f"DEBUG: negative_image_embeds - shape: {negative_image_embeds.shape}, min: {negative_image_embeds.min().item():.4f}, max: {negative_image_embeds.max().item():.4f}, mean: {negative_image_embeds.mean().item():.4f}")
        

        # 7. Denoising loop
        print("DEBUG: Starting denoising loop...")
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    print("DEBUG: Interruption detected, continuing to next step")
                    continue

                print(f"\nDEBUG: Denoising step {i+1}/{len(timesteps)}, t={t.item():.4f}")
                
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                    print("DEBUG: Added image_embeds to joint_attention_kwargs")
                    
                # broadcast to batch dimension
                timestep = t.expand([latents.shape[0]]).astype(latents.dtype)
                print(f"DEBUG: Timestep value: {t.item():.4f}, expanded shape: {timestep.shape}")

                if isinstance(self.controlnet, FluxMultiControlNetModel):
                    use_guidance = self.controlnet.nets[0].config.guidance_embeds
                else:
                    use_guidance = self.controlnet.config.guidance_embeds
                print(f"DEBUG: use_guidance: {use_guidance}")

                guidance = paddle.to_tensor([guidance_scale]) if use_guidance else None
                if guidance is not None:
                    guidance = guidance.expand([latents.shape[0]])
                    print(f"DEBUG: Created guidance tensor: {guidance.tolist()[:5]} shape: {guidance.shape}")

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                    print(f"DEBUG: cond_scale (list): {[f'{cs:.4f}' for cs in cond_scale]}")
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]
                    print(f"DEBUG: cond_scale (single): {cond_scale:.4f} = {controlnet_cond_scale} * {controlnet_keep[i]}")

                # controlnet
                print(f"DEBUG: Calling controlnet with latents - min: {latents.min().item():.4f}, max: {latents.max().item():.4f}, mean: {latents.mean().item():.4f}")
                try:
                    controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
                        hidden_states=latents,
                        controlnet_cond=control_image,
                        controlnet_mode=control_mode,
                        conditioning_scale=cond_scale,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )
                    print(f"DEBUG: controlnet call successful")
                    
                    # Print stats about controlnet outputs
                    if isinstance(controlnet_block_samples, list) and len(controlnet_block_samples) > 0:
                        print(f"DEBUG: controlnet_block_samples length: {len(controlnet_block_samples)}")
                        for j, block in enumerate(controlnet_block_samples[:2]):  # Print first 2 for brevity
                            print(f"DEBUG: block[{j}] - shape: {block.shape}, min: {block.min().item():.4f}, max: {block.max().item():.4f}, mean: {block.mean().item():.4f}")
                        if len(controlnet_block_samples) > 2:
                            print(f"DEBUG: ... and {len(controlnet_block_samples)-2} more blocks")
                    
                    if controlnet_single_block_samples is not None:
                        if isinstance(controlnet_single_block_samples, list):
                            print(f"DEBUG: controlnet_single_block_samples length: {len(controlnet_single_block_samples)}")
                            for j, block in enumerate(controlnet_single_block_samples[:2]):  # Print first 2 for brevity
                                print(f"DEBUG: single_block[{j}] - shape: {block.shape}, min: {block.min().item():.4f}, max: {block.max().item():.4f}, mean: {block.mean().item():.4f}")
                        else:
                            print(f"DEBUG: single_block - shape: {controlnet_single_block_samples.shape}, min: {controlnet_single_block_samples.min().item():.4f}, max: {controlnet_single_block_samples.max().item():.4f}")
                except Exception as e:
                    print(f"DEBUG: Error in controlnet call: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise

                guidance = paddle.to_tensor([guidance_scale]) if self.transformer.config.guidance_embeds else None
                if guidance is not None:
                    guidance = guidance.expand([latents.shape[0]])
                    print(f"DEBUG: Updated guidance tensor for transformer: {guidance.tolist()[:3]}...")

                print("DEBUG: Calling transformer...")
                try:
                    latents_before = latents.clone()
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        controlnet_blocks_repeat=controlnet_blocks_repeat,
                    )[0]
                    print(f"DEBUG: transformer call successful")
                    print(f"DEBUG: noise_pred - shape: {noise_pred.shape}, min: {noise_pred.min().item():.4f}, max: {noise_pred.max().item():.4f}, mean: {noise_pred.mean().item():.4f}, std: {noise_pred.std().item():.4f}")
                except Exception as e:
                    print(f"DEBUG: Error in transformer call: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise

                if do_true_cfg:
                    print("DEBUG: Applying true classifier-free guidance...")
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                        print("DEBUG: Set negative_image_embeds in joint_attention_kwargs")
                        
                    print("DEBUG: Calling transformer for negative prompt...")
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        controlnet_blocks_repeat=controlnet_blocks_repeat,
                    )[0]
                    print(f"DEBUG: neg_noise_pred - min: {neg_noise_pred.min().item():.4f}, max: {neg_noise_pred.max().item():.4f}, mean: {neg_noise_pred.mean().item():.4f}")
                    
                    noise_pred_before = noise_pred.clone()
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    print(f"DEBUG: CFG - before min/max/mean: {noise_pred_before.min().item():.4f}/{noise_pred_before.max().item():.4f}/{noise_pred_before.mean().item():.4f}")
                    print(f"DEBUG: CFG - after min/max/mean: {noise_pred.min().item():.4f}/{noise_pred.max().item():.4f}/{noise_pred.mean().item():.4f}")

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                print(f"DEBUG: Calling scheduler.step with t: {t.item():.4f}, latents mean: {latents.mean().item():.4f}")
                
                latents_before_step = latents.clone()
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                print(f"DEBUG: Scheduler step - before min/max/mean: {latents_before_step.min().item():.4f}/{latents_before_step.max().item():.4f}/{latents_before_step.mean().item():.4f}")
                print(f"DEBUG: Scheduler step - after min/max/mean: {latents.min().item():.4f}/{latents.max().item():.4f}/{latents.mean().item():.4f}")

                if callback_on_step_end is not None:
                    print("DEBUG: Running callback_on_step_end...")
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    control_image = callback_outputs.pop("control_image", control_image)
                    print(f"DEBUG: After callback - latents mean: {latents.mean().item():.4f}")

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        print("DEBUG: Denoising complete")

        if output_type == "latent":
            print("DEBUG: Returning latents directly")
            image = latents
        else:
            print("DEBUG: Post-processing latents...")
            latents_before = latents.clone()
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            print(f"DEBUG: Unpacked latents - before shape: {latents_before.shape}, after shape: {latents.shape}")
            print(f"DEBUG: Unpacked stats - min: {latents.min().item():.4f}, max: {latents.max().item():.4f}, mean: {latents.mean().item():.4f}")
            
            latents_before = latents.clone()
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            print(f"DEBUG: Normalized latents - before min/max/mean: {latents_before.min().item():.4f}/{latents_before.max().item():.4f}/{latents_before.mean().item():.4f}")
            print(f"DEBUG: Normalized latents - after min/max/mean: {latents.min().item():.4f}/{latents.max().item():.4f}/{latents.mean().item():.4f}")
            print(f"DEBUG: Used scaling_factor: {self.vae.config.scaling_factor}, shift_factor: {self.vae.config.shift_factor}")
            
            print("DEBUG: Decoding with VAE...")
            image = self.vae.decode(latents, return_dict=False)[0]
            print(f"DEBUG: Decoded image - shape: {image.shape}, min: {image.min().item():.4f}, max: {image.max().item():.4f}, mean: {image.mean().item():.4f}")
            
            print(f"DEBUG: Post-processing to {output_type}")
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        print("DEBUG: Offloading models...")
        self.maybe_free_model_hooks()
        print("DEBUG: Pipeline execution complete")

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)
