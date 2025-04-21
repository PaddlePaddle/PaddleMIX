# Copyright 2024 Genmo and The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import paddle

import numpy as np

from ppdiffusers.transformers import T5EncoderModel, T5Tokenizer

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...loaders.Mochi1LoraLoader import Mochi1LoraLoaderMixin
from ...models import AutoencoderKLMochi, MochiTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import (
    logging,
    replace_example_docstring,
)
from ...utils.paddle_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import MochiPipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import MochiPipeline
        >>> from diffusers.utils import export_to_video

        >>> pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", torch_dtype=torch.bfloat16)
        >>> pipe.enable_model_cpu_offload()
        >>> pipe.enable_vae_tiling()
        >>> prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
        >>> frames = pipe(prompt, num_inference_steps=28, guidance_scale=3.5).frames[0]
        >>> export_to_video(frames, "mochi.mp4")
        ```
"""


def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps**2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps**2)
    const = quadratic_coef * (linear_steps**2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i**2) + linear_coef * i + const for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
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


class MochiPipeline(DiffusionPipeline, Mochi1LoraLoaderMixin):
    r"""
    The mochi pipeline for text-to-video generation.

    Reference: https://github.com/genmoai/models

    Args:
        transformer ([`MochiTransformer3DModel`]):
            Conditional Transformer architecture to denoise the encoded video latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKLMochi`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLMochi,
        text_encoder: T5EncoderModel,
        tokenizer: T5Tokenizer,
        transformer: MochiTransformer3DModel,
        force_zeros_for_empty_prompt: bool = False,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        # TODO: determine these scaling factors from model parameters
        self.vae_spatial_scale_factor = 8
        self.vae_temporal_scale_factor = 6
        self.patch_size = 2

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 256
        )
        self.default_height = 480
        self.default_width = 848
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 256,
        dtype: Optional[paddle.dtype] = None,
    ):
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pd",
        )

        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.astype("bool")

        if self.config.force_zeros_for_empty_prompt and (prompt == "" or prompt[-1] == ""):
            text_input_ids = paddle.zeros_like(text_input_ids)
            prompt_attention_mask = paddle.zeros_like(prompt_attention_mask, dtype="bool")

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pd").input_ids

        if (
            untruncated_ids.shape[-1] >= text_input_ids.shape[-1]
            and not paddle.equal_all(text_input_ids, untruncated_ids)
        ):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids, attention_mask=prompt_attention_mask)[0]
        prompt_embeds = prompt_embeds.cast(dtype)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.tile([1, num_videos_per_prompt, 1])
        prompt_embeds = prompt_embeds.reshape([batch_size * num_videos_per_prompt, seq_len, -1])

        prompt_attention_mask = prompt_attention_mask.reshape([batch_size, -1])
        prompt_attention_mask = prompt_attention_mask.tile([num_videos_per_prompt, 1])

        return prompt_embeds, prompt_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        prompt_attention_mask: Optional[paddle.Tensor] = None,
        negative_prompt_attention_mask: Optional[paddle.Tensor] = None,
        max_sequence_length: int = 256,
        dtype: Optional[paddle.dtype] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                dtype=dtype,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_on_step_end_tensor_inputs=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

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
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Must provide `prompt_attention_mask` when specifying `prompt_embeds`.")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError("Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`.")

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

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

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        num_frames,
        dtype,
        generator,
        latents=None,
    ):
        height = height // self.vae_spatial_scale_factor
        width = width // self.vae_spatial_scale_factor
        num_frames = (num_frames - 1) // self.vae_temporal_scale_factor + 1

        shape = (batch_size, num_channels_latents, num_frames, height, width)

        if latents is not None:
            return latents.cast(dtype)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, dtype=dtype)
        latents = latents.cast(dtype)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps
    
    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt
    
    
    # @paddle.no_grad()
    # # @replace_example_docstring(EXAMPLE_DOC_STRING)
    # def __call__(
    #     self,
    #     prompt: Union[str, List[str]] = None,
    #     negative_prompt: Optional[Union[str, List[str]]] = None,
    #     height: Optional[int] = None,
    #     width: Optional[int] = None,
    #     num_frames: int = 19,
    #     num_inference_steps: int = 64,
    #     timesteps: List[int] = None,
    #     guidance_scale: float = 4.5,
    #     num_videos_per_prompt: Optional[int] = 1,
    #     generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
    #     latents: Optional[paddle.Tensor] = None,
    #     prompt_embeds: Optional[paddle.Tensor] = None,
    #     prompt_attention_mask: Optional[paddle.Tensor] = None,
    #     negative_prompt_embeds: Optional[paddle.Tensor] = None,
    #     negative_prompt_attention_mask: Optional[paddle.Tensor] = None,
    #     output_type: Optional[str] = "pil",
    #     return_dict: bool = True,
    #     callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    #     callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    #     max_sequence_length: int = 256,
    # ):
    #     # ... (docstring remains the same)

    #     print("====== 管道执行开始 ======")
    #     print(f"传入的 prompt 类型: {type(prompt)}")
    #     print(f"当前 transformer 的默认数据类型: {self.transformer._dtype}")




    #     if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
    #         callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    #     height = height or self.default_height
    #     width = width or self.default_width

    #     # 1. Check inputs
    #     self.check_inputs(
    #         prompt=prompt,
    #         height=height,
    #         width=width,
    #         callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
    #         prompt_embeds=prompt_embeds,
    #         negative_prompt_embeds=negative_prompt_embeds,
    #         prompt_attention_mask=prompt_attention_mask,
    #         negative_prompt_attention_mask=negative_prompt_attention_mask,
    #     )
        


    #     self._guidance_scale = guidance_scale
    #     self._current_timestep = None
    #     self._interrupt = False

    #     # 2. Define call parameters
    #     if prompt is not None and isinstance(prompt, str):
    #         batch_size = 1
    #     elif prompt is not None and isinstance(prompt, list):
    #         batch_size = len(prompt)
    #     else:
    #         batch_size = prompt_embeds.shape[0]

    #     # 3. Prepare text embeddings
    #     (
    #         prompt_embeds,
    #         prompt_attention_mask,
    #         negative_prompt_embeds,
    #         negative_prompt_attention_mask,
    #     ) = self.encode_prompt(
    #         prompt=prompt,
    #         negative_prompt=negative_prompt,
    #         do_classifier_free_guidance=self.do_classifier_free_guidance,
    #         num_videos_per_prompt=num_videos_per_prompt,
    #         prompt_embeds=prompt_embeds,
    #         negative_prompt_embeds=negative_prompt_embeds,
    #         prompt_attention_mask=prompt_attention_mask,
    #         negative_prompt_attention_mask=negative_prompt_attention_mask,
    #         max_sequence_length=max_sequence_length,
    #     )
        
    #     # encode_prompt 之后
    #     print("\n====== 编码后的提示词 ======")
    #     debug_print("prompt_embeds", prompt_embeds, detailed=True)
    #     debug_print("prompt_attention_mask", prompt_attention_mask)
    #     debug_print("negative_prompt_embeds", negative_prompt_embeds, detailed=True)
    #     debug_print("negative_prompt_attention_mask", negative_prompt_attention_mask)


    #     # 4. Prepare latent variables
    #     num_channels_latents = self.transformer.config.in_channels
    #     latents = self.prepare_latents(
    #         batch_size * num_videos_per_prompt,
    #         num_channels_latents,
    #         height,
    #         width,
    #         num_frames,
    #         prompt_embeds.dtype,
    #         generator,
    #         latents,
    #     )
        
    #     # prepare_latents之后
    #     print("\n====== 初始化的latents ======")
    #     debug_print("latents", latents, detailed=True, percentiles=True)


    #     if self.do_classifier_free_guidance:
    #         prompt_embeds = paddle.concat([negative_prompt_embeds, prompt_embeds], axis=0)
    #         prompt_attention_mask = paddle.concat([negative_prompt_attention_mask, prompt_attention_mask], axis=0)
        
        
    #     # 5. Prepare timestep
    #     threshold_noise = 0.025
    #     sigmas = linear_quadratic_schedule(num_inference_steps, threshold_noise)
    #     sigmas = np.array(sigmas)

    #     timesteps, num_inference_steps = retrieve_timesteps(
    #         self.scheduler,
    #         num_inference_steps,
    #         timesteps,
    #         sigmas,
    #     )
    #     num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    #     self._num_timesteps = len(timesteps)

    #     # 6. Denoising loop
    #     with self.progress_bar(total=num_inference_steps) as progress_bar:
    #         for i, t in enumerate(timesteps):
    #             if self.interrupt:
    #                 continue
            

    #             self._current_timestep = 1000 - t
    #             latent_model_input = paddle.concat([latents] * 2) if self.do_classifier_free_guidance else latents
    #             timestep = paddle.full((latent_model_input.shape[0],), t, dtype=latents.dtype)
                
    #             noise_pred = self.transformer(
    #                 hidden_states=latent_model_input,
    #                 encoder_hidden_states=prompt_embeds,
    #                 timestep=timestep,
    #                 encoder_attention_mask=prompt_attention_mask,
    #                 return_dict=False,
    #             )[0]
                
    #             # Transformer输出后
    #             print("\n====== Transformer输出 ======")
    #             debug_print("noise_pred (原始)", noise_pred, detailed=True, percentiles=True)

    #             # 类型转换
    #             noise_pred_before = noise_pred
    #             noise_pred = noise_pred.cast('float32')
    #             print(f"类型转换: {noise_pred_before.dtype} -> {noise_pred.dtype}")
    #             debug_print("noise_pred (转换后)", noise_pred)

    #             if self.do_classifier_free_guidance:
    #                 noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

    #                 # 在这里添加CFG打印代码，这里是正确的位置
    #                 print(f"\n====== CFG 组件详细信息 ======")
    #                 print(f"无条件预测: min={noise_pred_uncond.min().item():.4f}, max={noise_pred_uncond.max().item():.4f}, mean={noise_pred_uncond.mean().item():.4f}")
    #                 print(f"条件预测: min={noise_pred_text.min().item():.4f}, max={noise_pred_text.max().item():.4f}, mean={noise_pred_text.mean().item():.4f}")
    #                 print(f"差值统计: min={(noise_pred_text - noise_pred_uncond).min().item():.4f}, max={(noise_pred_text - noise_pred_uncond).max().item():.4f}")
                    
    #                 # 计算CFG
    #                 print(f"引导尺度: {self.guidance_scale}")
                    
    #                 # 检查是否有极端值
    #                 diff = noise_pred_text - noise_pred_uncond
    #                 extreme_diff = paddle.logical_or(diff > 10.0, diff < -10.0)
    #                 if paddle.any(extreme_diff):
    #                     print(f"⚠️ 检测到极端差值! 超过范围±10的元素比例: {paddle.sum(extreme_diff).item() / diff.numel():.6f}")
                    
    #                 # 执行CFG计算
    #                 noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
    #                 # CFG计算后检查
    #                 print(f"CFG后噪声预测: min={noise_pred.min().item():.4f}, max={noise_pred.max().item():.4f}, mean={noise_pred.mean().item():.4f}")
                    
    #             # Scheduler步骤前
    #             print(f"\n====== Scheduler步骤前 (步骤 {i}) ======")
    #             print(f"噪声预测: min={noise_pred.min().item():.4f}, max={noise_pred.max().item():.4f}, mean={noise_pred.mean().item():.4f}")
    #             print(f"当前latents: min={latents.min().item():.4f}, max={latents.max().item():.4f}, mean={latents.mean().item():.4f}")
    #             print(f"时间步: t={t}")

    #             latents_dtype = latents.dtype
    #             latents = self.scheduler.step(noise_pred, t, latents.cast('float32'), return_dict=False)[0]
    #             latents = latents.cast(latents_dtype)
                
    #             # 添加这些调试代码
    #             # Scheduler步骤后
    #             print(f"====== Scheduler步骤后 (步骤 {i}) ======")
    #             print(f"更新后latents: min={latents.min().item():.4f}, max={latents.max().item():.4f}, mean={latents.mean().item():.4f}")

    #             # 检查是否有数值异常增长
    #             if latents.max().item() > 10.0 or latents.min().item() < -10.0:
    #                 print(f"⚠️ 检测到latents数值异常! 超过±10范围")

    #             if callback_on_step_end is not None:
    #                 callback_kwargs = {}
    #                 for k in callback_on_step_end_tensor_inputs:
    #                     callback_kwargs[k] = locals()[k]
    #                 callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

    #                 latents = callback_outputs.pop("latents", latents)
    #                 prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    
    #             # 在去噪循环中
    #             print(f"Step {i}/{len(timesteps)}, noise_pred stats: min={noise_pred.min().item()}, max={noise_pred.max().item()}")
    #             print(f"After scheduler step: min={latents.min().item()}, max={latents.max().item()}")
                
    #             # 每10步或关键步骤进行详细分析
    #             if i % 10 == 0 or i == len(timesteps) - 1 or (i > 0 and (latents.max().item() > 15.0 or latents.min().item() < -15.0)):
    #                 print(f"\n====== 步骤 {i} 详细分析 ======")
    #                 # 分析latents的分布情况
    #                 percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    #                 latents_flat = latents.reshape([-1])
    #                 for p in percentiles:
    #                     q = float(p) / 100.0
    #                     val = paddle.quantile(latents_flat, q).item()
    #                     print(f"latents {p}% 分位数: {val:.4f}")
                    
    #                 # 检查是否有NaN或Inf
    #                 if paddle.isnan(latents).any().item() or paddle.isinf(latents).any().item():
    #                     print("⚠️ 检测到NaN或Inf值!")
                        
    #                 # 保存当前latent可视化
    #                 latent_frame = latents[0, :, 0]
    #                 save_latent_visualization(latent_frame, f"critical_latent_step_{i}")


    #             if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
    #                 progress_bar.update()

    #     self._current_timestep = None

    #     if output_type == "latent":
    #         video = latents
    #     else:
    #         has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
    #         has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
    #         if has_latents_mean and has_latents_std:
    #             latents_mean = paddle.to_tensor(self.vae.config.latents_mean).reshape([1, 12, 1, 1, 1]).astype(latents.dtype)
    #             latents_std = paddle.to_tensor(self.vae.config.latents_std).reshape([1, 12, 1, 1, 1]).astype(latents.dtype)
    #             latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
    #         else:
    #             latents = latents / self.vae.config.scaling_factor
                
    #         # VAE解码前
    #         print("\n====== VAE解码前 ======")
    #         print(f"解码前latents统计: min={latents.min().item():.4f}, max={latents.max().item():.4f}, mean={latents.mean().item():.4f}")
    #         print("VAE配置检查:")
    #         print(f"scaling_factor: {self.vae.config.scaling_factor}")
    #         if hasattr(self.vae.config, "latents_mean"):
    #             print(f"latents_mean: {self.vae.config.latents_mean}")
    #         if hasattr(self.vae.config, "latents_std"):
    #             print(f"latents_std: {self.vae.config.latents_std}")

    #         video = self.vae.decode(latents, return_dict=False)[0]
            
    #         # 添加:
    #         print("\n====== VAE解码后 ======")
    #         if output_type == "pil":
    #             first_frame = video[0][0]
    #             print(f"输出视频第一帧类型: {type(first_frame)}, 尺寸: {first_frame.size if hasattr(first_frame, 'size') else 'unknown'}")
    #             save_video_frames(video[0], "final_video")  # 假设video[0]是第一个生成的视频的所有帧
    #         else:
    #             print(f"输出视频类型: {type(video)}, 形状: {video.shape if hasattr(video, 'shape') else 'unknown'}")
    #             # 尝试适应可能的输出格式
    #             if hasattr(video, "shape"):
    #                 tensor_stat = f"min={video.min().item() if hasattr(video, 'min') else 'N/A'}, max={video.max().item() if hasattr(video, 'max') else 'N/A'}"
    #                 print(f"视频tensor统计: {tensor_stat}")
    #             save_video_frames(video, "final_video")
            
    #         video = self.video_processor.postprocess_video(video, output_type=output_type)

    #     # Offload all models
    #     self.maybe_free_model_hooks()

    #     if not return_dict:
    #         return (video,)

    #     return MochiPipelineOutput(frames=video)
    
    
    
    @paddle.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 19,
        num_inference_steps: int = 64,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        prompt_attention_mask: Optional[paddle.Tensor] = None,
        negative_prompt_embeds: Optional[paddle.Tensor] = None,
        negative_prompt_attention_mask: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
    ):
        # ... (docstring remains the same)

        print("====== Pipeline Execution Start ======")
        print(f"Transformer default dtype: {self.transformer._dtype}")

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.default_height
        width = width or self.default_width

        # 1. Check inputs
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        
        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Prepare text embeddings
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
        )
        
        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            prompt_embeds.dtype,
            generator,
            latents,
        )
        
        print("\n====== Initialized latents ======")
        print(f"min={latents.min().item():.4f}, max={latents.max().item():.4f}, mean={latents.mean().item():.4f}")

        if self.do_classifier_free_guidance:
            prompt_embeds = paddle.concat([negative_prompt_embeds, prompt_embeds], axis=0)
            prompt_attention_mask = paddle.concat([negative_prompt_attention_mask, prompt_attention_mask], axis=0)
        
        # 5. Prepare timestep
        threshold_noise = 0.025
        sigmas = linear_quadratic_schedule(num_inference_steps, threshold_noise)
        sigmas = np.array(sigmas)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            timesteps,
            sigmas,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
            
                self._current_timestep = 1000 - t
                latent_model_input = paddle.concat([latents] * 2) if self.do_classifier_free_guidance else latents
                timestep = paddle.full((latent_model_input.shape[0],), t, dtype=latents.dtype)
                
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    return_dict=False,
                )[0]
                
                # Type conversion
                noise_pred = noise_pred.cast('float32')

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    
                    # Perform CFG
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                # Scheduler step
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents.cast('float32'), return_dict=False)[0]
                latents = latents.cast(latents_dtype)
                
                # Print simple stats every 10 steps
                if i % 10 == 0 or i == len(timesteps) - 1:
                    print(f"Step {i}: noise_pred: {noise_pred.mean().item():.4f} | latents: {latents.mean().item():.4f} [{latents.min().item():.4f}, {latents.max().item():.4f}]")
                    
                    # Check for numerical issues
                    if latents.max().item() > 15.0 or latents.min().item() < -15.0:
                        print(f"⚠️ Large values detected in step {i}")
                        
                    if paddle.isnan(latents).any().item() or paddle.isinf(latents).any().item():
                        print(f"⚠️ NaN/Inf detected in step {i}")

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        if output_type == "latent":
            video = latents
        else:
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = paddle.to_tensor(self.vae.config.latents_mean).reshape([1, 12, 1, 1, 1]).astype(latents.dtype)
                latents_std = paddle.to_tensor(self.vae.config.latents_std).reshape([1, 12, 1, 1, 1]).astype(latents.dtype)
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor
                
            # VAE decode
            print("\n====== Pre-VAE decode ======")
            print(f"latents: min={latents.min().item():.4f}, max={latents.max().item():.4f}, mean={latents.mean().item():.4f}")
            print(f"scaling_factor: {self.vae.config.scaling_factor}")

            video = self.vae.decode(latents, return_dict=False)[0]
            
            print("\n====== Post-VAE decode ======")
            print(f"video: min={video.min().item() if hasattr(video, 'min') else 'N/A'}, max={video.max().item() if hasattr(video, 'max') else 'N/A'}")
            
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return MochiPipelineOutput(frames=video)
