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

import inspect
from typing import Callable, List, Optional, Tuple, Union

import paddle
from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

from ...models import AutoencoderKL, DiTLLaMAT2IModel
from ...schedulers import DDIMScheduler, DPMSolverMultistepScheduler
from ...utils import logging
from ...utils.paddle_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

logger = logging.get_logger(__name__)


class LDMTextToImageLargeDiTPipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular xxxx, etc.)

    Parameters:
        vae ([`AutoencoderKL`]):
        text_encoder (`paddlenlp.transformers.AutoModelForCausalLM`):
        tokenizer (`paddlenlp.transformers.AutoTokenizer`):
        transformer ([`DiTLLaMAT2IModel`]): DiTLLaMAT2IModel(LargeDiT_T2I) architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`DPMSolverMultistepScheduler`].
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        transformer: DiTLLaMAT2IModel,
        scheduler: Union[DDIMScheduler, DPMSolverMultistepScheduler],
    ):
        super().__init__()
        self.register_modules(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer, scheduler=scheduler
        )
        self.cap_feat_dim = self.text_encoder.config.hidden_size
        self.vae_scale_factor = 8

    def _encode_prompt(self, prompt):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """
        cap_tok = self.tokenizer(prompt, truncation=False, return_tensors="pd").input_ids[0]
        null_cap_tok = self.tokenizer("", truncation=False, return_tensors="pd").input_ids[0]
        tok = paddle.zeros([2, max(len(cap_tok), len(null_cap_tok))], dtype=paddle.int64)
        tok_mask = paddle.zeros_like(tok).cast("bool")

        tok[0, : len(cap_tok)] = cap_tok
        tok[1, : len(null_cap_tok)] = null_cap_tok
        tok_mask[0, : len(cap_tok)] = True
        tok_mask[1, : len(null_cap_tok)] = True

        with paddle.no_grad():
            cap_feats = self.text_encoder.get_decoder()(input_ids=tok)[0].cast("float32")

        return cap_feats, tok_mask

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
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
        shape = [
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        ]
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @paddle.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 10,
        guidance_scale: float = 4.0,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        latents: Optional[paddle.Tensor] = None,
        prompt_embeds: Optional[paddle.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to 256):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 256):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 1.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`paddle.Generator` or `List[paddle.Generator]`, *optional*):
                One or a list of paddle generator(s) to make generation deterministic.
            latents (`paddle.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`paddle.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: paddle.Tensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [ppdiffusers.cross_attention](https://github.com/PaddlePaddle/PaddleMIX/blob/develop/ppdiffusers/ppdiffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            prompt_embeds,
        )
        dtype = paddle.bfloat16  # TODO: make this a parameter
        num_channels_latents = self.transformer.in_channels
        latent_channels = self.transformer.config.in_channels

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        prompt_embeds, cap_mask = self._encode_prompt(prompt)
        prompt_embeds = prompt_embeds.cast(dtype)
        cap_mask = cap_mask.cast(dtype)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )
        latent_model_input = paddle.concat([latents] * 2) if guidance_scale > 1 else latents

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if guidance_scale > 1:
                    half = latent_model_input[: len(latent_model_input) // 2]
                    latent_model_input = paddle.concat([half, half], axis=0)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                timesteps = t
                if not paddle.is_tensor(timesteps):
                    if isinstance(timesteps, float):
                        dtype = paddle.float32
                    else:
                        dtype = paddle.int64
                    timesteps = paddle.to_tensor([timesteps], dtype=dtype)
                elif len(timesteps.shape) == 0:
                    timesteps = timesteps[None]
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = timesteps.expand(
                    [
                        latent_model_input.shape[0],
                    ]
                )

                # predict the noise residual
                noise_pred = self.transformer(
                    latent_model_input,  # [2, 4, 128, 128]
                    timesteps,
                    cap_feats=prompt_embeds,
                    cap_mask=cap_mask,
                    return_dict=True,
                ).sample

                # perform guidance
                if guidance_scale > 1:
                    eps, rest = noise_pred[:, :3], noise_pred[:, 3:]
                    cond_eps, uncond_eps = paddle.chunk(eps, 2, axis=0)
                    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                    eps = paddle.concat([half_eps, half_eps], axis=0)
                    noise_pred = paddle.concat([eps, rest], axis=1)

                # learned sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    model_output, _ = paddle.split(
                        noise_pred, [latent_channels, noise_pred.shape[1] - latent_channels], axis=1
                    )
                else:
                    model_output = noise_pred

                # compute previous image: x_t -> x_t-1
                latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if guidance_scale > 1:
            latents, _ = latent_model_input.chunk(2, axis=0)
        else:
            latents = latent_model_input

        # 7. Decode latents to image
        latents = 1 / self.vae.config.scaling_factor * latents
        samples = self.vae.decode(latents).sample
        samples = (samples / 2 + 0.5).clip(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        samples = samples.transpose([0, 2, 3, 1]).cast("float32").cpu().numpy()

        # 8. Convert to PIL Image
        if output_type == "pil":
            samples = self.numpy_to_pil(samples)

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
