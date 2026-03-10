# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
# with the following modifications:
# - It uses the patched version of `sde_step_with_logprob` from `sd3_sde_with_logprob.py`.
# - It returns all the intermediate latents of the denoising process as well as the log probs of each denoising step.
from typing import Any, Dict, List, Optional, Union

import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet

from ppdiffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)

from .sd3_sde_with_logprob import sde_step_with_logprob

try:
    # paddle.incubate.jit.inference is available in paddle develop but not in paddle 3.0beta, so we add a try except.
    from paddle.incubate.jit import is_inference_mode
except:

    def is_inference_mode(func):
        return False


@paddle.no_grad()
def pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    timesteps: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
    latents: Optional[paddle.Tensor] = None,
    prompt_embeds: Optional[paddle.Tensor] = None,
    negative_prompt_embeds: Optional[paddle.Tensor] = None,
    pooled_prompt_embeds: Optional[paddle.Tensor] = None,
    negative_pooled_prompt_embeds: Optional[paddle.Tensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    noise_level: float = 0.7,
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds,) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        clip_skip=self.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    if self.do_classifier_free_guidance:
        prompt_embeds = paddle.concat([negative_prompt_embeds, prompt_embeds], axis=0)
        pooled_prompt_embeds = paddle.concat([negative_pooled_prompt_embeds, pooled_prompt_embeds], axis=0)

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        generator,
        latents,
    )

    # 5. Prepare timesteps
    scheduler_kwargs = {}
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        timesteps=timesteps,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # 6. Prepare image embeddings
    all_latents = [latents]
    all_log_probs = []

    # 7. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = paddle.concat([latents] * 2) if self.do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latent_model_input.shape[0])

            enabled_cfg_dp = False
            if self.transformer.inference_dp_size > 1:
                enabled_cfg_dp = True
                assert self.do_classifier_free_guidance, "do_classifier_free_guidance must be true"

            if enabled_cfg_dp:
                dp_id = self.transformer.dp_id
                latent_input = paddle.split(latent_model_input, 2, axis=0)[dp_id]
                timestep_input = paddle.split(timestep, 2, axis=0)[dp_id]
                prompt_embeds_input = paddle.split(prompt_embeds, 2, axis=0)[dp_id]
                pooled_prompt_embeds_input = paddle.split(pooled_prompt_embeds, 2, axis=0)[dp_id]

            else:
                latent_input = latent_model_input
                timestep_input = timestep
                prompt_embeds_input = prompt_embeds
                pooled_prompt_embeds_input = pooled_prompt_embeds

            noise_pred = self.transformer(
                hidden_states=latent_input,
                timestep=timestep_input,
                encoder_hidden_states=prompt_embeds_input,
                pooled_projections=pooled_prompt_embeds_input,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )
            #
            if is_inference_mode(self.transformer):
                # NOTE:(changwenbin,zhoukangkang)
                # This is for paddle inference mode
                output = noise_pred
            else:
                output = noise_pred[0]

            if enabled_cfg_dp:
                tmp_shape = output.shape
                tmp_shape[0] *= 2
                noise_pred = paddle.zeros(tmp_shape, dtype=output.dtype)
                dist.all_gather(
                    noise_pred, output, group=fleet.get_hybrid_communicate_group().get_data_parallel_group()
                )
            else:
                noise_pred = output
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            noise_pred = noise_pred.to(prompt_embeds.dtype)
            latents_dtype = latents.dtype
            latents = latents.to(prompt_embeds.dtype)
            latents, log_prob, prev_latents_mean, std_dev_t = sde_step_with_logprob(
                self.scheduler,
                model_output=noise_pred.astype("float32"),
                timestep=t.unsqueeze(0),
                sample=latents.astype("float32"),
                noise_level=noise_level,
            )

            # Convert latents back to original dtype before storing
            latents = latents.to(latents_dtype)
            all_latents.append(latents)
            all_log_probs.append(log_prob)
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

    latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    latents = latents.to(dtype=self.vae.dtype)
    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    return image, all_latents, all_log_probs
