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

from typing import Any, Callable, Dict, List, Optional, Union

import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
from fkd_class import FKD
from rewards import get_reward_function

import ppdiffusers
from ppdiffusers import StableDiffusion3Pipeline
from ppdiffusers.models import AutoencoderKL, SD3Transformer2DModel
from ppdiffusers.pipelines.stable_diffusion_3.pipeline_output import (
    StableDiffusion3PipelineOutput,
)
from ppdiffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)
from ppdiffusers.schedulers import FlowMatchEulerDiscreteScheduler
from ppdiffusers.transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
)

try:
    from paddle.incubate.jit import is_inference_mode
except:

    def is_inference_mode(func):
        return False


logger = ppdiffusers.utils.logging.get_logger(__name__)


class FKDStableDiffusion3Pipeline(StableDiffusion3Pipeline):
    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5Tokenizer,
    ):
        super().__init__(
            transformer=transformer,
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            text_encoder_3=text_encoder_3,
            tokenizer_3=tokenizer_3,
        )

    @paddle.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
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
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        fkd_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):

        # 0. Default height and width to unet
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
            batch_size = tuple(prompt_embeds.shape)[0]

        # 3. Encode input prompt

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
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

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = paddle.concat(x=[negative_prompt_embeds, prompt_embeds], axis=0)
            pooled_prompt_embeds = paddle.concat([negative_pooled_prompt_embeds, pooled_prompt_embeds], axis=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, timesteps)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
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

        # reward_fn
        def postprocess_and_apply_reward_fn(x):
            imagesx = self.image_processor.postprocess(x, output_type=output_type)
            imagesx = [image for image in imagesx]
            rewards = get_reward_function(
                fkd_args["guidance_reward_fn"],
                images=imagesx,
                prompts=prompt,
                metric_to_chase=fkd_args.get("metric_to_chase", None),
            )
            return paddle.to_tensor(data=rewards)

        print("Args:", fkd_args)
        if fkd_args is not None and fkd_args["use_smc"]:
            fkd = FKD(
                latent_to_decode_fn=lambda x: self.latent_to_decode(output_type=output_type, latents=x),
                reward_fn=postprocess_and_apply_reward_fn,
                **fkd_args,
            )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat(x=[latents] * 2) if self.do_classifier_free_guidance else latents

                timestep = t.expand(latent_model_input.shape[0])

                enabled_cfg_dp = False
                # if self.transformer.inference_dp_size > 1:
                #     enabled_cfg_dp = True
                #     assert self.do_classifier_free_guidance, "do_classifier_free_guidance must be true"

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

                model_output = self.transformer(
                    hidden_states=latent_input,
                    timestep=timestep_input,
                    encoder_hidden_states=prompt_embeds_input,
                    pooled_projections=pooled_prompt_embeds_input,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )

                if is_inference_mode(self.transformer):
                    # NOTE:(changwenbin,zhoukangkang)
                    # This is for paddle inference mode
                    output = model_output
                else:
                    output = model_output[0]
                output = model_output[0]

                if enabled_cfg_dp:
                    tmp_shape = output.shape
                    tmp_shape[0] *= 2
                    noise_pred = paddle.zeros(tmp_shape, dtype=output.dtype)
                    dist.all_gather(
                        noise_pred, output, group=fleet.get_hybrid_communicate_group().get_data_parallel_group()
                    )
                else:
                    noise_pred = output

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(chunks=2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                x0_preds = latents

                if fkd_args is not None and fkd_args["use_smc"]:

                    latents, _ = fkd.resample(sampling_idx=i, latents=latents, x0_preds=x0_preds)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                if i == len(timesteps) - 1 or i + 1 > num_warmup_steps and (i + 1) % self.scheduler.order == 0:
                    progress_bar.update()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return image
        return StableDiffusion3PipelineOutput(images=image)

    def latent_to_decode(self, latents, output_type):
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

        image = self.vae.decode(latents, return_dict=False)[0]

        return image
