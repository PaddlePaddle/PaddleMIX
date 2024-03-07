# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/pipelines/pipeline_animation.py
import inspect
import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import paddle
from einops import rearrange
from paddlenlp.transformers import CLIPImageProcessor
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.pipelines.context import get_context_scheduler
from src.pipelines.utils import get_tensor_interpolation_method
from tqdm import tqdm

from ppdiffusers import DiffusionPipeline
from ppdiffusers.image_processor import VaeImageProcessor
from ppdiffusers.models.modeling_utils import faster_set_state_dict
from ppdiffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from ppdiffusers.utils import BaseOutput, smart_load
from ppdiffusers.utils.paddle_utils import randn_tensor


@dataclass
class Pose2VideoPipelineOutput(BaseOutput):
    videos: Union[paddle.Tensor, np.ndarray]


class Pose2VideoPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae,
        image_encoder,
        reference_unet,
        denoising_unet,
        pose_guider,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        image_proj_model=None,
        tokenizer=None,
        text_encoder=None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
            image_proj_model=image_proj_model,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

    def load_pretrained(self, config):
        # loading the denoising unet with motion module
        state_dict_denoise = smart_load(config.denoising_unet_path)
        state_dict_motion = smart_load(config.motion_module_path)
        state_dict_denoise.update(state_dict_motion)
        faster_set_state_dict(self.denoising_unet, state_dict_denoise)

        # loading the reference unet
        state_dict_ref = smart_load(config.reference_unet_path)
        faster_set_state_dict(self.reference_unet, state_dict_ref)

        # loading the pose guider
        state_dict_pose_guider = smart_load(config.pose_guider_path)
        faster_set_state_dict(self.pose_guider, state_dict_pose_guider)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        video = paddle.concat(x=video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clip(min=0, max=1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.astype(dtype=latents.dtype).numpy()
        return video

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

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)
        else:
            latents = latents

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _encode_prompt(
        self,
        prompt,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pd",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pd").input_ids

        if (
            untruncated_ids.shape[-1] >= text_input_ids.shape[-1]
            and not paddle.equal_all(x=text_input_ids, y=untruncated_ids).item()
        ):
            _ = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.tile((1, num_videos_per_prompt, 1))
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pd",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids,
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.tile((1, num_videos_per_prompt, 1))
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes

            text_embeddings = paddle.concat(x=[uncond_embeddings, text_embeddings])

        return text_embeddings

    def interpolate_latents(self, latents: paddle.Tensor, interpolation_factor: int):
        if interpolation_factor < 2:
            return latents

        new_latents = paddle.zeros(
            shape=(
                latents.shape[0],
                latents.shape[1],
                (latents.shape[2] - 1) * interpolation_factor + 1,
                latents.shape[3],
                latents.shape[4],
            ),
            dtype=latents.dtype,
        )

        org_video_length = latents.shape[2]
        rate = [i / interpolation_factor for i in range(interpolation_factor)][1:]

        new_index = 0

        v0 = None
        v1 = None

        for i0, i1 in zip(range(org_video_length), range(org_video_length)[1:]):
            v0 = latents[:, :, i0, :, :]
            v1 = latents[:, :, i1, :, :]

            new_latents[:, :, new_index, :, :] = v0
            new_index += 1

            for f in rate:

                v = get_tensor_interpolation_method()(v0, v1, f)
                new_latents[:, :, new_index, :, :] = v

                new_index += 1

        new_latents[:, :, new_index, :, :] = v1
        new_index += 1

        return new_latents

    @paddle.no_grad()
    def __call__(
        self,
        ref_image,
        pose_images,
        width,
        height,
        video_length,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        eta: float = 0.0,
        generator: Optional[Union[paddle.Generator, List[paddle.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, paddle.Tensor], None]] = None,
        callback_steps: Optional[int] = 1,
        context_schedule="uniform",
        context_frames=24,
        context_stride=1,
        context_overlap=4,
        context_batch_size=1,
        interpolation_factor=1,
        **kwargs,
    ):
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        # Prepare clip image embeds
        clip_image = self.clip_image_processor.preprocess(
            ref_image.resize((224, 224)), return_tensors="pd"
        ).pixel_values

        clip_image_embeds = self.image_encoder(clip_image).image_embeds

        encoder_hidden_states = clip_image_embeds.unsqueeze(1)
        uncond_encoder_hidden_states = paddle.zeros_like(x=encoder_hidden_states)

        if do_classifier_free_guidance:
            encoder_hidden_states = paddle.concat(x=[uncond_encoder_hidden_states, encoder_hidden_states], axis=0)

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )

        num_channels_latents = self.denoising_unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            clip_image_embeds.dtype,
            generator,
        )

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Prepare ref image latents
        ref_image_tensor = self.ref_image_processor.preprocess(ref_image, height=height, width=width).astype(
            latents.dtype
        )  # (bs, c, width, height)

        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)

        # Prepare a list of pose condition images
        pose_cond_tensor_list = []
        for pose_image in pose_images:
            pose_cond_tensor = self.cond_image_processor.preprocess(pose_image, height=height, width=width).astype(
                latents.dtype
            )
            pose_cond_tensor = pose_cond_tensor.unsqueeze(2)  # (bs, c, 1, h, w)
            pose_cond_tensor_list.append(pose_cond_tensor)
        pose_cond_tensor = paddle.concat(x=pose_cond_tensor_list, axis=2)

        pose_fea = self.pose_guider(pose_cond_tensor)

        context_scheduler = get_context_scheduler(context_schedule)

        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                noise_pred = paddle.zeros(
                    shape=(latents.shape[0] * (2 if do_classifier_free_guidance else 1), *latents.shape[1:]),
                    dtype=latents.dtype,
                )
                counter = paddle.zeros(shape=(1, 1, latents.shape[2], 1, 1), dtype=latents.dtype)

                # 1. Forward reference image
                if i == 0:
                    self.reference_unet(
                        ref_image_latents.tile(((2 if do_classifier_free_guidance else 1), 1, 1, 1)),
                        paddle.zeros_like(x=t),
                        encoder_hidden_states=encoder_hidden_states,
                        return_dict=False,
                    )
                    reference_control_reader.update(reference_control_writer)

                context_queue = list(
                    context_scheduler(
                        0,
                        num_inference_steps,
                        latents.shape[2],
                        context_frames,
                        context_stride,
                        0,
                    )
                )
                num_context_batches = math.ceil(len(context_queue) / context_batch_size)

                context_queue = list(
                    context_scheduler(
                        0,
                        num_inference_steps,
                        latents.shape[2],
                        context_frames,
                        context_stride,
                        context_overlap,
                    )
                )

                num_context_batches = math.ceil(len(context_queue) / context_batch_size)
                global_context = []
                for i in range(num_context_batches):
                    global_context.append(context_queue[i * context_batch_size : (i + 1) * context_batch_size])

                for context in global_context:

                    latent_model_input = paddle.concat(x=[latents[:, :, c] for c in context]).transpose(
                        [2, 0, 1, 3, 4]
                    )
                    latent_model_input = latent_model_input.tile(
                        ((2 if do_classifier_free_guidance else 1), 1, 1, 1, 1)
                    )

                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    b, c, f, h, w = latent_model_input.shape

                    latent_pose_input = paddle.concat(x=[pose_fea[:, :, c] for c in context]).transpose(
                        [2, 0, 1, 3, 4]
                    )
                    latent_pose_input = latent_pose_input.tile((2 if do_classifier_free_guidance else 1, 1, 1, 1, 1))

                    pred = self.denoising_unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=encoder_hidden_states[:b],
                        pose_cond_fea=latent_pose_input,
                        return_dict=False,
                    )[0]

                    for j, c in enumerate(context):

                        add_pred_noise = noise_pred[:, :, c].transpose([2, 0, 1, 3, 4]) + pred
                        add_1_conuter = counter[:, :, c].transpose([2, 0, 1, 3, 4]) + 1
                        for index, value_c in enumerate(c):

                            noise_pred[:, :, value_c] = add_pred_noise[:, :, index]
                            counter[:, :, value_c] = add_1_conuter[:, :, index]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = (noise_pred / counter).chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample.cast(
                    latents.dtype
                )

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

            reference_control_reader.clear()
            reference_control_writer.clear()

        if interpolation_factor > 0:
            latents = self.interpolate_latents(latents, interpolation_factor)
        # Post-processing
        images = self.decode_latents(latents)  # (b, c, f, h, w)

        # Convert to tensor
        if output_type == "tensor":
            images = paddle.to_tensor(data=images, dtype=latents.dtype)

        if not return_dict:
            return images

        return Pose2VideoPipelineOutput(videos=images)
