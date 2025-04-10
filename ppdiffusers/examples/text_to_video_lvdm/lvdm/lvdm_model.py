# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import inspect
import json
import os
import random

import numpy as np
import paddle
import paddle.nn as nn
from einops import rearrange
from paddlenlp.utils.log import logger

from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    LVDMAutoencoderKL,
    LVDMUNet3DModel,
    is_ppxformers_available,
)
from ppdiffusers.initializer import (
    normal_,
    reset_initialized_parameter,
    xavier_uniform_,
    zeros_,
)
from ppdiffusers.models.ema import LitEma
from ppdiffusers.models.lvdm_attention_temporal import (
    RelativePosition,
    TemporalCrossAttention,
)
from ppdiffusers.models.lvdm_distributions import DiagonalGaussianDistribution
from ppdiffusers.training_utils import freeze_params
from ppdiffusers.transformers import AutoTokenizer, CLIPTextModel


def set_seed(seed: int = 1234, args=None):
    if args is None:
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    if args is not None:
        if args.use_hybrid_parallel:
            from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker

            random.seed(args.seed + args.dataset_rank)
            np.random.seed(args.seed + args.dataset_rank)
            paddle.seed(args.seed + args.dataset_rank)

            # local_seed/ global_seed is used to control dropout in ModelParallel
            local_seed = args.seed + 59999 + args.tensor_parallel_rank * 10 + args.pipeline_parallel_rank * 1000
            global_seed = args.seed + 100003 + args.dataset_rank
            tracker = get_rng_state_tracker()

            if "global_seed" not in tracker.states_:
                tracker.add("global_seed", global_seed)
            if "local_seed" not in tracker.states_:
                tracker.add("local_seed", local_seed)
        else:
            random.seed(args.seed)
            np.random.seed(args.seed)
            paddle.seed(args.seed)


def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def split_video_to_clips(video, clip_length, drop_left=True):
    video_length = video.shape[2]
    shape = video.shape
    if video_length % clip_length != 0 and drop_left:
        video = video[:, :, : video_length // clip_length * clip_length, :, :]
        print(f"[split_video_to_clips] Drop frames from {shape} to {video.shape}")
    nclips = video_length // clip_length
    clips = rearrange(video, "b c (nc cl) h w -> (b nc) c cl h w", cl=clip_length, nc=nclips)
    return clips


def merge_clips_to_videos(clips, bs):
    nclips = clips.shape[0] // bs
    video = rearrange(clips, "(b nc) c t h w -> b c (nc t) h w", nc=nclips)
    return video


class LatentVideoDiffusion(nn.Layer):
    def __init__(self, model_args):
        super().__init__()
        # initialization
        assert model_args.task_type in ["short", "text2video"]
        self.task_type = model_args.task_type

        # init tokenizer
        if model_args.task_type == "text2video":
            tokenizer_name_or_path = (
                model_args.tokenizer_name_or_path
                if model_args.pretrained_model_name_or_path is None
                else os.path.join(model_args.pretrained_model_name_or_path, "tokenizer")
            )
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        # init vae
        vae_name_or_path = (
            model_args.vae_name_or_path
            if model_args.pretrained_model_name_or_path is None
            else os.path.join(model_args.pretrained_model_name_or_path, "vae")
        )
        self.vae_type = model_args.vae_type
        self.encoder_type = model_args.vae_type
        if model_args.vae_type == "2d":
            self.vae = AutoencoderKL.from_pretrained(vae_name_or_path)
        elif model_args.vae_type == "3d":
            self.vae = LVDMAutoencoderKL.from_pretrained(vae_name_or_path)
        else:
            raise ValueError("`vae_type` to be `2d` or `3d`.")
        freeze_params(self.vae.parameters())
        logger.info("Freeze vae parameters!")

        # init text_encoder
        if model_args.task_type == "text2video":
            text_encoder_name_or_path = (
                model_args.text_encoder_name_or_path
                if model_args.pretrained_model_name_or_path is None
                else os.path.join(model_args.pretrained_model_name_or_path, "text_encoder")
            )
            self.text_encoder_is_pretrained = text_encoder_name_or_path is not None
            if self.text_encoder_is_pretrained:
                self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_name_or_path)
            else:
                self.text_encoder = CLIPTextModel(**read_json(model_args.text_encoder_config_file))
                self.init_text_encoder_weights()
            if not model_args.is_text_encoder_trainable:
                freeze_params(self.text_encoder.parameters())
                logger.info("Freeze text_encoder parameters!")

        # init unet
        unet_name_or_path = (
            model_args.unet_name_or_path
            if model_args.pretrained_model_name_or_path is None
            else os.path.join(model_args.pretrained_model_name_or_path, "unet")
        )
        self.unet_is_pretrained = model_args.pretrained_model_name_or_path is not None
        if self.unet_is_pretrained:
            self.unet = LVDMUNet3DModel.from_pretrained(unet_name_or_path)
        else:
            self.unet = LVDMUNet3DModel(**read_json(model_args.unet_config_file))
            self.init_unet_weights()

        # init train scheduler
        self.noise_scheduler = DDPMScheduler(
            beta_start=model_args.scheduler_beta_start,
            beta_end=model_args.scheduler_beta_end,
            beta_schedule="scaled_linear",
            num_train_timesteps=model_args.scheduler_num_train_timesteps,
        )

        # init eval scheduler
        self.eval_scheduler = DDIMScheduler(
            beta_start=model_args.scheduler_beta_start,
            beta_end=model_args.scheduler_beta_end,
            beta_schedule="scaled_linear",
            num_train_timesteps=model_args.scheduler_num_train_timesteps,
            steps_offset=1,
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.eval_scheduler.set_timesteps(model_args.eval_scheduler_num_inference_steps)

        # set training parameters
        self.use_ema = model_args.use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.unet)

        if model_args.enable_xformers_memory_efficient_attention and is_ppxformers_available():
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warn(
                    "Could not enable memory efficient attention. Make sure develop paddlepaddle is installed"
                    f" correctly and a GPU is available: {e}"
                )
        self.scale_factor = model_args.scale_factor
        self.shift_factor = model_args.shift_factor
        self.loss_type = model_args.loss_type

        # set alignment parameters
        self.use_preconfig_latents = False
        if model_args.latents_path:
            self.use_preconfig_latents = True
            self.register_buffer("preconfig_latents", paddle.load(model_args.latents_path))

        self.if_numpy_genarator_random_alignment = model_args.if_numpy_genarator_random_alignment
        if self.if_numpy_genarator_random_alignment:
            self.generator = np.random.RandomState(model_args.numpy_genarator_random_seed)

        self.set_seed_for_alignment = model_args.set_seed_for_alignment

    def init_text_encoder_weights(self):
        if not self.text_encoder_is_pretrained:
            reset_initialized_parameter(self.text_encoder)
            normal_(self.text_encoder.embeddings.word_embeddings.weight, 0, 0.02)
            normal_(self.text_encoder.embeddings.position_embeddings.weight, 0, 0.02)

    def init_unet_weights(self):
        if not self.unet_is_pretrained:
            reset_initialized_parameter(self.unet)
            for _, m in self.unet.named_sublayers():
                if isinstance(m, TemporalCrossAttention):
                    zeros_(m.to_q.weight)
                    zeros_(m.to_k.weight)
                    zeros_(m.to_v.weight)
                    zeros_(m.to_out[0].weight)
                    zeros_(m.o_out[0].bias)
                if isinstance(m, RelativePosition):
                    xavier_uniform_(m.embeddings_table)

    @contextlib.contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.controlnet.parameters())
            self.model_ema.copy_to(self.controlnet)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.controlnet.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self):
        if self.use_ema:
            self.model_ema(self.unet)

    # for latents encode
    def get_first_stage_encoding(self, encoder_posterior, noise=None):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample(noise=noise)
        elif isinstance(encoder_posterior, paddle.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        z = self.scale_factor * (z + self.shift_factor)
        return z

    @paddle.no_grad()
    def encode_first_stage(self, x):
        if self.vae_type == "2d" and x.dim() == 5:
            b, _, t, _, _ = x.shape
            x = rearrange(x, "b c t h w -> (b t) c h w")
            results = self.vae.encode(x).latent_dist.sample()
            results = rearrange(results, "(b t) c h w -> b c t h w", b=b, t=t)
        else:
            results = self.vae.encode(x).latent_dist.sample()
        return results

    def encode_latents(self, x):
        b, _, t, h, w = x.shape
        if self.vae_type == "2d":
            x = rearrange(x, "b c t h w -> (b t) c h w")
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        if self.vae_type == "2d":
            z = rearrange(z, "(b t) c h w -> b c t h w", b=b, t=t)
        return z

    # for latents decode
    @paddle.no_grad()
    def decode(self, z, **kwargs):
        z = 1.0 / self.scale_factor * z - self.shift_factor
        results = self.vae.decode(z).sample
        return results

    @paddle.no_grad()
    def overlapped_decode(self, z, max_z_t=None, overlap_t=2, predict_cids=False, force_not_quantize=False):
        if max_z_t is None:
            max_z_t = z.shape[2]
        assert max_z_t > overlap_t
        max_x_t = max_z_t * 4
        drop_r = overlap_t // 2
        drop_l = overlap_t - drop_r
        drop_r_x = drop_r * 4
        drop_l_x = drop_l * 4
        start = 0
        end = max_z_t
        zs = []
        while start <= z.shape[2]:
            zs.append(z[:, :, start:end, :, :])
            start += max_z_t - overlap_t
            end += max_z_t - overlap_t
        reses = []
        for i, z_ in enumerate(zs):
            if i == 0:
                res = self.decode(z_, predict_cids, force_not_quantize).cpu()[:, :, : max_x_t - drop_r_x, :, :]
            elif i == len(zs) - 1:
                res = self.decode(z_, predict_cids, force_not_quantize).cpu()[:, :, drop_l_x:, :, :]
            else:
                res = self.decode(z_, predict_cids, force_not_quantize).cpu()[
                    :, :, drop_l_x : max_x_t - drop_r_x, :, :
                ]
            reses.append(res)
        results = paddle.concat(x=reses, axis=2)
        return results

    @paddle.no_grad()
    def decode_first_stage_2DAE_video(self, z, decode_bs=16, return_cpu=True, **kwargs):
        b, _, t, _, _ = z.shape
        z = rearrange(z, "b c t h w -> (b t) c h w")
        if decode_bs is None:
            results = self.decode(z, **kwargs)
        else:
            z = paddle.split(x=z, num_or_sections=z.shape[0] // decode_bs, axis=0)
            if return_cpu:
                results = paddle.concat(x=[self.decode(z_, **kwargs).cpu() for z_ in z], axis=0)
            else:
                results = paddle.concat(x=[self.decode(z_, **kwargs) for z_ in z], axis=0)
        results = rearrange(results, "(b t) c h w -> b c t h w", b=b, t=t).contiguous()
        return results

    @paddle.no_grad()
    def decode_latents(
        self,
        z,
        decode_bs=16,
        return_cpu=True,
        bs=None,
        decode_single_video_allframes=False,
        max_z_t=None,
        overlapped_length=0,
        **kwargs,
    ):
        b, _, t, _, _ = z.shape
        if self.encoder_type == "2d" and z.dim() == 5:
            return self.decode_first_stage_2DAE_video(z, decode_bs=decode_bs, return_cpu=return_cpu, **kwargs)
        if decode_single_video_allframes:
            z = paddle.split(x=z, num_or_sections=z.shape[0] // 1, axis=0)
            cat_dim = 0
        elif max_z_t is not None:
            if self.encoder_type == "3d":
                z = paddle.split(x=z, num_or_sections=z.shape[2] // max_z_t, axis=2)
                cat_dim = 2
            if self.encoder_type == "2d":
                z = paddle.split(x=z, num_or_sections=z.shape[0] // max_z_t, axis=0)
                cat_dim = 0
        # elif self.split_clips and self.downfactor_t is not None or self.clip_length is not None and self.downfactor_t is not None and z.shape[
        #     2
        #     ] > self.clip_length // self.downfactor_t and self.encoder_type == '3d':
        #     split_z_t = self.clip_length // self.downfactor_t
        #     print(f'split z ({z.shape}) to length={split_z_t} clips')
        #     z = split_video_to_clips(z, clip_length=split_z_t, drop_left=True)
        #     if bs is not None and z.shape[0] > bs:
        #         print(f'split z ({z.shape}) to bs={bs}')
        #         z = paddle.split(x=z, num_or_sections=z.shape[0] // bs, axis=0)
        #         cat_dim = 0
        paddle.device.cuda.empty_cache()
        if isinstance(z, tuple):
            zs = [self.decode(z_, **kwargs).cpu() for z_ in z]
            results = paddle.concat(x=zs, axis=cat_dim)
        elif isinstance(z, paddle.Tensor):
            results = self.decode(z, **kwargs)
        else:
            raise ValueError
        # if self.split_clips and self.downfactor_t is not None:
        #     results = merge_clips_to_videos(results, bs=b)
        return results

    def get_loss(self, pred, target, mean=True, mask=None):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = paddle.nn.functional.mse_loss(target, pred)
            else:
                loss = paddle.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")
        if mask is not None:
            assert mean is False
            assert loss.shape[2:] == mask.shape[2:]  # thw need be the same
            loss = loss * mask
        return loss

    global_i = 0

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        self.train()
        with paddle.amp.auto_cast(enable=False):
            with paddle.no_grad():
                self.vae.eval()
                if self.task_type == "text2video":
                    self.text_encoder.eval()
                latents = self.encode_latents(pixel_values)
                if self.set_seed_for_alignment:
                    set_seed(23)
                    self.set_seed_for_alignment = False
                if self.if_numpy_genarator_random_alignment:
                    timesteps = paddle.to_tensor(
                        self.generator.randint(
                            0,
                            self.noise_scheduler.num_train_timesteps,
                            size=(latents.shape[0],),
                        ),
                        dtype="int64",
                    )
                    noise = paddle.to_tensor(self.generator.randn(*latents.shape), dtype="float32")
                else:
                    timesteps = paddle.randint(
                        0, self.noise_scheduler.num_train_timesteps, (latents.shape[0],)
                    ).astype("int64")
                    noise = paddle.randn_like(latents)

                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = None
                if self.task_type == "text2video":
                    encoder_hidden_states = self.text_encoder(input_ids)[0]

        # predict the noise residual
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            context=encoder_hidden_states,
        ).sample

        loss = self.get_loss(noise_pred, noise, mean=True)
        return loss

    @paddle.no_grad()
    def log_reconstruct_frames(self, pixel_values=None, **kwargs):
        self.eval()
        if pixel_values.shape[0] > 2:
            pixel_values = pixel_values[:2]
        latents = self.encode_latents(pixel_values)
        sampled_videos = self.decode_latents(latents)

        videos_frames = []
        for idx in range(sampled_videos.shape[0]):
            video = sampled_videos[idx]
            for fidx in range(video.shape[1]):
                frame = video[:, fidx]
                frame = (frame / 2 + 0.5).clip(0, 1)
                frame = frame.transpose([1, 2, 0]).astype("float32").numpy()
                frame = (frame * 255).round()
                videos_frames.append(frame)
        videos_frames = np.stack(videos_frames, axis=0)
        return videos_frames

    @paddle.no_grad()
    def log_text2video_sample_frames(
        self,
        input_ids=None,
        height=256,
        width=256,
        eta=1.0,
        guidance_scale=9,
        num_frames=16,
        **kwargs,
    ):
        self.eval()
        with self.ema_scope():
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
            # only log 2 video
            if input_ids.shape[0] > 2:
                input_ids = input_ids[:2]

            text_embeddings = self.text_encoder(input_ids)[0]
            do_classifier_free_guidance = guidance_scale > 1.0
            if do_classifier_free_guidance:
                batch_size, max_length = input_ids.shape
                uncond_input = self.tokenizer(
                    [""] * batch_size,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pd",
                )
                uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
                text_embeddings = paddle.concat([uncond_embeddings, text_embeddings], axis=0)
            if self.use_preconfig_latents:
                latents = self.preconfig_latents
            else:
                shape = [
                    input_ids.shape[0],
                    self.unet.in_channels,
                    num_frames,
                    height // 8,
                    width // 8,
                ]
                latents = paddle.randn(shape)

            accepts_eta = "eta" in set(inspect.signature(self.eval_scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = eta

            for t in self.eval_scheduler.timesteps:
                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents

                # ddim do not use this
                latent_model_input = self.eval_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    context=text_embeddings,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.eval_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            sampled_videos = self.decode_latents(latents)

        videos_frames = []
        for idx in range(sampled_videos.shape[0]):
            video = sampled_videos[idx]
            for fidx in range(video.shape[1]):
                frame = video[:, fidx]
                frame = (frame / 2 + 0.5).clip(0, 1)
                frame = frame.transpose([1, 2, 0]).astype("float32").numpy()
                frame = (frame * 255).round()
                videos_frames.append(frame)
        videos_frames = np.stack(videos_frames, axis=0)
        return videos_frames

    @paddle.no_grad()
    def log_short_sample_frames(self, height=256, width=256, eta=0.0, guidance_scale=9, num_frames=16, **kwargs):
        self.eval()
        with self.ema_scope():
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
            # only log 2 video
            batch_size = 2

            if self.use_preconfig_latents:
                latents = self.preconfig_latents
            else:
                shape = [
                    batch_size,
                    self.unet.in_channels,
                    num_frames,
                    height // 8,
                    width // 8,
                ]
                latents = paddle.randn(shape)

            accepts_eta = "eta" in set(inspect.signature(self.eval_scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = eta

            for t in self.eval_scheduler.timesteps:
                # expand the latents if we are doing classifier free guidance
                latent_model_input = latents

                # ddim do not use this
                latent_model_input = self.eval_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                ).sample

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.eval_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            sampled_videos = self.decode_latents(latents)

        videos_frames = []
        for idx in range(sampled_videos.shape[0]):
            video = sampled_videos[idx]
            for fidx in range(video.shape[1]):
                frame = video[:, fidx]
                frame = (frame / 2 + 0.5).clip(0, 1)
                frame = frame.transpose([1, 2, 0]).astype("float32").numpy()
                frame = (frame * 255).round()
                videos_frames.append(frame)
        videos_frames = np.stack(videos_frames, axis=0)
        return videos_frames

    def set_recompute(self, value=False):
        def fn(layer):
            if hasattr(layer, "gradient_checkpointing"):
                layer.gradient_checkpointing = value
                print("Set", layer.__class__, "recompute", layer.gradient_checkpointing)

        self.unet.apply(fn)
