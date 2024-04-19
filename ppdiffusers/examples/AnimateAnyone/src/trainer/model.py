# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import random

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from paddlenlp.transformers import CLIPVisionModelWithProjection
from paddlenlp.utils.log import logger

from ppdiffusers import AutoencoderKL, DDIMScheduler
from ppdiffusers.models.animate_anyone.mutual_self_attention import (
    ReferenceAttentionControl,
)
from ppdiffusers.models.animate_anyone.pose_guider import PoseGuider
from ppdiffusers.models.animate_anyone.unet_2d_condition import UNet2DConditionModel
from ppdiffusers.models.animate_anyone.unet_3d import UNet3DConditionModel
from ppdiffusers.training_utils import freeze_params, unfreeze_params


class AnimateAnyoneModel_stage1(nn.Layer):
    def __init__(self, model_args):
        super().__init__()

        self.train_noise_scheduler = DDIMScheduler(
            beta_start=model_args.beta_start,
            beta_end=model_args.beta_end,
            beta_schedule=model_args.beta_schedule,
            steps_offset=model_args.steps_offset,
            clip_sample=model_args.clip_sample,
            rescale_betas_zero_snr=model_args.rescale_betas_zero_snr,
            timestep_spacing=model_args.timestep_spacing,
        )

        self.train_noise_scheduler.num_train_timesteps = model_args.num_train_timesteps
        self.train_noise_scheduler.prediction_type = model_args.prediction_type

        self.vae = AutoencoderKL.from_pretrained(
            model_args.vae_model_path,
        )

        self.reference_unet = UNet2DConditionModel.from_pretrained(
            model_args.base_model_path,
            subfolder="unet",
        )

        self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            denoising_unet_config_path=model_args.denoising_unet_config_path,
            base_model_path=model_args.denoising_unet_base_model_path,
            unet_additional_kwargs={
                "use_motion_module": False,
                "unet_use_temporal_attention": False,
            },
        )

        self.reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks="full",
        )
        self.reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=False,
            mode="read",
            fusion_blocks="full",
        )

        self.image_enc = CLIPVisionModelWithProjection.from_pretrained(
            model_args.image_encoder_path,
            subfolder="image_encoder",
        )

        if model_args.pose_guider_pretrain:
            self.pose_guider = PoseGuider(
                conditioning_embedding_channels=320,
                block_out_channels=(16, 32, 96, 256),
            )

            # load pretrained controlnet-openpose params for pose_guider
            controlnet_openpose_state_dict = paddle.load(model_args.controlnet_openpose_path)
            state_dict_to_load = {}
            for k in controlnet_openpose_state_dict.keys():
                if k.startswith("controlnet_cond_embedding.") and k.find("conv_out") < 0:
                    new_k = k.replace("controlnet_cond_embedding.", "")
                    state_dict_to_load[new_k] = controlnet_openpose_state_dict[k]
            miss, _ = self.pose_guider.set_state_dict(state_dict=state_dict_to_load)
            logger.info(f"Missing key for pose guider: {len(miss)}")
        else:
            self.pose_guider = PoseGuider(
                conditioning_embedding_channels=320,
            )

        self.noise_offset = model_args.noise_offset
        self.uncond_ratio = model_args.uncond_ratio
        self.snr_gamma = model_args.snr_gamma

        freeze_params(self.vae.parameters())
        freeze_params(self.image_enc.parameters())
        unfreeze_params(self.denoising_unet.parameters())

        #  Some top layer parames of reference_unet don't need grad
        for name, param in self.reference_unet.named_parameters():
            if "up_blocks.3" in name:
                freeze_params(param)
            else:
                unfreeze_params(param)

        unfreeze_params(self.pose_guider.parameters())

        self.reference_unet.train()
        self.denoising_unet.train()
        self.pose_guider.train()
        self.vae.eval()
        self.image_enc.eval()

    def compute_snr(self, noise_scheduler, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[timesteps].cast("float32")
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[timesteps].cast("float32")
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def forward(self, batch, **kwargs):

        pixel_values = batch["img"]
        with paddle.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents.unsqueeze(2)  # (b, c, 1, h, w)
            latents = latents * 0.18215

        noise = paddle.randn(shape=latents.shape, dtype=latents.dtype)

        if self.noise_offset > 0.0:
            noise += self.noise_offset * paddle.randn((noise.shape[0], noise.shape[1], 1, 1, 1))

        bsz = latents.shape[0]
        # Sample a random timestep for each video
        timesteps = paddle.randint(0, self.train_noise_scheduler.num_train_timesteps, (bsz,))
        timesteps = timesteps.astype(dtype="int64")

        tgt_pose_img = batch["tgt_pose"]
        tgt_pose_img = tgt_pose_img.unsqueeze(2)  # (bs, 3, 1, 512, 512)

        uncond_fwd = random.random() < self.uncond_ratio

        clip_image_list = []
        ref_image_list = []
        for _, (ref_img, clip_img) in enumerate(
            zip(
                batch["ref_img"],
                batch["clip_images"],
            )
        ):
            if uncond_fwd:
                clip_image_list.append(paddle.zeros_like(clip_img))
            else:
                clip_image_list.append(clip_img)
            ref_image_list.append(ref_img)

        with paddle.no_grad():
            ref_img = paddle.stack(ref_image_list, axis=0)
            ref_image_latents = self.vae.encode(ref_img).latent_dist.sample()  # (bs, d, 64, 64)
            ref_image_latents = ref_image_latents * 0.18215

            clip_img = paddle.stack(clip_image_list, axis=0)
            clip_image_embeds = self.image_enc(clip_img).image_embeds
            image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)

        # add noise
        noisy_latents = self.train_noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the target for loss depending on the prediction type
        if self.train_noise_scheduler.prediction_type == "epsilon":
            target = noise
        elif self.train_noise_scheduler.prediction_type == "v_prediction":
            target = self.train_noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.train_noise_scheduler.prediction_type}")

        pose_fea = self.pose_guider(tgt_pose_img)
        if not uncond_fwd:
            ref_timesteps = paddle.zeros_like(timesteps)

            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=image_prompt_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            pose_cond_fea=pose_fea,
            encoder_hidden_states=image_prompt_embeds,
        ).sample

        if self.snr_gamma == 0:
            loss = F.mse_loss(model_pred.astype("float32"), target.astype("float32"), reduction="mean")
        else:
            snr = self.compute_snr(self.train_noise_scheduler, timesteps)
            if self.train_noise_scheduler.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                paddle.stack([snr, self.snr_gamma * paddle.ones_like(timesteps)], axis=1).min(1)[0] / snr
            )
            loss = F.mse_loss(model_pred.cast("float32"), target.cast("float32"), reduction="none")
            loss = loss.mean(axis=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss


class AnimateAnyoneModel_stage2(nn.Layer):
    def __init__(self, model_args):
        super().__init__()

        self.train_noise_scheduler = DDIMScheduler(
            beta_start=model_args.beta_start,
            beta_end=model_args.beta_end,
            beta_schedule=model_args.beta_schedule,
            steps_offset=model_args.steps_offset,
            clip_sample=model_args.clip_sample,
            rescale_betas_zero_snr=model_args.rescale_betas_zero_snr,
            timestep_spacing=model_args.timestep_spacing,
        )

        self.train_noise_scheduler.num_train_timesteps = model_args.num_train_timesteps
        self.train_noise_scheduler.prediction_type = model_args.prediction_type

        self.vae = AutoencoderKL.from_pretrained(
            model_args.vae_model_path,
        )

        self.image_enc = CLIPVisionModelWithProjection.from_pretrained(
            model_args.image_encoder_path,
            subfolder="image_encoder",
        )

        self.reference_unet = UNet2DConditionModel.from_pretrained(
            model_args.base_model_path,
            subfolder="unet",
        )

        infer_config = OmegaConf.load(model_args.inference_config_path)

        self.denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            denoising_unet_config_path=model_args.denoising_unet_config_path,
            base_model_path=model_args.denoising_unet_base_model_path,
            motion_module_path=model_args.motion_module_path,
            unet_additional_kwargs=OmegaConf.to_container(infer_config.unet_additional_kwargs),
        )

        self.pose_guider = PoseGuider(conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256))

        reference_unet_state_dict = paddle.load(
            model_args.reference_unet_path,
        )

        for k in reference_unet_state_dict.keys():
            reference_unet_state_dict[k] = reference_unet_state_dict[k]

        self.reference_unet.set_state_dict(
            reference_unet_state_dict,
        )

        pose_guider_state_dict = paddle.load(
            model_args.pose_guider_path,
        )
        for k in pose_guider_state_dict.keys():
            pose_guider_state_dict[k] = pose_guider_state_dict[k]

        self.pose_guider.set_state_dict(
            pose_guider_state_dict,
        )

        freeze_params(self.vae.parameters())
        freeze_params(self.image_enc.parameters())
        freeze_params(self.reference_unet.parameters())
        freeze_params(self.denoising_unet.parameters())
        freeze_params(self.pose_guider.parameters())

        # Set motion module learnable
        for name, module in self.denoising_unet.named_sublayers():
            if "motion_modules" in name:
                unfreeze_params(module.parameters())

        self.reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=False,
            mode="write",
            fusion_blocks="full",
        )
        self.reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=False,
            mode="read",
            fusion_blocks="full",
        )

        self.noise_offset = model_args.noise_offset
        self.uncond_ratio = model_args.uncond_ratio
        self.snr_gamma = model_args.snr_gamma

        self.reference_unet.eval()
        self.denoising_unet.train()
        self.pose_guider.eval()
        self.vae.eval()
        self.image_enc.eval()

    def compute_snr(self, noise_scheduler, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[timesteps].cast("float32")
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[timesteps].cast("float32")
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def forward(self, batch, **kwargs):

        # Convert videos to latent space
        pixel_values_vid = batch["pixel_values_vid"]
        with paddle.no_grad():
            video_length = pixel_values_vid.shape[1]
            pixel_values_vid = rearrange(pixel_values_vid, "b f c h w -> (b f) c h w")
            latents = self.vae.encode(pixel_values_vid).latent_dist.sample()
            latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
            latents = latents * 0.18215

        noise = paddle.randn(shape=latents.shape, dtype=latents.dtype)

        if self.noise_offset > 0.0:
            noise += self.noise_offset * paddle.randn((noise.shape[0], noise.shape[1], 1, 1, 1))

        bsz = latents.shape[0]
        # Sample a random timestep for each video
        timesteps = paddle.randint(0, self.train_noise_scheduler.num_train_timesteps, (bsz,))

        timesteps = timesteps.astype(dtype="int64")

        pixel_values_pose = batch["pixel_values_pose"]  # (bs, f, c, H, W)

        pixel_values_pose = pixel_values_pose.transpose(perm=[0, 2, 1, 3, 4])  # (bs, c, f, H, W)

        uncond_fwd = random.random() < self.uncond_ratio

        clip_image_list = []
        ref_image_list = []

        for _, (ref_img, clip_img) in enumerate(
            zip(
                batch["pixel_values_ref_img"],
                batch["clip_ref_img"],
            )
        ):
            if uncond_fwd:
                clip_image_list.append(paddle.zeros_like(clip_img))
            else:
                clip_image_list.append(clip_img)
            ref_image_list.append(ref_img)

        with paddle.no_grad():
            ref_img = paddle.stack(ref_image_list, axis=0).astype(dtype=self.vae.dtype)
            ref_image_latents = self.vae.encode(ref_img).latent_dist.sample()  # (bs, d, 64, 64)
            ref_image_latents = ref_image_latents * 0.18215

            clip_img = paddle.stack(clip_image_list, axis=0).astype(dtype=self.image_enc._dtype)
            clip_image_embeds = self.image_enc(clip_img).image_embeds

            clip_image_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)
        # add noise
        noisy_latents = self.train_noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the target for loss depending on the prediction type
        if self.train_noise_scheduler.prediction_type == "epsilon":
            target = noise
        elif self.train_noise_scheduler.prediction_type == "v_prediction":
            target = self.train_noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.train_noise_scheduler.prediction_type}")

        pose_fea = self.pose_guider(pixel_values_pose)
        if not uncond_fwd:
            ref_timesteps = paddle.zeros_like(timesteps)

            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            pose_cond_fea=pose_fea,
            encoder_hidden_states=clip_image_embeds,
        ).sample

        if self.snr_gamma == 0:
            loss = F.mse_loss(model_pred.astype("float32"), target.astype("float32"), reduction="mean")
        else:
            snr = self.compute_snr(self.train_noise_scheduler, timesteps)
            if self.train_noise_scheduler.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = (
                paddle.stack([snr, self.snr_gamma * paddle.ones_like(timesteps)], axis=1).min(1)[0] / snr
            )
            loss = F.mse_loss(model_pred.astype("float32"), target.astype("float32"), reduction="none")
            loss = loss.mean(axis=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return loss
