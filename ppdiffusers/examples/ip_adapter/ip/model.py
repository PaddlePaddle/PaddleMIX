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
import inspect
import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
    is_ppxformers_available,
)
from ppdiffusers.models.embeddings import ImageProjection
from ppdiffusers.training_utils import freeze_params
from ppdiffusers.transformers import (
    AutoTokenizer,
    CLIPTextModel,
    CLIPVisionModelWithProjection,
)


class IPAdapterModel(nn.Layer):
    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args
        tokenizer_name_or_path = (
            model_args.tokenizer_name
            if model_args.tokenizer_name is not None
            else os.path.join(model_args.pretrained_model_name_or_path, "tokenizer")
        )
        vae_name_or_path = (
            model_args.vae_name_or_path
            if model_args.vae_name_or_path is not None
            else os.path.join(model_args.pretrained_model_name_or_path, "vae")
        )
        text_encoder_name_or_path = (
            model_args.text_encoder_name_or_path
            if model_args.text_encoder_name_or_path is not None
            else os.path.join(model_args.pretrained_model_name_or_path, "text_encoder")
        )
        unet_name_or_path = (
            model_args.unet_name_or_path
            if model_args.unet_name_or_path is not None
            else os.path.join(model_args.pretrained_model_name_or_path, "unet")
        )

        image_encoder_name_or_path = (
            model_args.image_encoder_name_or_path
            if model_args.image_encoder_name_or_path is not None
            else os.path.join(model_args.pretrained_model_name_or_path, "image_encoder")
        )

        # init model and tokenizer
        tokenizer_kwargs = {}
        if model_args.model_max_length is not None:
            tokenizer_kwargs["model_max_length"] = model_args.model_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
        self.vae = AutoencoderKL.from_pretrained(vae_name_or_path)
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_name_or_path)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_name_or_path)

        freeze_params(self.vae.parameters())
        freeze_params(self.text_encoder.parameters())
        freeze_params(self.image_encoder.parameters())
        self.vae.eval()
        self.text_encoder.eval()
        self.image_encoder.eval()

        self.unet = UNet2DConditionModel.from_pretrained(unet_name_or_path)
        self.create_ip_adapter_unet()

        # init noise_scheduler and eval_scheduler
        assert self.model_args.prediction_type in ["epsilon", "v_prediction"]
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=self.model_args.prediction_type,
        )
        self.register_buffer("alphas_cumprod", self.noise_scheduler.alphas_cumprod)
        self.eval_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type=self.model_args.prediction_type,
        )
        self.eval_scheduler.set_timesteps(self.model_args.num_inference_steps)

    def create_ip_adapter_unet(self, num_image_text_embeds=4):
        unet_sd = self.unet.state_dict()
        freeze_params(self.unet.parameters())
        from ppdiffusers.models.attention_processor import (
            AttnProcessor,
            AttnProcessor2_5,
            IPAdapterAttnProcessor,
            IPAdapterAttnProcessor2_5,
        )

        class AttnProcessor2_5_Layer(nn.Layer):
            def __init__(self, attention_op=None):
                super().__init__()
                assert attention_op in [None, "math", "auto", "flash", "cutlass", "memory_efficient"]
                self.attention_op = attention_op

            __call__ = AttnProcessor2_5.__call__

        class AttnProcessor_Layer(nn.Layer):
            def __init__(self):
                super().__init__()

            __call__ = AttnProcessor.__call__

        attn_procs = {}
        key_id = 1
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim is None or "motion_modules" in name:
                attn_processor_class = AttnProcessor2_5_Layer if is_ppxformers_available() else AttnProcessor_Layer
                attn_procs[name] = attn_processor_class()
            else:
                attn_processor_class = (
                    IPAdapterAttnProcessor2_5 if is_ppxformers_available() else IPAdapterAttnProcessor
                )
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = attn_processor_class(
                    hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, scale=1.0
                )
                attn_procs[name].load_dict(weights)
                key_id += 2

        self.unet.set_attn_processor(attn_procs)
        self.adapter_modules = nn.LayerList(self.unet.attn_processors.values())
        self.unet.encoder_hid_proj = ImageProjection(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            image_embed_dim=self.image_encoder.config.projection_dim,
            num_image_text_embeds=num_image_text_embeds,
        )

        self.unet.config.encoder_hid_dim_type = "ip_image_proj"

    def compute_snr(self, timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        sqrt_alphas_cumprod = self.alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - self.alphas_cumprod) ** 0.5

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

    def forward(self, input_ids=None, pixel_values=None, clip_images=None, drop_image_embeds=None, **kwargs):
        self.vae.eval()
        self.text_encoder.eval()
        self.image_encoder.eval()

        # vae encode
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = paddle.randn(latents.shape)
        if self.model_args.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.model_args.noise_offset * paddle.randn(
                (latents.shape[0], latents.shape[1], 1, 1), dtype=noise.dtype
            )
        if self.model_args.input_perturbation:
            new_noise = noise + self.model_args.input_perturbation * paddle.randn(noise.shape, dtype=noise.dtype)

        timesteps = paddle.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],)).cast(
            "int64"
        )
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        if self.model_args.input_perturbation:
            noisy_latents = self.add_noise(latents, new_noise, timesteps)
        else:
            noisy_latents = self.add_noise(latents, noise, timesteps)

        # process image embeddings
        image_embeds = self.image_encoder(clip_images)[0]
        image_embeds_ = []
        for image_embed, drop_image_embed in zip(image_embeds, drop_image_embeds):
            if drop_image_embed == 1:
                image_embeds_.append(paddle.zeros_like(image_embed))
            else:
                image_embeds_.append(image_embed)
        image_embeds = paddle.stack(image_embeds_)
        added_cond_kwargs = {"image_embeds": image_embeds}

        # text encode
        encoder_hidden_states = self.text_encoder(input_ids)[0]

        # unet
        model_pred = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )[0]

        # Get the target for loss depending on the prediction type
        if self.model_args.prediction_type == "epsilon":
            target = noise
        elif self.model_args.prediction_type == "v_prediction":
            target = self.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.model_args.prediction_type}")

        # compute loss
        if self.model_args.snr_gamma is None:
            loss = (
                F.mse_loss(model_pred.cast("float32"), target.cast("float32"), reduction="none").mean([1, 2, 3]).mean()
            )
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                paddle.stack([snr, self.model_args.snr_gamma * paddle.ones_like(timesteps)], axis=1,).min(
                    axis=1
                )[0]
                / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = F.mse_loss(model_pred.cast("float32"), target.cast("float32"), reduction="none")
            loss = loss.mean(list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        return loss

    def add_noise(
        self,
        original_samples: paddle.Tensor,
        noise: paddle.Tensor,
        timesteps: paddle.Tensor,
    ) -> paddle.Tensor:
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def get_velocity(self, sample: paddle.Tensor, noise: paddle.Tensor, timesteps: paddle.Tensor) -> paddle.Tensor:
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    @paddle.no_grad()
    def decode_image(self, pixel_values=None, max_batch=8, **kwargs):
        self.eval()
        if pixel_values.shape[0] > max_batch:
            pixel_values = pixel_values[:max_batch]
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1])
        image = (image * 255.0).cast("float32").numpy().round()
        return image

    @paddle.no_grad()
    def log_image(
        self,
        input_ids=None,
        clip_images=None,
        height=512,
        width=512,
        eta=0.0,
        guidance_scale=7.5,
        max_batch=8,
        **kwargs,
    ):
        self.eval()
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
        # only log max_batch image
        if input_ids.shape[0] > max_batch:
            input_ids = input_ids[:max_batch]
            clip_images = clip_images[:max_batch]
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

        image_embeds = self.image_encoder(clip_images)[0]
        if do_classifier_free_guidance:
            negative_image_embeds = paddle.zeros_like(image_embeds)
            image_embeds = paddle.concat([negative_image_embeds, image_embeds])

        added_cond_kwargs = {"image_embeds": image_embeds}

        latents = paddle.randn((input_ids.shape[0], self.unet.config.in_channels, height // 8, width // 8))
        latents = latents * self.eval_scheduler.init_noise_sigma
        accepts_eta = "eta" in set(inspect.signature(self.eval_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        for t in self.eval_scheduler.timesteps:
            latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.eval_scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings, added_cond_kwargs=added_cond_kwargs
            ).sample
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.eval_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1]) * 255.0
        return image.cast("float32").numpy().round()

    def set_recompute(self, use_recompute=False):
        if use_recompute:
            self.unet.enable_gradient_checkpointing()

    def gradient_checkpointing_enable(self):
        self.set_recompute(True)
