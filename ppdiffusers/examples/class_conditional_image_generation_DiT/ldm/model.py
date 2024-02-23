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
import contextlib
import inspect
import os
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .models import DiT
from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    is_ppxformers_available,
)
from ppdiffusers.models.attention import AttentionBlock
from ppdiffusers.models.ema import LitEma
from ppdiffusers.training_utils import freeze_params

try:
    from ppdiffusers.models.attention import SpatialTransformer
except ImportError:
    from ppdiffusers.models.transformer_2d import (
        Transformer2DModel as SpatialTransformer,
    )

import json
from paddlenlp.utils.log import logger
from ppdiffusers.initializer import normal_, reset_initialized_parameter, zeros_
from ppdiffusers.models.resnet import ResnetBlock2D


def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return paddle.mean(axis=list(range(1, len(tensor.shape))))


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = paddle.to_tensor(arr)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + paddle.zeros(broadcast_shape) #, device=timesteps.device)


class DiTDiffusionModel(nn.Layer):
    def __init__(self, model_args):
        super().__init__()
        # init vae
        vae_name_or_path = (
            model_args.vae_name_or_path
            if model_args.pretrained_model_name_or_path is None
            else os.path.join(model_args.pretrained_model_name_or_path, "vqvae")
        )
        self.vae = AutoencoderKL.from_pretrained(vae_name_or_path)
        freeze_params(self.vae.parameters())
        logger.info("Freeze vae parameters!")

        #self.transformer = DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16)
        self.transformer = DiT(**read_json(model_args.unet_config_file))
        self.transformer_is_pretrained = True

        assert model_args.prediction_type in ["epsilon", "v_prediction"]
        self.prediction_type = model_args.prediction_type
        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            prediction_type=self.prediction_type,
        )
        self.register_buffer("alphas_cumprod", self.noise_scheduler.alphas_cumprod)

        if model_args.image_logging_steps > 0:
            self.eval_scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                num_train_timesteps=1000,
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
                prediction_type=self.prediction_type,
            )
            self.eval_scheduler.set_timesteps(model_args.num_inference_steps)
        # self.init_weights()
        self.use_ema = model_args.use_ema
        self.noise_offset = model_args.noise_offset
        if self.use_ema:
            self.model_ema = LitEma(self.transformer)

        # if model_args.enable_xformers_memory_efficient_attention and is_ppxformers_available():
        #     try:
        #         self.unet.enable_xformers_memory_efficient_attention()
        #     except Exception as e:
        #         logger.warn(
        #             "Could not enable memory efficient attention. Make sure develop paddlepaddle is installed"
        #             f" correctly and a GPU is available: {e}"
        #         )

        # make sure unet text_encoder in train mode, vae in eval mode
        self.transformer.train()
        #self.text_encoder.train()
        self.vae.eval()

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

    def init_weights(self):
        # init text_encoder
        # if not self.text_encoder_is_pretrained:
        #     reset_initialized_parameter(self.text_encoder)
        #     normal_(self.text_encoder.embeddings.word_embeddings.weight, 0, 0.02)
        #     normal_(self.text_encoder.embeddings.position_embeddings.weight, 0, 0.02)
        # init unet
        if not self.transformer_is_pretrained:
            reset_initialized_parameter(self.transformer)
            # zeros_(self.transformer.conv_out.weight)
            # zeros_(self.transformer.conv_out.bias)
            for _, m in self.transformer.named_sublayers():
                if isinstance(m, AttentionBlock):
                    zeros_(m.proj_attn.weight)
                    zeros_(m.proj_attn.bias)
                if isinstance(m, ResnetBlock2D):
                    zeros_(m.conv2.weight)
                    zeros_(m.conv2.bias)
                if isinstance(m, SpatialTransformer):
                    zeros_(m.proj_out.weight)
                    zeros_(m.proj_out.bias)

    @contextlib.contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.transformer.parameters())
            self.model_ema.copy_to(self.transformer)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.transformer.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self):
        if self.use_ema:
            self.model_ema(self.transformer)

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = paddle.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, latents=None, label_id=None, **kwargs):
        x_start = latents
        timesteps = paddle.randint(0, self.noise_scheduler.num_train_timesteps, (latents.shape[0],)).astype(
            "int64"
        )
        
        self.vae.eval()
        noise = paddle.randn(latents.shape)
        x_t = self.q_sample(latents, timesteps, noise=noise)

        model_output = self.transformer(x=x_t, t=timesteps, y=label_id) #.sample

        # # Get the target for loss depending on the prediction type
        # if self.prediction_type == "epsilon":
        #     target = noise
        # elif self.prediction_type == "v_prediction":
        #     target = self.get_velocity(latents, noise, timesteps)
        # else:
        #     raise ValueError(f"Unknown prediction type {self.prediction_type}")

        if 1:
            B, C = x_t.shape[:2]
            assert model_output.shape == (B, C * 2, *x_t.shape[2:])
            model_output, model_var_values = paddle.split(model_output, C, axis=1)
            # Learn the variance using the variational bound, but don't let
            # it affect our mean prediction.
            frozen_out = paddle.concat([model_output, model_var_values], axis=1)
            vb_loss = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=latents,
                    x_t=x_t,
                    t=timesteps,
                    clip_denoised=False,
                )["output"]

        target = noise # EPSILON
        assert model_output.shape == target.shape == x_start.shape
        mse_loss = mean_flat((target - model_output) ** 2)
        if 1:
            loss = mse_loss + vb_loss
        else:
            loss = mse_loss
        #loss = F.mse_loss(model_output.cast("float32"), target.cast("float32"), reduction="none").mean([1, 2, 3]).mean()
        return loss

    @paddle.no_grad()
    def decode_image(self, pixel_values=None, **kwargs):
        self.eval()
        if pixel_values.shape[0] > 8:
            pixel_values = pixel_values[:8]
        latents = self.vae.encode(pixel_values).latent_dist.sample()
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1])
        image = (image * 255.0).cast("float32").numpy().round()
        return image

    @paddle.no_grad()
    def log_image(
        self,
        input_ids=None,
        height=256,
        width=256,
        eta=0.0,
        guidance_scale=7.5,
        **kwargs,
    ):
        self.eval()
        with self.ema_scope():
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
            # only log 8 image
            if input_ids.shape[0] > 8:
                input_ids = input_ids[:8]

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

            latents = paddle.randn((input_ids.shape[0], self.transformer.in_channels, height // 8, width // 8))
            # ddim donot use this
            latents = latents * self.eval_scheduler.init_noise_sigma

            accepts_eta = "eta" in set(inspect.signature(self.eval_scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = eta

            for t in self.eval_scheduler.timesteps:
                # expand the latents if we are doing classifier free guidance
                latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
                # ddim donot use this
                latent_model_input = self.eval_scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.transformer(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.eval_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            latents = 1 / 0.18215 * latents
            image = self.vae.decode(latents).sample
            image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1]) * 255.0
        return image.cast("float32").numpy().round()

    def set_recompute(self, value=False):
        def fn(layer):
            # ldmbert
            if hasattr(layer, "enable_recompute"):
                layer.enable_recompute = value
                print("Set", layer.__class__, "recompute", layer.enable_recompute)
            # unet
            if hasattr(layer, "gradient_checkpointing"):
                layer.gradient_checkpointing = value
                print("Set", layer.__class__, "recompute", layer.gradient_checkpointing)

        self.apply(fn)
