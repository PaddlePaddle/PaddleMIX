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
import json
import os

import einops
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddle.nn.initializer import TruncatedNormal
from paddlenlp.transformers import AutoTokenizer, CLIPTextModel
from paddlenlp.utils.log import logger

from ppdiffusers import AutoencoderKL, DDIMScheduler, is_ppxformers_available
from ppdiffusers.initializer import reset_initialized_parameter, zeros_
from ppdiffusers.models.attention_processor import Attention
from ppdiffusers.models.ema import LitEma
from ppdiffusers.models.vae import DiagonalGaussianDistribution
from ppdiffusers.training_utils import freeze_params

from .uvit_t2i import UViTT2IModel

trunc_normal_ = TruncatedNormal(std=0.02)


def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = paddle.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=paddle.float64) ** 2
    return _betas.numpy()


def get_skip(alphas, betas):
    N = len(betas) - 1
    skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1 :] = alphas[s + 1 :].cumprod()
    skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1 : t + 1] * skip_alphas[1 : t + 1, t]
        skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]
    return skip_alphas, skip_betas


def stp(s, ts: paddle.Tensor):
    # scalar tensor product
    if isinstance(s, np.ndarray):
        s = paddle.to_tensor(s).cast(ts.dtype)
    extra_dims = (1,) * (len(ts.shape) - 1)
    return s.reshape([-1, *extra_dims]) * ts


class Schedule(object):
    # discrete time
    def __init__(self, _betas):
        r"""_betas[0...999] = betas[1...1000]
        for n>=1, betas[n] is the variance of q(xn|xn-1)
        for n=0,  betas[0]=0
        """

        self._betas = _betas
        self.betas = np.append(0.0, _betas)
        self.alphas = 1.0 - self.betas
        self.N = len(_betas)

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        # skip_alphas[s, t] = alphas[s + 1: t + 1].prod()
        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def sample(self, x0):
        # sample from q(xn|x0), where n is uniform
        n = np.random.choice(list(range(1, self.N + 1)), (len(x0),))
        eps = paddle.randn(shape=x0.shape)
        xn = stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps)
        return paddle.to_tensor(n), eps, xn

    def __repr__(self):
        return f"Schedule({self.betas[:10]}..., {self.N})"


class LatentDiffusionModel(nn.Layer):
    def __init__(self, model_args):
        super().__init__()
        self.model_args = model_args

        # init unconditional embeddings for CFG
        self.uncond_embeddings_path = f"{model_args.feature_path}/empty_context.npy"

        # init text_encoder
        text_encoder_name_or_path = (
            model_args.text_encoder_name_or_path
            if model_args.text_encoder_name_or_path is not None
            else os.path.join(model_args.pretrained_model_name_or_path, "text_encoder")
        )
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_name_or_path)

        # init tokenizer
        tokenizer_name_or_path = (
            model_args.tokenizer_name
            if model_args.pretrained_model_name_or_path is None
            else os.path.join(model_args.pretrained_model_name_or_path, "tokenizer")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, model_max_length=model_args.model_max_length
        )

        # init vae
        vae_name_or_path = (
            model_args.vae_name_or_path
            if model_args.pretrained_model_name_or_path is None
            else os.path.join(model_args.pretrained_model_name_or_path, "vqvae")
        )
        self.vae = AutoencoderKL.from_pretrained(vae_name_or_path)
        self.autoencoder_scale = 0.23010  # not 0.18215

        # init unet2d(uvit)
        if model_args.pretrained_model_name_or_path is None:
            self.unet = UViTT2IModel(**read_json(model_args.uvit_config_file))
            self.init_uvit_weights()
            logger.info("Init uvit model from scratch!")
        else:
            self.unet = UViTT2IModel.from_pretrained(model_args.pretrained_model_name_or_path, subfolder="unet")

        # make sure unet in train mode, vae and text_encoder in eval mode
        freeze_params(self.vae.parameters())
        logger.info("Freeze vae parameters!")
        if not self.model_args.train_text_encoder:
            freeze_params(self.text_encoder.parameters())
            logger.info("Freeze text_encoder parameters!")
            self.text_encoder.eval()
            self.train_text_encoder = False
        else:
            self.text_encoder.train()
            self.train_text_encoder = True
        self.unet.train()
        self.vae.eval()

        self.use_ema = model_args.use_ema
        self.model_ema = None
        if self.use_ema:
            self.model_ema = LitEma(self.unet, decay=0.9999)

        if model_args.enable_xformers_memory_efficient_attention and is_ppxformers_available():
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(
                    "Could not enable memory efficient attention. Make sure develop paddlepaddle is installed"
                    f" correctly and a GPU is available: {e}"
                )

        # init schedule
        _betas = stable_diffusion_beta_schedule()
        self._schedule = Schedule(_betas)

        # init eval_scheduler
        assert model_args.prediction_type in ["epsilon", "v_prediction"]  # default epsilon
        self.prediction_type = model_args.prediction_type
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

    def init_uvit_weights(self):
        reset_initialized_parameter(self.unet)
        for _, m in self.unet.named_sublayers():
            if isinstance(m, Attention) and getattr(m, "group_norm", None) is not None:
                zeros_(m.to_out[0].weight)
                zeros_(m.to_out[0].bias)
            if isinstance(
                m, (nn.Linear, fleet.meta_parallel.ColumnParallelLinear, fleet.meta_parallel.RowParallelLinear)
            ):
                trunc_normal_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    zeros_(m.bias)

    @contextlib.contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.unet.parameters())
            self.model_ema.copy_to(self.unet)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.unet.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self):
        if self.use_ema:
            self.model_ema(self.unet)

    def forward(self, input_ids=None, pixel_values=None, **kwargs):
        # input_ids [16, 77, 768] pixel_values [16, 8, 32, 32]
        with paddle.no_grad():
            self.vae.eval()
            latents = DiagonalGaussianDistribution(pixel_values).sample()  # [16, 8, 32, 32] -> [16, 4, 32, 32]
            latents = latents * self.autoencoder_scale  # Note: 0.23010, not 0.18215
            timesteps, noise, noisy_latents = self._schedule.sample(latents)  # [16] [16, 4, 32, 32] [16, 4, 32, 32]

        noise_pred = self.unet(noisy_latents, timesteps, input_ids)

        # Get the target for loss depending on the prediction type
        if self.prediction_type == "epsilon":
            target = noise
        else:
            raise ValueError(f"Unknown prediction type {self.prediction_type}")

        loss = F.mse_loss(noise_pred, target, reduction="none").mean([1, 2, 3]).mean()
        return loss

    @paddle.no_grad()
    def decode_image(self, pixel_values=None, **kwargs):
        self.eval()
        if pixel_values.shape[0] > 8:
            pixel_values = pixel_values[:8]
        # latents = self.vae.encode(pixel_values).latent_dist.sample()
        latents = DiagonalGaussianDistribution(pixel_values).sample()
        # latents = 1 / self.autoencoder_scale * latents # add or not?
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

            text_embeddings = input_ids  # [8, 77, 768]

            do_classifier_free_guidance = guidance_scale > 0.0  # not 1.0
            if do_classifier_free_guidance:
                uncond_embeddings = paddle.to_tensor(np.load(self.uncond_embeddings_path))
                uncond_embeddings = einops.repeat(uncond_embeddings, "L D -> B L D", B=input_ids.shape[0])
                text_embeddings = paddle.concat([uncond_embeddings, text_embeddings], axis=0)

            latents = paddle.randn(
                (input_ids.shape[0], self.unet.in_channels, height // 8, width // 8)
            )  # [bs, 4, 32, 32]
            # ddim do not use this
            latents = latents * self.eval_scheduler.init_noise_sigma  # 1.0

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
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)  # .sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.eval_scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            latents = 1 / self.autoencoder_scale * latents
            image = self.vae.decode(latents).sample
            image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1]) * 255.0
        return image.cast("float32").numpy().round()

    def set_recompute(self, use_recompute=False):
        if use_recompute:
            self.unet.enable_gradient_checkpointing()
            if self.model_args.train_text_encoder and hasattr(self.text_encoder, "gradient_checkpointing_enable"):
                self.text_encoder.gradient_checkpointing_enable()

    def gradient_checkpointing_enable(self):
        self.set_recompute(True)

    def set_xformers(self, use_xformers=False):
        if use_xformers:
            if not is_ppxformers_available():
                raise ValueError(
                    'Please run `python -m pip install "paddlepaddle-gpu>=2.6.0" -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html first.'
                )
            else:
                try:
                    attention_op = os.getenv("FLAG_XFORMERS_ATTENTION_OP", "none").lower()

                    if attention_op == "none":
                        attention_op = None

                    self.unet.enable_xformers_memory_efficient_attention(attention_op)
                    if hasattr(self.vae, "enable_xformers_memory_efficient_attention"):
                        self.vae.enable_xformers_memory_efficient_attention(attention_op)
                    if hasattr(self.text_encoder, "enable_xformers_memory_efficient_attention"):
                        self.text_encoder.enable_xformers_memory_efficient_attention(attention_op)
                except Exception as e:
                    logger.warning(
                        "Could not enable memory efficient attention. Make sure develop paddlepaddle is installed"
                        f" correctly and a GPU is available: {e}"
                    )
        else:
            if hasattr(self.unet, "set_default_attn_processor"):
                self.unet.set_default_attn_processor()
            if hasattr(self.vae, "set_default_attn_processor"):
                self.vae.set_default_attn_processor()

    def set_ema(self, use_ema=False):
        self.use_ema = use_ema
        if use_ema:
            self.model_ema = LitEma(self.unet)
