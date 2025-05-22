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

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
from paddlenlp.utils.log import logger

from ppdiffusers import AutoencoderKL, DDIMScheduler, is_ppxformers_available
from ppdiffusers.models.ema import LitEma
from ppdiffusers.training_utils import freeze_params

from .diffusion_utils import discretized_gaussian_log_likelihood, get_mesh, normal_kl
from .dit_auto import DiT_AUTO
from .dit_llama_auto import DiT_Llama_AUTO
from .gaussian_diffusion import _extract_into_tensor, get_named_beta_schedule, mean_flat


def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


class DiTDiffusionModelAuto(nn.Layer):
    def __init__(self, model_args, training_args):
        super().__init__()
        self.model_args = model_args

        self.mp_degree = training_args.tensor_parallel_degree
        # self.mp_group, self.mp_rank = build_mp_group(self.mp_degree)

        # init vae
        vae_name_or_path = (
            model_args.vae_name_or_path
            if model_args.pretrained_model_name_or_path is None
            else os.path.join(model_args.pretrained_model_name_or_path, "vae")
        )
        self.vae = AutoencoderKL.from_pretrained(vae_name_or_path)

        # init DiT
        if model_args.pretrained_model_name_or_path is None:
            if model_args.config_file.startswith("config/LargeDiT_"):
                self.transformer = DiT_Llama_AUTO(**read_json(model_args.config_file))
            else:
                self.transformer = DiT_AUTO(**read_json(model_args.config_file))
            # Note: Initialize DiT in diffusion/dit.py
            logger.info("Init DiT model from scratch!")
        else:
            if model_args.config_file.startswith("config/LargeDiT_"):
                self.transformer = DiT_Llama_AUTO.from_pretrained(
                    os.path.join(model_args.pretrained_model_name_or_path, "transformer")
                )
            else:
                self.transformer = DiT_AUTO.from_pretrained(
                    os.path.join(model_args.pretrained_model_name_or_path, "transformer")
                )

            logger.info(f"Init DiT model from {model_args.pretrained_model_name_or_path}!")

        # make sure unet in train mode, vae and text_encoder in eval mode
        freeze_params(self.vae.parameters())
        logger.info("Freeze vae parameters!")
        self.vae.eval()
        self.transformer.train()

        self.use_ema = False
        self.model_ema = None
        if self.use_ema:
            self.model_ema = LitEma(self.transformer)

        if model_args.enable_xformers_memory_efficient_attention and is_ppxformers_available():
            try:
                self.transformer.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(
                    "Could not enable memory efficient attention. Make sure develop paddlepaddle is installed"
                    f" correctly and a GPU is available: {e}"
                )

        # other settings
        self.model_mean_type = "epsilon"  # PREVIOUS_X START_X EPSILON
        self.model_var_type = "learned_range"  # LEARNED FIXED_SMALL FIXED_LARGE LEARNED_RANGE
        self.loss_type = "mse"  # MSE RESCALED_MSE KL(is_vb) RESCALED_KL(is_vb)

        # init scheduler
        betas = get_named_beta_schedule("linear", 1000)
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = (
            np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
            if len(self.posterior_variance) > 1
            else np.array([])
        )
        self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)

        # init eval_scheduler
        assert model_args.prediction_type in ["epsilon", "v_prediction"]
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

    def forward(self, latents=None, label_id=None, **kwargs):
        if isinstance(latents, list):
            latents, label_id = latents

        x_start = latents
        timesteps = paddle.randint(0, self.num_timesteps, (latents.shape[0],))
        timesteps = dist.shard_tensor(timesteps, label_id.process_mesh, label_id.placements)  # as label_id

        self.vae.eval()
        noise = paddle.randn(latents.shape)
        noise = dist.shard_tensor(noise, latents.process_mesh, latents.placements)  # as latents
        x_t = self.q_sample(latents, timesteps, noise=noise)

        model_output = self.transformer(x=x_t, t=timesteps, y=label_id)

        # Get the target for loss depending on the prediction type
        if self.prediction_type == "epsilon":  # default
            target = noise
            target = dist.reshard(target, get_mesh(-1), [dist.Shard(0), dist.Replicate()])
        else:
            raise ValueError(f"Unknown prediction type {self.prediction_type}")

        if self.loss_type == "mse":
            B, C = x_t.shape[:2]
            assert model_output.shape == [B, C * 2, *x_t.shape[2:]]
            model_output, model_var_values = paddle.split(model_output, 2, axis=1)
            # Learn the variance using the variational bound, but don't let
            # it affect our mean prediction.
            frozen_out = paddle.concat([model_output.detach(), model_var_values], axis=1)
            vb_loss = self._vb_terms_bpd(
                model=lambda *args, r=frozen_out: r,
                x_start=latents,
                x_t=x_t,
                t=timesteps,
                clip_denoised=False,
            )["output"]

        assert model_output.shape == target.shape == x_start.shape
        mse_loss = mean_flat((target - model_output) ** 2)
        if self.loss_type == "mse":
            loss = mse_loss + vb_loss
        else:
            loss = mse_loss
        return loss.mean().astype("float32")

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
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, noise.shape) * noise
        )

    def _vb_terms_bpd(self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        x_start = dist.reshard(x_start, get_mesh(-1), [dist.Shard(0), dist.Replicate()])
        x_t = dist.reshard(x_t, get_mesh(-1), [dist.Shard(0), dist.Replicate()])
        t = dist.reshard(t, get_mesh(-1), [dist.Shard(0), dist.Replicate()])

        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_variance(model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = paddle.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == [B]
        model_output = model(x, t, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None

        if self.model_var_type in ["learned_range"]:
            assert model_output.shape == [B, C * 2, *x.shape[2:]]
            model_output, model_var_values = paddle.split(model_output, 2, axis=1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = paddle.exp(model_log_variance)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clip(-1, 1)
            return x

        if self.model_mean_type == "start_x":
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, eps.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_start.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @paddle.no_grad()
    def log_image(
        self,
        input_ids=None,
        height=256,
        width=256,
        eta=0.0,
        class_labels=None,
        guidance_scale=4.0,
        max_batch=8,
        **kwargs,
    ):
        self.eval()
        with self.ema_scope():
            assert input_ids is None
            if height % 8 != 0 or width % 8 != 0:
                raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
            if class_labels.shape[0] > max_batch:
                class_labels = class_labels[:max_batch]
            batch_size = class_labels.shape[0]
            latent_channels = self.transformer.in_channels

            latents = paddle.randn((class_labels.shape[0], self.transformer.in_channels, height // 8, width // 8))
            latent_model_input = paddle.concat([latents] * 2) if guidance_scale > 1 else latents

            class_null = paddle.to_tensor([1000] * batch_size)
            class_labels_input = paddle.concat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

            accepts_eta = "eta" in set(inspect.signature(self.eval_scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = eta

            for t in self.eval_scheduler.timesteps:
                if guidance_scale > 1:
                    half = latent_model_input[: len(latent_model_input) // 2]
                    latent_model_input = paddle.concat([half, half], axis=0)
                latent_model_input = self.eval_scheduler.scale_model_input(latent_model_input, t)

                timesteps = t
                if not paddle.is_tensor(timesteps):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
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
                # predict noise model_output
                noise_pred = self.transformer(x=latent_model_input, t=timesteps, y=class_labels_input)

                # perform guidance
                if guidance_scale > 1:
                    eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                    bs = eps.shape[0]
                    # TODO torch.split vs paddle.split
                    cond_eps, uncond_eps = paddle.split(eps, [bs // 2, bs - bs // 2], axis=0)

                    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                    eps = paddle.concat([half_eps, half_eps], axis=0)

                    noise_pred = paddle.concat([eps, rest], axis=1)

                # learned sigma
                if self.transformer.out_channels // 2 == latent_channels:
                    # TODO torch.split vs paddle.split
                    model_output, _ = paddle.split(
                        noise_pred, [latent_channels, noise_pred.shape[1] - latent_channels], axis=1
                    )
                else:
                    model_output = noise_pred

                # compute previous image: x_t -> x_t-1
                latent_model_input = self.eval_scheduler.step(model_output, t, latent_model_input).prev_sample

            if guidance_scale > 1:
                latents, _ = latent_model_input.chunk(2, axis=0)
            else:
                latents = latent_model_input

            latents = 1 / self.vae.config.scaling_factor * latents
            image = self.vae.decode(latents).sample
            image = (image / 2 + 0.5).clip(0, 1).transpose([0, 2, 3, 1]) * 255.0
        return image.cast("float32").numpy().round()

    def set_recompute(self, use_recompute=False):
        if use_recompute:
            self.transformer.enable_gradient_checkpointing()

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

                    self.transformer.enable_xformers_memory_efficient_attention(attention_op)
                    if hasattr(self.vae, "enable_xformers_memory_efficient_attention"):
                        self.vae.enable_xformers_memory_efficient_attention(attention_op)
                except Exception as e:
                    logger.warning(
                        "Could not enable memory efficient attention. Make sure develop paddlepaddle is installed"
                        f" correctly and a GPU is available: {e}"
                    )
        else:
            if hasattr(self.transformer, "set_default_attn_processor"):
                self.transformer.set_default_attn_processor()
            if hasattr(self.vae, "set_default_attn_processor"):
                self.vae.set_default_attn_processor()

    def set_ema(self, use_ema=False):
        self.use_ema = use_ema
        if use_ema:
            self.model_ema = LitEma(self.transformer)
