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

from __future__ import annotations

import sys, os
from collections import deque
from functools import partial
from typing import List, Tuple

import numpy as np

import paddle
from tqdm import tqdm

from paddlemix.models.diffsinger.modules.backbones import build_backbone
from paddlemix.models.diffsinger.utils.hparams import hparams


def extract(a, t, x_shape):
    b, *_ = tuple(t.shape)
    out = a.take_along_axis(axis=-1, indices=t, broadcast=False)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: paddle.randn(shape=(1, *shape[1:])).tile(
        repeat_times=[shape[0], *((1,) * (len(shape) - 1))]
    )
    noise = lambda: paddle.randn(shape=shape)
    return repeat_noise() if repeat else noise()


def linear_beta_schedule(timesteps, max_beta=0.01):
    """
    linear schedule
    """
    betas = np.linspace(0.0001, max_beta, timesteps)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos((x / steps + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return np.clip(betas, a_min=0, a_max=0.999)


beta_schedule = {"cosine": cosine_beta_schedule, "linear": linear_beta_schedule}


class GaussianDiffusion(paddle.nn.Layer):
    def __init__(
        self,
        out_dims,
        num_feats=1,
        timesteps=1000,
        k_step=1000,
        backbone_type=None,
        backbone_args=None,
        betas=None,
        spec_min=None,
        spec_max=None,
    ):
        super().__init__()
        self.denoise_fn: paddle.nn.Layer = build_backbone(out_dims, num_feats, backbone_type, backbone_args)
        self.out_dims = out_dims
        self.num_feats = num_feats
        if betas is not None:
            betas = betas.detach().cpu().numpy() if isinstance(betas, paddle.Tensor) else betas
        else:
            betas = beta_schedule[hparams["schedule_type"]](timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.use_shallow_diffusion = hparams.get("use_shallow_diffusion", False)
        if self.use_shallow_diffusion:
            assert k_step <= timesteps, "K_step should not be larger than timesteps."
        self.timesteps = timesteps
        self.k_step = k_step if self.use_shallow_diffusion else timesteps
        self.noise_list = deque(maxlen=4)
        to_torch = partial(paddle.to_tensor, dtype="float32")
        self.register_buffer(name="betas", tensor=to_torch(betas))
        self.register_buffer(name="alphas_cumprod", tensor=to_torch(alphas_cumprod))
        self.register_buffer(name="alphas_cumprod_prev", tensor=to_torch(alphas_cumprod_prev))
        self.register_buffer(name="sqrt_alphas_cumprod", tensor=to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(name="sqrt_one_minus_alphas_cumprod", tensor=to_torch(np.sqrt(1.0 - alphas_cumprod)))
        self.register_buffer(name="log_one_minus_alphas_cumprod", tensor=to_torch(np.log(1.0 - alphas_cumprod)))
        self.register_buffer(name="sqrt_recip_alphas_cumprod", tensor=to_torch(np.sqrt(1.0 / alphas_cumprod)))
        self.register_buffer(name="sqrt_recipm1_alphas_cumprod", tensor=to_torch(np.sqrt(1.0 / alphas_cumprod - 1)))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer(name="posterior_variance", tensor=to_torch(posterior_variance))
        self.register_buffer(
            name="posterior_log_variance_clipped", tensor=to_torch(np.log(np.maximum(posterior_variance, 1e-20)))
        )
        self.register_buffer(
            name="posterior_mean_coef1", tensor=to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        )
        self.register_buffer(
            name="posterior_mean_coef2",
            tensor=to_torch((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)),
        )
        spec_min = paddle.to_tensor(data=spec_min, dtype="float32")[None, None, :out_dims].transpose(
            perm=paddle_aux.transpose_aux_func(
                paddle.to_tensor(data=spec_min, dtype="float32")[None, None, :out_dims].ndim, -3, -2
            )
        )
        spec_max = paddle.to_tensor(data=spec_max, dtype="float32")[None, None, :out_dims].transpose(
            perm=paddle_aux.transpose_aux_func(
                paddle.to_tensor(data=spec_max, dtype="float32")[None, None, :out_dims].ndim, -3, -2
            )
        )
        self.register_buffer(name="spec_min", tensor=spec_min)
        self.register_buffer(name="spec_max", tensor=spec_max)
        self.time_scale_factor = self.timesteps
        self.t_start = 1 - self.k_step / self.timesteps
        factors = paddle.to_tensor(
            data=[i for i in range(1, self.timesteps + 1) if self.timesteps % i == 0], dtype="int64"
        )
        self.register_buffer(name="timestep_factors", tensor=factors, persistable=False)

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, tuple(x_start.shape)) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, tuple(x_start.shape))
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, tuple(x_start.shape))
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, tuple(x_t.shape)) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, tuple(x_t.shape)) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, tuple(x_t.shape)) * x_start
            + extract(self.posterior_mean_coef2, t, tuple(x_t.shape)) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, tuple(x_t.shape))
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, tuple(x_t.shape))
        return (posterior_mean, posterior_variance, posterior_log_variance_clipped)

    def p_mean_variance(self, x, t, cond):
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @paddle.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        b, *_, device = *tuple(x.shape), x.place
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond)
        noise = noise_like(tuple(x.shape), device, repeat_noise)
        nonzero_mask = (1 - (t == 0).astype(dtype="float32")).reshape(b, *((1,) * (len(tuple(x.shape)) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @paddle.no_grad()
    def p_sample_ddim(self, x, t, interval, cond):
        a_t = extract(self.alphas_cumprod, t, tuple(x.shape))
        a_prev = extract(self.alphas_cumprod, paddle_aux.max(t - interval, paddle.zeros_like(x=t)), tuple(x.shape))
        noise_pred = self.denoise_fn(x, t, cond=cond)
        x_prev = a_prev.sqrt() * (
            x / a_t.sqrt() + (((1 - a_prev) / a_prev).sqrt() - ((1 - a_t) / a_t).sqrt()) * noise_pred
        )
        return x_prev

    @paddle.no_grad()
    def p_sample_plms(self, x, t, interval, cond, clip_denoised=True, repeat_noise=False):
        """
        Use the PLMS method from
        [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778).
        """

        def get_x_pred(x, noise_t, t):
            a_t = extract(self.alphas_cumprod, t, tuple(x.shape))
            a_prev = extract(self.alphas_cumprod, paddle_aux.max(t - interval, paddle.zeros_like(x=t)), tuple(x.shape))
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()
            x_delta = (a_prev - a_t) * (
                1 / (a_t_sq * (a_t_sq + a_prev_sq)) * x
                - 1 / (a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t
            )
            x_pred = x + x_delta
            return x_pred

        noise_list = self.noise_list
        noise_pred = self.denoise_fn(x, t, cond=cond)
        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(x_pred, max(t - interval, 0), cond=cond)
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2
        elif len(noise_list) == 1:
            noise_pred_prime = (3 * noise_pred - noise_list[-1]) / 2
        elif len(noise_list) == 2:
            noise_pred_prime = (23 * noise_pred - 16 * noise_list[-1] + 5 * noise_list[-2]) / 12
        else:
            noise_pred_prime = (55 * noise_pred - 59 * noise_list[-1] + 37 * noise_list[-2] - 9 * noise_list[-3]) / 24
        x_prev = get_x_pred(x, noise_pred_prime, t)
        noise_list.append(noise_pred)
        return x_prev

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, tuple(x_start.shape)) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, tuple(x_start.shape)) * noise
        )

    def p_losses(self, x_start, t, cond, noise=None):
        if noise is None:
            noise = paddle.randn(shape=x_start.shape, dtype=x_start.dtype)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t, cond)
        return x_recon, noise

    def inference(self, cond, b=1, x_start=None, device=None):
        depth = hparams.get("K_step_infer", self.k_step)
        speedup = hparams["diff_speedup"]
        if speedup > 0:
            assert depth % speedup == 0, f"Acceleration ratio must be a factor of diffusion depth {depth}."
        noise = paddle.randn(shape=[b, self.num_feats, self.out_dims, tuple(cond.shape)[2]])
        if self.use_shallow_diffusion:
            t_max = min(depth, self.k_step)
        else:
            t_max = self.k_step
        if t_max >= self.timesteps:
            x = noise
        elif t_max > 0:
            assert x_start is not None, "Missing shallow diffusion source."
            x = self.q_sample(x_start, paddle.full(shape=(b,), fill_value=t_max - 1, dtype="int64"), noise)
        else:
            assert x_start is not None, "Missing shallow diffusion source."
            x = x_start
        if speedup > 1 and t_max > 0:
            algorithm = hparams["diff_accelerator"]
            if algorithm == "dpm-solver":
                from inference.dpm_solver_pytorch import (
                    DPM_Solver,
                    NoiseScheduleVP,
                    model_wrapper,
                )

                noise_schedule = NoiseScheduleVP(schedule="discrete", betas=self.betas[:t_max])

                def my_wrapper(fn):
                    def wrapped(x, t, **kwargs):
                        ret = fn(x, t, **kwargs)
                        self.bar.update(1)
                        return ret

                    return wrapped

                model_fn = model_wrapper(
                    my_wrapper(self.denoise_fn), noise_schedule, model_type="noise", model_kwargs={"cond": cond}
                )
                dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                steps = t_max // hparams["diff_speedup"]
                self.bar = tqdm(desc="sample time step", total=steps, disable=not hparams["infer"], leave=False)
                x = dpm_solver.sample(x, steps=steps, order=2, skip_type="time_uniform", method="multistep")
                self.bar.close()
            elif algorithm == "unipc":
                from inference.uni_pc import NoiseScheduleVP, UniPC, model_wrapper

                noise_schedule = NoiseScheduleVP(schedule="discrete", betas=self.betas[:t_max])

                def my_wrapper(fn):
                    def wrapped(x, t, **kwargs):
                        ret = fn(x, t, **kwargs)
                        self.bar.update(1)
                        return ret

                    return wrapped

                model_fn = model_wrapper(
                    my_wrapper(self.denoise_fn), noise_schedule, model_type="noise", model_kwargs={"cond": cond}
                )
                uni_pc = UniPC(model_fn, noise_schedule, variant="bh2")
                steps = t_max // hparams["diff_speedup"]
                self.bar = tqdm(desc="sample time step", total=steps, disable=not hparams["infer"], leave=False)
                x = uni_pc.sample(x, steps=steps, order=2, skip_type="time_uniform", method="multistep")
                self.bar.close()
            elif algorithm == "pndm":
                self.noise_list = deque(maxlen=4)
                iteration_interval = speedup
                for i in tqdm(
                    reversed(range(0, t_max, iteration_interval)),
                    desc="sample time step",
                    total=t_max // iteration_interval,
                    disable=not hparams["infer"],
                    leave=False,
                ):
                    x = self.p_sample_plms(
                        x, paddle.full(shape=(b,), fill_value=i, dtype="int64"), iteration_interval, cond=cond
                    )
            elif algorithm == "ddim":
                iteration_interval = speedup
                for i in tqdm(
                    reversed(range(0, t_max, iteration_interval)),
                    desc="sample time step",
                    total=t_max // iteration_interval,
                    disable=not hparams["infer"],
                    leave=False,
                ):
                    x = self.p_sample_ddim(
                        x, paddle.full(shape=(b,), fill_value=i, dtype="int64"), iteration_interval, cond=cond
                    )
            else:
                raise ValueError(f"Unsupported acceleration algorithm for DDPM: {algorithm}.")
        else:
            for i in tqdm(
                reversed(range(0, t_max)),
                desc="sample time step",
                total=t_max,
                disable=not hparams["infer"],
                leave=False,
            ):
                x = self.p_sample(x, paddle.full(shape=(b,), fill_value=i, dtype="int64"), cond)
        x = x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 2, 3)).squeeze(axis=1)
        return x

    def forward(self, condition, gt_spec=None, src_spec=None, infer=True):
        """
        conditioning diffusion, use fastspeech2 encoder output as the condition
        """
        cond = condition.transpose(perm=paddle_aux.transpose_aux_func(condition.ndim, 1, 2))
        b, device = tuple(condition.shape)[0], condition.place
        if not infer:
            spec = self.norm_spec(gt_spec).transpose(
                perm=paddle_aux.transpose_aux_func(self.norm_spec(gt_spec).ndim, -2, -1)
            )
            if self.num_feats == 1:
                spec = spec[:, None, :, :]
            t = paddle.randint(low=0, high=self.k_step, shape=(b,)).astype(dtype="int64")
            x_recon, noise = self.p_losses(spec, t, cond=cond)
            return x_recon, noise
        else:
            if src_spec is not None:
                spec = self.norm_spec(src_spec).transpose(
                    perm=paddle_aux.transpose_aux_func(self.norm_spec(src_spec).ndim, -2, -1)
                )
                if self.num_feats == 1:
                    spec = spec[:, None, :, :]
            else:
                spec = None
            x = self.inference(cond, b=b, x_start=spec, device=device)
            return self.denorm_spec(x)

    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min


class RepetitiveDiffusion(GaussianDiffusion):
    def __init__(
        self,
        vmin: (float | int | list),
        vmax: (float | int | list),
        repeat_bins: int,
        timesteps=1000,
        k_step=1000,
        backbone_type=None,
        backbone_args=None,
        betas=None,
    ):
        assert isinstance(vmin, (float, int)) and isinstance(vmin, (float, int)) or len(vmin) == len(vmax)
        num_feats = 1 if isinstance(vmin, (float, int)) else len(vmin)
        spec_min = [vmin] if num_feats == 1 else [[v] for v in vmin]
        spec_max = [vmax] if num_feats == 1 else [[v] for v in vmax]
        self.repeat_bins = repeat_bins
        super().__init__(
            out_dims=repeat_bins,
            num_feats=num_feats,
            timesteps=timesteps,
            k_step=k_step,
            backbone_type=backbone_type,
            backbone_args=backbone_args,
            betas=betas,
            spec_min=spec_min,
            spec_max=spec_max,
        )

    def norm_spec(self, x):
        """

        :param x: [B, T] or [B, F, T]
        :return [B, T, R] or [B, F, T, R]
        """
        if self.num_feats == 1:
            repeats = [1, 1, self.repeat_bins]
        else:
            repeats = [1, 1, 1, self.repeat_bins]
        return super().norm_spec(x.unsqueeze(axis=-1).tile(repeat_times=repeats))

    def denorm_spec(self, x):
        """

        :param x: [B, T, R] or [B, F, T, R]
        :return [B, T] or [B, F, T]
        """
        return super().denorm_spec(x).mean(axis=-1)


class PitchDiffusion(RepetitiveDiffusion):
    def __init__(
        self,
        vmin: float,
        vmax: float,
        cmin: float,
        cmax: float,
        repeat_bins,
        timesteps=1000,
        k_step=1000,
        backbone_type=None,
        backbone_args=None,
        betas=None,
    ):
        self.vmin = vmin
        self.vmax = vmax
        self.cmin = cmin
        self.cmax = cmax
        super().__init__(
            vmin=vmin,
            vmax=vmax,
            repeat_bins=repeat_bins,
            timesteps=timesteps,
            k_step=k_step,
            backbone_type=backbone_type,
            backbone_args=backbone_args,
            betas=betas,
        )

    def norm_spec(self, x):
        return super().norm_spec(x.clip(min=self.cmin, max=self.cmax))

    def denorm_spec(self, x):
        return super().denorm_spec(x).clip(min=self.cmin, max=self.cmax)


class MultiVarianceDiffusion(RepetitiveDiffusion):
    def __init__(
        self,
        ranges: List[Tuple[float, float]],
        clamps: List[Tuple[float | None, float | None] | None],
        repeat_bins,
        timesteps=1000,
        k_step=1000,
        backbone_type=None,
        backbone_args=None,
        betas=None,
    ):
        assert len(ranges) == len(clamps)
        self.clamps = clamps
        vmin = [r[0] for r in ranges]
        vmax = [r[1] for r in ranges]
        if len(vmin) == 1:
            vmin = vmin[0]
        if len(vmax) == 1:
            vmax = vmax[0]
        super().__init__(
            vmin=vmin,
            vmax=vmax,
            repeat_bins=repeat_bins,
            timesteps=timesteps,
            k_step=k_step,
            backbone_type=backbone_type,
            backbone_args=backbone_args,
            betas=betas,
        )

    def clamp_spec(self, xs: (list | tuple)):
        clamped = []
        for x, c in zip(xs, self.clamps):
            if c is None:
                clamped.append(x)
                continue
            clamped.append(x.clip(min=c[0], max=c[1]))
        return clamped

    def norm_spec(self, xs: (list | tuple)):
        """

        :param xs: sequence of [B, T]
        :return: [B, F, T] => super().norm_spec(xs) => [B, F, T, R]
        """
        assert len(xs) == self.num_feats
        clamped = self.clamp_spec(xs)
        xs = paddle.stack(x=clamped, axis=1)
        if self.num_feats == 1:
            xs = xs.squeeze(axis=1)
        return super().norm_spec(xs)

    def denorm_spec(self, xs):
        """

        :param xs: [B, T, R] or [B, F, T, R] => super().denorm_spec(xs) => [B, T] or [B, F, T]
        :return: sequence of [B, T]
        """
        xs = super().denorm_spec(xs)
        if self.num_feats == 1:
            xs = [xs]
        else:
            xs = xs.unbind(axis=1)
        assert len(xs) == self.num_feats
        return self.clamp_spec(xs)
