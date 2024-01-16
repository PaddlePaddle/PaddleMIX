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

import paddle


def beta_schedule(schedule, n_timestep, linear_start=0.0001, linear_end=0.02, cosine_s=0.008):
    if schedule == "linear":
        betas = (
            paddle.linspace(start=linear_start**0.5, stop=linear_end**0.5, num=n_timestep).astype(paddle.float64)
            ** 2
        )
        return betas
    else:
        raise ValueError(f"Unsupported schedule: {schedule}")


def _i(tensor, t, x):
    r"""Index tensor using t and format the output according to x."""
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    if tensor.place != x.place:
        tensor = paddle.to_tensor(tensor, place=x.place)
    return tensor[t].reshape(shape).astype(x.dtype)


class GaussianDiffusion(object):
    def __init__(
        self,
        betas,
        mean_type="eps",
        var_type="learned_range",
        loss_type="mse",
        epsilon=1e-12,
        rescale_timesteps=False,
        noise_strength=0.0,
    ):

        betas = paddle.to_tensor(betas, dtype=paddle.float64)
        self.betas = betas
        self.num_timesteps = len(betas)
        self.mean_type = mean_type
        self.var_type = var_type
        self.loss_type = loss_type
        self.epsilon = epsilon
        self.rescale_timesteps = rescale_timesteps
        self.noise_strength = noise_strength

        # alphas
        alphas = 1 - self.betas
        self.alphas_cumprod = paddle.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = paddle.concat([paddle.ones([1]).astype(alphas.dtype), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = paddle.concat([self.alphas_cumprod[1:], paddle.zeros([1]).astype(alphas.dtype)])

        # q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = paddle.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = paddle.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = paddle.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = paddle.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = paddle.sqrt(1.0 / self.alphas_cumprod - 1)

        # q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = paddle.log(self.posterior_variance.clip(1e-20))
        self.posterior_mean_coef1 = betas * paddle.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * paddle.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def p_mean_variance(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None):
        r"""Distribution of p(x_{t-1} | x_t)."""
        # predict distribution
        if guide_scale is None:
            out = model(xt, self._scale_timesteps(t), **model_kwargs).sample
        else:
            # classifier-free guidance
            # (model_kwargs[0]: conditional kwargs; model_kwargs[1]: non-conditional kwargs)
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
            y_out = model(xt, self._scale_timesteps(t), **model_kwargs[0]).sample
            u_out = model(xt, self._scale_timesteps(t), **model_kwargs[1]).sample

            dim = y_out.shape[1] if self.var_type.startswith("fixed") else y_out.shape[1] // 2
            out = paddle.concat(
                [u_out[:, :dim] + guide_scale * (y_out[:, :dim] - u_out[:, :dim]), y_out[:, dim:]], axis=1  # noqa
            )
        # compute variance
        if self.var_type == "learned":
            out, log_var = out.chunk(2, dim=1)
            var = paddle.exp(log_var)
        elif self.var_type == "learned_range":
            out, fraction = out.chunk(2, dim=1)
            min_log_var = _i(self.posterior_log_variance_clipped, t, xt)
            max_log_var = _i(paddle.log(self.betas), t, xt)
            fraction = (fraction + 1) / 2.0
            log_var = fraction * max_log_var + (1 - fraction) * min_log_var
            var = paddle.exp(log_var)
        elif self.var_type == "fixed_large":
            var = _i(paddle.concat([self.posterior_variance[1:2], self.betas[1:]]), t, xt)
            log_var = paddle.log(var)
        elif self.var_type == "fixed_small":
            var = _i(self.posterior_variance, t, xt)
            log_var = _i(self.posterior_log_variance_clipped, t, xt)

        # compute mean and x0
        if self.mean_type == "x_{t-1}":
            mu = out
            x0 = _i(1.0 / self.posterior_mean_coef1, t, xt) * mu - (
                _i(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, xt) * xt
            )
        elif self.mean_type == "x0":
            x0 = out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        elif self.mean_type == "eps":
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - (_i(self.sqrt_recipm1_alphas_cumprod, t, xt) * out)
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        elif self.mean_type == "v":
            x0 = _i(self.sqrt_alphas_cumprod, t, xt) * xt - (_i(self.sqrt_one_minus_alphas_cumprod, t, xt) * out)
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)

        # restrict the range of x0
        if percentile is not None:
            assert 0 < percentile <= 1
            s = paddle.quantile(x0.flatten(1).abs(), percentile, axis=1).clip_(1.0).reshape([-1, 1, 1, 1])
            x0 = paddle.min(s, paddle.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)
        return mu, var, log_var, x0

    def q_posterior_mean_variance(self, x0, xt, t):
        r"""Distribution of q(x_{t-1} | x_t, x_0)."""
        mu = _i(self.posterior_mean_coef1, t, xt) * x0 + _i(self.posterior_mean_coef2, t, xt) * xt
        var = _i(self.posterior_variance, t, xt)
        log_var = _i(self.posterior_log_variance_clipped, t, xt)
        return mu, var, log_var

    @paddle.no_grad()
    def ddim_sample(
        self,
        xt,
        t,
        model,
        model_kwargs={},
        clamp=None,
        percentile=None,
        condition_fn=None,
        guide_scale=None,
        ddim_timesteps=20,
        eta=0.0,
    ):
        r"""Sample from p(x_{t-1} | x_t) using DDIM.
        - condition_fn: for classifier-based guidance (guided-diffusion).
        - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        stride = self.num_timesteps // ddim_timesteps

        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

        if condition_fn is not None:
            # x0 -> eps
            alpha = _i(self.alphas_cumprod, t, xt)
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / (_i(self.sqrt_recipm1_alphas_cumprod, t, xt))
            eps = eps - (1 - alpha).sqrt() * condition_fn(xt, self._scale_timesteps(t), **model_kwargs)

            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - (_i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps)

        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / (_i(self.sqrt_recipm1_alphas_cumprod, t, xt))
        alphas = _i(self.alphas_cumprod, t, xt)
        alphas_prev = _i(self.alphas_cumprod, (t - stride).clip(0), xt)
        sigmas = eta * paddle.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))  # noqa

        # random sample
        noise = paddle.randn(shape=xt.shape, dtype=xt.dtype)
        direction = paddle.sqrt(1 - alphas_prev - sigmas**2) * eps
        mask = (t != 0).astype(paddle.float32).reshape([-1, *((1,) * (xt.ndim - 1))])
        xt_1 = paddle.sqrt(alphas_prev) * x0 + direction + mask * sigmas * noise
        return xt_1, x0

    @paddle.no_grad()
    def ddim_sample_loop(
        self,
        noise,
        model,
        model_kwargs={},
        clamp=None,
        percentile=None,
        condition_fn=None,
        guide_scale=None,
        ddim_timesteps=20,
        eta=0.0,
    ):
        # prepare input
        b = noise.shape[0]
        xt = noise
        # diffusion process (TODO: clamp is inaccurate! Consider replacing the stride by explicit prev/next steps)
        steps = (
            (1 + paddle.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps))
            .clip(0, self.num_timesteps - 1)
            .flip(0)
        )
        for step in steps:
            t = paddle.full((b,), step, dtype=paddle.int64)
            xt, _ = self.ddim_sample(
                xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale, ddim_timesteps, eta
            )
        return xt

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * 1000.0 / self.num_timesteps
        return t
