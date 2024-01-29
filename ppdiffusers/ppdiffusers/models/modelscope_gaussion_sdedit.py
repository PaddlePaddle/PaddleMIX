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

import math
import random

import paddle
from tqdm.auto import trange


def _logsnr_cosine(n, logsnr_min=-15, logsnr_max=15):
    t_min = math.atan(math.exp(-0.5 * logsnr_min))
    t_max = math.atan(math.exp(-0.5 * logsnr_max))
    t = paddle.linspace(1, 0, n)
    logsnrs = -2 * paddle.log(paddle.tan(t_min + t * (t_max - t_min)))
    return logsnrs


def _logsnr_cosine_shifted(n, logsnr_min=-15, logsnr_max=15, scale=2):
    logsnrs = _logsnr_cosine(n, logsnr_min, logsnr_max)
    logsnrs += 2 * math.log(1 / scale)
    return logsnrs


def logsnrs_to_sigmas(logsnrs):
    return paddle.sqrt(paddle.nn.functional.sigmoid(-logsnrs))


def _logsnr_cosine_interp(n, logsnr_min=-15, logsnr_max=15, scale_min=2, scale_max=4):
    t = paddle.linspace(1, 0, n)
    logsnrs_min = _logsnr_cosine_shifted(n, logsnr_min, logsnr_max, scale_min)
    logsnrs_max = _logsnr_cosine_shifted(n, logsnr_min, logsnr_max, scale_max)
    logsnrs = t * logsnrs_min + (1 - t) * logsnrs_max
    return logsnrs


def logsnr_cosine_interp_schedule(n, logsnr_min=-15, logsnr_max=15, scale_min=2, scale_max=4):
    return logsnrs_to_sigmas(_logsnr_cosine_interp(n, logsnr_min, logsnr_max, scale_min, scale_max))


def noise_schedule(schedule="logsnr_cosine_interp", n=1000, zero_terminal_snr=False, **kwargs):
    # compute sigmas
    sigmas = {"logsnr_cosine_interp": logsnr_cosine_interp_schedule}[schedule](n, **kwargs)

    # post-processing
    if zero_terminal_snr and sigmas.max() != 1.0:
        scale = (1.0 - sigmas.min()) / (sigmas.max() - sigmas.min())
        sigmas = sigmas.min() + scale * (sigmas - sigmas.min())
    return sigmas


def _i(tensor, t, x):
    r"""Index tensor using t and format the output according to x."""
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    if tensor.place != x.place:
        tensor = paddle.to_tensor(tensor, place=x.place)
    return tensor[t].reshape(shape).astype(x.dtype)


def get_scalings(sigma):
    c_out = -sigma
    c_in = 1 / (sigma**2 + 1.0**2) ** 0.5
    return c_out, c_in


def karras_schedule(n, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    ramp = paddle.linspace(1, 0, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    sigmas = paddle.sqrt(sigmas**2 / (1 + sigmas**2))
    return sigmas


@paddle.no_grad()
def sample_heun(noise, model, sigmas, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, show_progress=True):
    """
    Implements Algorithm 2 (Heun steps) from Karras et al. (2022).
    """
    x = noise * sigmas[0]
    for i in trange(len(sigmas) - 1, disable=not show_progress):
        gamma = 0.0
        if s_tmin <= sigmas[i] <= s_tmax and sigmas[i] < float("inf"):
            gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
        eps = paddle.randn(shape=x.shape, dtype=x.dtype) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        if sigmas[i] == float("inf"):
            # Euler method
            denoised = model(noise, sigma_hat)
            x = denoised + sigmas[i + 1] * (gamma + 1) * noise
        else:
            _, c_in = get_scalings(sigma_hat)
            denoised = model(x * c_in, sigma_hat)
            d = (x - denoised) / sigma_hat
            dt = sigmas[i + 1] - sigma_hat
            if sigmas[i + 1] == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                _, c_in = get_scalings(sigmas[i + 1])
                denoised_2 = model(x_2 * c_in, sigmas[i + 1])
                d_2 = (x_2 - denoised_2) / sigmas[i + 1]
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
    return x


class BatchedBrownianTree:
    """
    A wrapper around torchsde.BrownianTree that enables batches of entropy.
    """

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        import paddlesde

        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get("w0", paddle.zeros_like(x))
        if seed is None:
            seed = paddle.randint(0, 2**31 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        self.trees = [paddlesde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        w = paddle.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)
        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """
    A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x):
        self.transform = transform
        t0 = self.transform(paddle.to_tensor(sigma_min))
        t1 = self.transform(paddle.to_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed)

    def __call__(self, sigma, sigma_next):
        t0 = self.transform(paddle.to_tensor(sigma))
        t1 = self.transform(paddle.to_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


@paddle.no_grad()
def sample_dpmpp_2m_sde(noise, model, sigmas, eta=1.0, s_noise=1.0, solver_type="midpoint", show_progress=True):
    """
    DPM-Solver++ (2M) SDE.
    """
    assert solver_type in {"heun", "midpoint"}

    x = noise * sigmas[0]
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas[sigmas < float("inf")].max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max)
    old_denoised = None
    h_last = None

    for i in trange(len(sigmas) - 1, disable=not show_progress):
        if sigmas[i] == float("inf"):
            # Euler method
            denoised = model(noise, sigmas[i])
            x = denoised + sigmas[i + 1] * noise
        else:
            _, c_in = get_scalings(sigmas[i])
            denoised = model(x * c_in, sigmas[i])
            if sigmas[i + 1] == 0:
                # Denoising step
                x = denoised
            else:
                # DPM-Solver++(2M) SDE
                t, s = -sigmas[i].log(), -sigmas[i + 1].log()
                h = s - t
                eta_h = eta * h

                x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised
                if old_denoised is not None:
                    r = h_last / h
                    if solver_type == "heun":
                        x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                    elif solver_type == "midpoint":
                        x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

                x = (
                    x
                    + noise_sampler(sigmas[i], sigmas[i + 1])
                    * sigmas[i + 1]
                    * (-2 * eta_h).expm1().neg().sqrt()
                    * s_noise
                )

            old_denoised = denoised
            h_last = h
    return x


class GaussianDiffusion_SDEdit(object):
    def __init__(self, sigmas, prediction_type="eps"):
        assert prediction_type in {"x0", "eps", "v"}
        self.sigmas = sigmas
        self.alphas = paddle.sqrt(1 - sigmas**2)
        self.num_timesteps = len(sigmas)
        self.prediction_type = prediction_type

    def diffuse(self, x0, t, noise=None):
        noise = paddle.randn(shape=x0.shape, dtype=x0.dtype) if noise is None else noise
        xt = _i(self.alphas, t, x0) * x0 + _i(self.sigmas, t, x0) * noise
        return xt

    def denoise(
        self, xt, t, s, model, model_kwargs={}, guide_scale=None, guide_rescale=None, clamp=None, percentile=None
    ):
        s = t - 1 if s is None else s

        # hyperparams
        sigmas = _i(self.sigmas, t, xt)
        alphas = _i(self.alphas, t, xt)
        alphas_s = _i(self.alphas, s.clip(0), xt)
        alphas_s[s < 0] = 1.0
        sigmas_s = paddle.sqrt(1 - alphas_s**2)

        # precompute variables
        betas = 1 - (alphas / alphas_s) ** 2
        coef1 = betas * alphas_s / sigmas**2
        coef2 = (alphas * sigmas_s**2) / (alphas_s * sigmas**2)
        var = betas * (sigmas_s / sigmas) ** 2
        log_var = paddle.log(var).clip_(-20, 20)

        # prediction
        if guide_scale is None:
            assert isinstance(model_kwargs, dict)
            out = model(xt, t=t, **model_kwargs).sample
        else:
            # classifier-free guidance
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
            y_out = model(xt, t=t, **model_kwargs[0]).sample
            if guide_scale == 1.0:
                out = y_out
            else:
                u_out = model(xt, t=t, **model_kwargs[1]).sample
                out = u_out + guide_scale * (y_out - u_out)

                if guide_rescale is not None:
                    assert 0 <= guide_rescale <= 1
                    ratio = (
                        paddle.std(y_out.flatten(1), axis=1) / (paddle.std(out.flatten(1), axis=1) + 1e-12)  # noqa
                    ).reshape(list((-1,) + (1,) * (y_out.ndim - 1)))
                    out *= guide_rescale * ratio + (1 - guide_rescale) * 1.0

        # compute x0
        if self.prediction_type == "x0":
            x0 = out
        elif self.prediction_type == "eps":
            x0 = (xt - sigmas * out) / alphas
        elif self.prediction_type == "v":
            x0 = alphas * xt - sigmas * out
        else:
            raise NotImplementedError(f"prediction_type {self.prediction_type} not implemented")

        # restrict the range of x0
        if percentile is not None:
            assert 0 < percentile <= 1
            s = paddle.quantile(x0.flatten(1).abs(), percentile, axis=1).clip_(1.0).reshape([-1, 1, 1, 1])
            x0 = paddle.min(s, paddle.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clip_(-clamp, clamp)

        # recompute eps using the restricted x0
        eps = (xt - alphas * x0) / sigmas

        # compute mu (mean of posterior distribution) using the restricted x0
        mu = coef1 * x0 + coef2 * xt
        return mu, var, log_var, x0, eps

    @paddle.no_grad()
    def sample(
        self,
        noise,
        model,
        model_kwargs={},
        condition_fn=None,
        guide_scale=None,
        guide_rescale=None,
        clamp=None,
        percentile=None,
        solver="euler_a",
        steps=20,
        t_max=None,
        t_min=None,
        discretization=None,
        discard_penultimate_step=None,
        return_intermediate=None,
        show_progress=False,
        seed=-1,
        **kwargs
    ):
        # sanity check
        assert isinstance(steps, (int, "paddle.int64"))
        assert t_max is None or (0 < t_max <= self.num_timesteps - 1)
        assert t_min is None or (0 <= t_min < self.num_timesteps - 1)
        assert discretization in (None, "leading", "linspace", "trailing")
        assert discard_penultimate_step in (None, True, False)
        assert return_intermediate in (None, "x0", "xt")

        # function of diffusion solver
        solver_fn = {"heun": sample_heun, "dpmpp_2m_sde": sample_dpmpp_2m_sde}[solver]

        # options
        schedule = "karras" if "karras" in solver else None
        discretization = discretization or "linspace"
        seed = seed if seed >= 0 else random.randint(0, 2**31)

        if isinstance(steps, paddle.Tensor):
            discard_penultimate_step = False
        if discard_penultimate_step is None:
            discard_penultimate_step = (
                True
                if solver
                in (
                    "dpm2",
                    "dpm2_ancestral",
                    "dpmpp_2m_sde",
                    "dpm2_karras",
                    "dpm2_ancestral_karras",
                    "dpmpp_2m_sde_karras",
                )
                else False
            )

        # function for denoising xt to get x0
        intermediates = []

        def model_fn(xt, sigma):
            # denoising
            t = self._sigma_to_t(sigma).tile(len(xt)).round().astype("int64")
            x0 = self.denoise(xt, t, None, model, model_kwargs, guide_scale, guide_rescale, clamp, percentile)[-2]

            # collect intermediate outputs
            if return_intermediate == "xt":
                intermediates.append(xt)
            elif return_intermediate == "x0":
                intermediates.append(x0)
            return x0

        # get timesteps
        if isinstance(steps, int):
            steps += 1 if discard_penultimate_step else 0
            t_max = self.num_timesteps - 1 if t_max is None else t_max
            t_min = 0 if t_min is None else t_min

            # discretize timesteps
            if discretization == "leading":
                steps = paddle.arange(t_min, t_max + 1, (t_max - t_min + 1) / steps).flip(0)
            elif discretization == "linspace":
                steps = paddle.linspace(t_max, t_min, steps)
            elif discretization == "trailing":
                steps = paddle.arange(t_max, t_min - 1, -((t_max - t_min + 1) / steps))
            else:
                raise NotImplementedError(f"{discretization} discretization not implemented")
            steps = steps.clip_(t_min, t_max)
        steps = paddle.to_tensor(steps, dtype=paddle.float32, place=noise.place)

        # get sigmas
        sigmas = self._t_to_sigma(steps)
        sigmas = paddle.concat([sigmas, paddle.zeros([1]).astype(sigmas.dtype)])
        if schedule == "karras":
            if sigmas[0] == float("inf"):
                sigmas = karras_schedule(
                    n=len(steps) - 1,
                    sigma_min=sigmas[sigmas > 0].min().item(),
                    sigma_max=sigmas[sigmas < float("inf")].max().item(),
                    rho=7.0,
                ).to(sigmas)
                sigmas = paddle.concat(
                    [sigmas.to_tensor([float("inf")]), sigmas, paddle.zeros([1]).astype(sigmas.dtype)]
                )
            else:
                sigmas = karras_schedule(
                    n=len(steps), sigma_min=sigmas[sigmas > 0].min().item(), sigma_max=sigmas.max().item(), rho=7.0
                ).to(sigmas)
                sigmas = paddle.concat([sigmas, paddle.zeros([1]).astype(sigmas.dtype)])
        if discard_penultimate_step:
            sigmas = paddle.concat([sigmas[:-2], sigmas[-1:]])

        # sampling
        x0 = solver_fn(noise, model_fn, sigmas, show_progress=show_progress, **kwargs)
        return (x0, intermediates) if return_intermediate is not None else x0

    def _sigma_to_t(self, sigma):
        if sigma == float("inf"):
            t = paddle.full_like(sigma, len(self.sigmas) - 1)
        else:
            log_sigmas = paddle.sqrt(self.sigmas**2 / (1 - self.sigmas**2)).log().astype(sigma.dtype)  # noqa
            log_sigma = sigma.log()
            dists = log_sigma - log_sigmas[:, None]

            low_idx = dists.greater_equal(paddle.to_tensor(0, dtype=dists.dtype)).astype(dists.dtype)
            low_idx = paddle.cumsum(low_idx, axis=0).argmax(axis=0).clip_(max=log_sigmas.shape[0] - 2)
            high_idx = low_idx + 1
            low, high = log_sigmas[low_idx], log_sigmas[high_idx]
            w = (low - log_sigma) / (low - high)
            w = w.clip_(0, 1)
            t = (1 - w) * low_idx + w * high_idx
            t = t.reshape(sigma.shape)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        return t

    def _t_to_sigma(self, t):
        t = t.astype("float32")
        low_idx, high_idx, w = t.floor().astype("int64"), t.ceil().astype("int64"), t.frac()
        log_sigmas = paddle.sqrt(self.sigmas**2 / (1 - self.sigmas**2)).log().astype(t.dtype)  # noqa
        log_sigma = (1 - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx]
        log_sigma[paddle.isnan(log_sigma) | paddle.isinf(log_sigma)] = float("inf")
        return log_sigma.exp()
