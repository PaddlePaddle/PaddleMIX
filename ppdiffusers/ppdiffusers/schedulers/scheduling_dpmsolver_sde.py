# Copyright 2023 Katherine Crowson, The HuggingFace Team and hlky. All rights reserved.
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
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import numpy as np
import paddle
import paddlesde

from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput


class BatchedBrownianTree:
    """A wrapper around paddlesde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get("w0", paddle.zeros_like(x))
        if seed is None:
            seed = paddle.randint(
                0,
                2**31 - 1,  # 2**63 - 1,paddle will overflow
                [
                    1,
                ],
            ).item()
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
    """A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will use one BrownianTree per batch item, each
            with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x):
        self.transform = transform
        t0, t1 = self.transform(paddle.to_tensor(sigma_min)), self.transform(paddle.to_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(paddle.to_tensor(sigma)), self.transform(paddle.to_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


# Copied from ppdiffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return paddle.to_tensor(betas, dtype=paddle.float32)


class DPMSolverSDEScheduler(SchedulerMixin, ConfigMixin):
    """
    Implements Stochastic Sampler (Algorithm 2) from Karras et al. (2022). Based on the original k-diffusion
    implementation by Katherine Crowson:
    https://github.com/crowsonkb/k-diffusion/blob/41b4cb6df0506694a7776af31349acf082bf6091/k_diffusion/sampling.py#L543

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model. beta_start (`float`): the
        starting `beta` value of inference. beta_end (`float`): the final `beta` value. beta_schedule (`str`):
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        prediction_type (`str`, default `epsilon`, optional):
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion
            process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4
            https://imagen.research.google/video/paper.pdf)
        use_karras_sigmas (`bool`, *optional*, defaults to `False`):
             This parameter controls whether to use Karras sigmas (Karras et al. (2022) scheme) for step sizes in the
             noise schedule during the sampling process. If True, the sigmas will be determined according to a sequence
             of noise levels {σi} as defined in Equation (5) of the paper https://arxiv.org/pdf/2206.00364.pdf.
        noise_sampler_seed (`int`, *optional*, defaults to `None`):
            The random seed to use for the noise sampler. If `None`, a random seed will be generated.
        timestep_spacing (`str`, default `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2. of [Common Diffusion Noise Schedules and Sample
            Steps are Flawed](https://arxiv.org/abs/2305.08891) for more information.
        steps_offset (`int`, default `0`):
            an offset added to the inference steps. You can use a combination of `offset=1` and
            `set_alpha_to_one=False`, to make the last step use step 0 for the previous alpha product, as done in
            stable diffusion.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 2

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,  # sensible defaults
        beta_end: float = 0.012,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        prediction_type: str = "epsilon",
        use_karras_sigmas: Optional[bool] = False,
        noise_sampler_seed: Optional[int] = None,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
    ):
        if trained_betas is not None:
            self.betas = paddle.to_tensor(trained_betas, dtype=paddle.float32)
        elif beta_schedule == "linear":
            self.betas = paddle.linspace(beta_start, beta_end, num_train_timesteps, dtype=paddle.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                paddle.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=paddle.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = paddle.cumprod(self.alphas, 0)

        #  set all values
        self.set_timesteps(num_train_timesteps, None, num_train_timesteps)
        self.use_karras_sigmas = use_karras_sigmas
        self.noise_sampler = None
        self.noise_sampler_seed = noise_sampler_seed

    # Copied from ppdiffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler.index_for_timestep
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        if len(self._index_counter) == 0:
            pos = 1 if len(indices) > 1 else 0
        else:
            timestep_int = timestep.item() if paddle.is_tensor(timestep) else timestep
            pos = self._index_counter[timestep_int]

        return indices[pos].item()

    @property
    def init_noise_sigma(self):
        # standard deviation of the initial noise distribution
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            return self.sigmas.max()

        return (self.sigmas.max() ** 2 + 1) ** 0.5

    def scale_model_input(
        self,
        sample: paddle.Tensor,
        timestep: Union[float, paddle.Tensor],
    ) -> paddle.Tensor:
        """
        Args:
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.
            sample (`paddle.Tensor`): input sample timestep (`int`, optional): current timestep
        Returns:
            `paddle.Tensor`: scaled input sample
        """
        step_index = self.index_for_timestep(timestep)

        sigma = self.sigmas[step_index]
        sigma_input = sigma if self.state_in_first_order else self.mid_point_sigma
        sample = sample / ((sigma_input**2 + 1) ** 0.5)
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str = None,
        num_train_timesteps: Optional[int] = None,
    ):
        """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.

        """
        self.num_inference_steps = num_inference_steps

        num_train_timesteps = num_train_timesteps or self.config.num_train_timesteps

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.config.timestep_spacing == "linspace":
            timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps, dtype=float)[::-1].copy()
        elif self.config.timestep_spacing == "leading":
            step_ratio = num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(float)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(num_train_timesteps, 0, -step_ratio)).round().copy().astype(float)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )

        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)

        if self.use_karras_sigmas:
            sigmas = self._convert_to_karras(in_sigmas=sigmas)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])

        second_order_timesteps = self._second_order_timesteps(sigmas, log_sigmas)

        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        sigmas = paddle.to_tensor(sigmas)
        self.sigmas = paddle.concat([sigmas[:1], sigmas[1:-1].repeat_interleave(2), sigmas[-1:]])

        timesteps = paddle.to_tensor(timesteps)
        second_order_timesteps = paddle.to_tensor(second_order_timesteps)
        timesteps = paddle.concat([timesteps[:1], timesteps[1:].repeat_interleave(2)])
        timesteps[1::2] = second_order_timesteps
        self.timesteps = timesteps.cast(paddle.float32)

        # empty first order variables
        self.sample = None
        self.mid_point_sigma = None

        # for exp beta schedules, such as the one for `pipeline_shap_e.py`
        # we need an index counter
        self._index_counter = defaultdict(int)

    def _second_order_timesteps(self, sigmas, log_sigmas):
        def sigma_fn(_t):
            return np.exp(-_t)

        def t_fn(_sigma):
            return -np.log(_sigma)

        midpoint_ratio = 0.5
        t = t_fn(sigmas)
        delta_time = np.diff(t)
        t_proposed = t[:-1] + delta_time * midpoint_ratio
        sig_proposed = sigma_fn(t_proposed)
        timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sig_proposed])
        return timesteps

    # copied from ppdiffusers.schedulers.scheduling_euler_discrete._sigma_to_t
    def _sigma_to_t(self, sigma, log_sigmas):
        # get log sigma
        log_sigma = np.log(sigma)

        # get distribution
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # get sigmas range
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # interpolate sigmas
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        return t

    # copied from ppdiffusers.schedulers.scheduling_euler_discrete._convert_to_karras
    def _convert_to_karras(self, in_sigmas: paddle.Tensor) -> paddle.Tensor:
        """Constructs the noise schedule of Karras et al. (2022)."""

        sigma_min: float = in_sigmas[-1].item()
        sigma_max: float = in_sigmas[0].item()

        rho = 7.0  # 7.0 is the value used in the paper
        ramp = np.linspace(0, 1, self.num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    @property
    def state_in_first_order(self):
        return self.sample is None

    def step(
        self,
        model_output: Union[paddle.Tensor, np.ndarray],
        timestep: Union[float, paddle.Tensor],
        sample: Union[paddle.Tensor, np.ndarray],
        return_dict: bool = True,
        s_noise: float = 1.0,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Args:
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).
        model_output (Union[paddle.Tensor, np.ndarray]): Direct output from learned diffusion model.
        timestep (Union[float, paddle.Tensor]): Current discrete timestep in the diffusion chain.
        sample (Union[paddle.Tensor, np.ndarray]): Current instance of sample being created by diffusion process.
        return_dict (bool, optional): Option for returning tuple rather than SchedulerOutput class. Defaults to True.
        s_noise (float, optional): Scaling factor for the noise added to the sample. Defaults to 1.0.
        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.SchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        step_index = self.index_for_timestep(timestep)

        # advance index counter by 1
        timestep_int = timestep.item() if paddle.is_tensor(timestep) else timestep
        self._index_counter[timestep_int] += 1

        # Create a noise sampler if it hasn't been created yet
        if self.noise_sampler is None:
            min_sigma, max_sigma = self.sigmas[self.sigmas > 0].min(), self.sigmas.max()
            self.noise_sampler = BrownianTreeNoiseSampler(sample, min_sigma, max_sigma, self.noise_sampler_seed)

        # Define functions to compute sigma and t from each other
        def sigma_fn(_t: paddle.Tensor) -> paddle.Tensor:
            return _t.neg().exp()

        def t_fn(_sigma: paddle.Tensor) -> paddle.Tensor:
            return _sigma.log().neg()

        if self.state_in_first_order:
            sigma = self.sigmas[step_index]
            sigma_next = self.sigmas[step_index + 1]
        else:
            # 2nd order
            sigma = self.sigmas[step_index - 1]
            sigma_next = self.sigmas[step_index]

        # Set the midpoint and step size for the current step
        midpoint_ratio = 0.5
        t, t_next = t_fn(sigma), t_fn(sigma_next)
        delta_time = t_next - t
        t_proposed = t + delta_time * midpoint_ratio

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            sigma_input = sigma if self.state_in_first_order else sigma_fn(t_proposed)
            pred_original_sample = sample - sigma_input * model_output
        elif self.config.prediction_type == "v_prediction":
            sigma_input = sigma if self.state_in_first_order else sigma_fn(t_proposed)
            pred_original_sample = model_output * (-sigma_input / (sigma_input**2 + 1) ** 0.5) + (
                sample / (sigma_input**2 + 1)
            )
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        if sigma_next == 0:
            derivative = (sample - pred_original_sample) / sigma
            dt = sigma_next - sigma
            prev_sample = sample + derivative * dt
        else:
            if self.state_in_first_order:
                t_next = t_proposed
            else:
                sample = self.sample

            sigma_from = sigma_fn(t)
            sigma_to = sigma_fn(t_next)
            sigma_up = min(sigma_to, (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
            sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
            ancestral_t = t_fn(sigma_down)
            prev_sample = (sigma_fn(ancestral_t) / sigma_fn(t)) * sample - (
                t - ancestral_t
            ).expm1() * pred_original_sample
            prev_sample = prev_sample + self.noise_sampler(sigma_fn(t), sigma_fn(t_next)) * s_noise * sigma_up

            if self.state_in_first_order:
                # store for 2nd order step
                self.sample = sample
                self.mid_point_sigma = sigma_fn(t_next)
            else:
                # free for "first order mode"
                self.sample = None
                self.mid_point_sigma = None

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    # Copied from ppdiffusers.schedulers.scheduling_heun_discrete.HeunDiscreteScheduler.add_noise
    def add_noise(
        self,
        original_samples: paddle.Tensor,
        noise: paddle.Tensor,
        timesteps: paddle.Tensor,
    ) -> paddle.Tensor:
        # Fix 0D tensor
        if paddle.is_tensor(timesteps) and timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.cast(original_samples.dtype)

        schedule_timesteps = self.timesteps

        step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
