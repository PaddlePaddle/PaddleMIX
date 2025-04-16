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

import numpy as np
import paddle


def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x.shape) - 1)
    t = t.reshape([t.shape[0], *dims])
    return t


class ICPlan:
    """Linear Coupling Plan"""

    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        return t, 1

    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return 1 - t, -1

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        return 1 / t

    def compute_drift(self, x, t):
        """We always output sde according to score parametrization;"""
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t**2) - sigma_t * d_sigma_t
        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        """Compute the diffusion term of the SDE
        Args:
          x: [batch_dim, ...], data point
          t: [batch_dim,], time vector
          form: str, form of the diffusion term
          norm: float, norm of the diffusion term
        """
        t = expand_t_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * paddle.cos(np.pi * t) + 1) ** 2,
            "inccreasing-decreasing": norm * paddle.sin(np.pi * t) ** 2,
        }

        try:
            diffusion = choices[form]
        except KeyError:
            raise NotImplementedError(f"Diffusion form {form} not implemented")

        return diffusion

    def get_score_from_velocity(self, velocity, x, t):
        """Wrapper function: transform velocity prediction model to score
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score

    def get_noise_from_velocity(self, velocity, x, t):
        """Wrapper function: transform velocity prediction model to denoiser
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = reverse_alpha_ratio * d_sigma_t - sigma_t
        noise = (reverse_alpha_ratio * velocity - mean) / var
        return noise

    def get_velocity_from_score(self, score, x, t):
        """Wrapper function: transform score prediction model to velocity
        Args:
            score: [batch_dim, ...] shaped tensor; score model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        drift, var = self.compute_drift(x, t)
        velocity = var * score - drift
        return velocity

    def compute_mu_t(self, t, x0, x1):
        """Compute the mean of time-dependent density p_t"""
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t, x0, x1):
        """Sample xt from time-dependent density p_t; rng is required"""
        xt = self.compute_mu_t(t, x0, x1)
        return xt

    def compute_ut(self, t, x0, x1, xt):
        """Compute the vector field corresponding to p_t"""
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def plan(self, t, x0, x1):
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut


class VPCPlan(ICPlan):
    """class for VP path flow matching"""

    def __init__(self, sigma_min=0.1, sigma_max=20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_mean_coeff = (
            lambda t: -0.25 * ((1 - t) ** 2) * (self.sigma_max - self.sigma_min) - 0.5 * (1 - t) * self.sigma_min
        )
        self.d_log_mean_coeff = lambda t: 0.5 * (1 - t) * (self.sigma_max - self.sigma_min) + 0.5 * self.sigma_min

    def compute_alpha_t(self, t):
        """Compute coefficient of x1"""
        alpha_t = self.log_mean_coeff(t)
        alpha_t = paddle.exp(alpha_t)
        d_alpha_t = alpha_t * self.d_log_mean_coeff(t)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        """Compute coefficient of x0"""
        p_sigma_t = 2 * self.log_mean_coeff(t)
        sigma_t = paddle.sqrt(1 - paddle.exp(p_sigma_t))
        d_sigma_t = paddle.exp(p_sigma_t) * (2 * self.d_log_mean_coeff(t)) / (-2 * sigma_t)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return self.d_log_mean_coeff(t)

    def compute_drift(self, x, t):
        """Compute the drift term of the SDE"""
        t = expand_t_like_x(t, x)
        beta_t = self.sigma_min + (1 - t) * (self.sigma_max - self.sigma_min)
        return -0.5 * beta_t * x, beta_t / 2


class GVPCPlan(ICPlan):
    def __init__(self, sigma=0.0):
        super().__init__(sigma)

    def compute_alpha_t(self, t):
        """Compute coefficient of x1"""
        alpha_t = paddle.sin(t * np.pi / 2)
        d_alpha_t = np.pi / 2 * paddle.cos(t * np.pi / 2)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        """Compute coefficient of x0"""
        sigma_t = paddle.cos(t * np.pi / 2)
        d_sigma_t = -np.pi / 2 * paddle.sin(t * np.pi / 2)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return np.pi / (2 * paddle.tan(t * np.pi / 2))
