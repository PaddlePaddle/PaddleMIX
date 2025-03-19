# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""
Feynman-Kac Diffusion (FKD) steering mechanism paddle implementation.
"""

from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np
import paddle


class PotentialType(Enum):
    DIFF = "diff"
    MAX = "max"
    ADD = "add"


def paddle_max(*args, **kwargs):
    if "input" in kwargs:
        kwargs["x"] = kwargs.pop("input")

    out_v = None
    if "out" in kwargs:
        out_v = kwargs.pop("out")

    if "other" in kwargs:
        kwargs["y"] = kwargs.pop("other")
        ret = paddle.maximum(*args, **kwargs)
    elif len(args) == 2 and isinstance(args[1], paddle.Tensor):
        ret = paddle.maximum(*args, **kwargs)
    else:
        if "dim" in kwargs:
            kwargs["axis"] = kwargs.pop("dim")

        if "axis" in kwargs or len(args) >= 2:
            if out_v:
                ret = paddle.max(*args, **kwargs), paddle.argmax(*args, **kwargs)
                paddle.assign(ret[0], out_v[0])
                paddle.assign(ret[1], out_v[1])
                return out_v
            else:
                ret = paddle.max(*args, **kwargs), paddle.argmax(*args, **kwargs)
                return ret
        else:
            ret = paddle.max(*args, **kwargs)
            return ret

    if out_v:
        paddle.assign(ret, out_v)
        return out_v
    else:
        return ret


class FKD:
    """
    Implements the FKD steering mechanism. Should be initialized along the diffusion process. .resample() should be invoked at each diffusion timestep.
    See FKD fkd_pipeline_sdxl
    Args:
        potential_type: Type of potential function must be one of PotentialType.
        lmbda: Lambda hyperparameter controlling weight scaling.
        num_particles: Number of particles to maintain in the population.
        adaptive_resampling: Whether to perform adaptive resampling.
        resample_frequency: Frequency (in timesteps) to perform resampling.
        resampling_t_start: Timestep to start resampling.
        resampling_t_end: Timestep to stop resampling.
        time_steps: Total number of timesteps in the sampling process.
        reward_fn: Function to compute rewards from decoded latents.
        reward_min_value: Minimum value for rewards (default: 0.0). Important for the Max potential type.
        latent_to_decode_fn: Function to decode latents to images, relevant for latent diffusion models (default: identity function).
        device: Device on which computations will be performed (default: CUDA).
        **kwargs: Additional keyword arguments, unused.
    """

    def __init__(
        self,
        *,
        potential_type: PotentialType,
        lmbda: float,
        num_particles: int,
        adaptive_resampling: bool,
        resample_frequency: int,
        resampling_t_start: int,
        resampling_t_end: int,
        time_steps: int,
        reward_fn: Callable[[paddle.Tensor], paddle.Tensor],
        reward_min_value: float = 0.0,
        latent_to_decode_fn: Callable[[paddle.Tensor], paddle.Tensor] = lambda x: x,
        **kwargs,
    ) -> None:
        self.potential_type = PotentialType(potential_type)
        self.lmbda = lmbda
        self.num_particles = num_particles
        self.adaptive_resampling = adaptive_resampling
        self.resample_frequency = resample_frequency
        self.resampling_t_start = resampling_t_start
        self.resampling_t_end = resampling_t_end
        self.time_steps = time_steps
        self.reward_fn = reward_fn
        self.latent_to_decode_fn = latent_to_decode_fn
        self.population_rs = paddle.ones(shape=self.num_particles) * reward_min_value
        self.product_of_potentials = paddle.ones(shape=self.num_particles)

    def resample(
        self, *, sampling_idx: int, latents: paddle.Tensor, x0_preds: paddle.Tensor
    ) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
        """
        Perform resampling of particles if conditions are met.
        Should be invoked at each timestep in the reverse diffusion process.

        Args:
            sampling_idx: Current sampling index (timestep).
            latents: Current noisy latents.
            x0_preds: Predictions for x0 based on latents.

        Returns:
            A tuple containing resampled latents and optionally resampled images.
        """
        resampling_interval = np.arange(self.resampling_t_start, self.resampling_t_end + 1, self.resample_frequency)
        resampling_interval = np.append(resampling_interval, self.time_steps - 1)
        if sampling_idx not in resampling_interval:
            return latents, None

        population_images = self.latent_to_decode_fn(x0_preds)
        rs_candidates = self.reward_fn(population_images)

        if self.potential_type == PotentialType.MAX:
            w = paddle.exp(x=self.lmbda * paddle_max(rs_candidates, self.population_rs))
        elif self.potential_type == PotentialType.ADD:
            rs_candidates = rs_candidates + self.population_rs
            w = paddle.exp(x=self.lmbda * rs_candidates)
        elif self.potential_type == PotentialType.DIFF:
            diffs = rs_candidates - self.population_rs
            w = paddle.exp(x=self.lmbda * diffs)
        else:
            raise ValueError(f"potential_type {self.potential_type} not recognized")

        if sampling_idx == self.time_steps - 1:
            if self.potential_type == PotentialType.MAX or self.potential_type == PotentialType.ADD:
                w = paddle.exp(x=self.lmbda * rs_candidates) / self.product_of_potentials
        w = paddle.clip(x=w, min=0, max=10000000000.0)
        w[paddle.isnan(x=w)] = 0.0

        if self.adaptive_resampling or sampling_idx == self.time_steps - 1:
            normalized_w = w / w.sum()
            ess = 1.0 / normalized_w.pow(y=2).sum()
            if ess < 0.5 * self.num_particles:
                print(f"Resampling at timestep {sampling_idx} with ESS: {ess}")
                indices = paddle.multinomial(x=w, num_samples=self.num_particles, replacement=True)
                resampled_latents = latents[indices]
                self.population_rs = rs_candidates[indices]
                resampled_images = population_images[indices]
                self.product_of_potentials = self.product_of_potentials[indices] * w[indices]
            else:
                resampled_images = population_images
                resampled_latents = latents
                self.population_rs = rs_candidates

        else:
            indices = paddle.multinomial(x=w, num_samples=self.num_particles, replacement=True)

            resampled_latents = latents[indices]
            self.population_rs = rs_candidates[indices]

            resampled_images = population_images[indices]
            self.product_of_potentials = self.product_of_potentials[indices] * w[indices]

        return resampled_latents, resampled_images


if __name__ == "__main__":
    import random

    import matplotlib.pyplot as plt

    random.seed(0)
    num_particles = 8
    x0s = paddle.rand(shape=[num_particles, 1, 1])

    reward_function = lambda x: -0.5 * x.sum(axis=(1, 2))
    fkds = FKD(
        potential_type=PotentialType.DIFF,
        lmbda=10.0,
        num_particles=num_particles,
        adaptive_resampling=False,
        resample_frequency=1,
        resampling_t_start=-1,
        resampling_t_end=100,
        time_steps=100,
        reward_fn=reward_function,
    )
    sampling_idx = 0
    resampled_latents, resampled_images = fkds.resample(sampling_idx=sampling_idx, latents=x0s, x0_preds=x0s)

    plt.rc("text", usetex=True)
    fig, axs = plt.subplots(2, num_particles)
    axs[0, 0].set_title("Initial")
    axs[1, 0].set_title("Resampled")
    for i in range(num_particles):
        axs[0, i].imshow(x0s[i].detach().numpy(), cmap="gray", vmin=0, vmax=1)
        axs[1, i].imshow(resampled_images[i].detach().numpy(), cmap="gray", vmin=0, vmax=1)
        axs[1, i].axis("off")
        axs[0, i].axis("off")
    out_path = "resampled_examples.png"
    plt.savefig(out_path)
    print("Saved resampled examples to:", out_path)
