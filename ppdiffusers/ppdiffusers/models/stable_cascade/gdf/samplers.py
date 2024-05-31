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

import paddle

import ppdiffusers  # noqa: F401


class SimpleSampler:
    def __init__(self, gdf):
        self.gdf = gdf
        self.current_step = -1

    def __call__(self, *args, **kwargs):
        self.current_step += 1
        return self.step(*args, **kwargs)

    def init_x(self, shape):
        generator = paddle.Generator().manual_seed(1)
        return paddle.randn(shape=shape, generator=generator)

    def step(self, x, x0, epsilon, logSNR, logSNR_prev):
        raise NotImplementedError("You should override the 'apply' function.")


def expand_to_match(tensor, target_shape):
    # Expand tensor dimensions to match the target shape for broadcasting
    # Assuming tensor initially has shape [batch_size, 1] and target_shape is like [batch_size, channels, height, width]
    return tensor.unsqueeze(-1).unsqueeze(-1).expand(target_shape[0], target_shape[1], 1, 1)


class DDIMSampler(SimpleSampler):
    def step(self, x, x0, epsilon, logSNR, logSNR_prev, eta=0):
        a, b = self.gdf.input_scaler(logSNR)
        if len(a.shape) == 1:
            a, b = a.reshape([-1] + [1] * (len(x0.shape) - 1)), b.reshape([-1] + [1] * (len(x0.shape) - 1))

        a_prev, b_prev = self.gdf.input_scaler(logSNR_prev)

        if len(a_prev.shape) == 1:
            a_prev, b_prev = a_prev.reshape([-1] + [1] * (len(x0.shape) - 1)), b_prev.reshape(
                [-1] + [1] * (len(x0.shape) - 1)
            )
        sigma_tau = (
            eta * paddle.sqrt(b_prev**2 / b**2) * paddle.sqrt(1 - a**2 / a_prev**2)
            if eta > 0
            else paddle.zeros_like(x0)
        )
        x = (
            a_prev * x0
            + paddle.sqrt(b_prev**2 - sigma_tau**2) * epsilon
            + sigma_tau * paddle.randn(x0.shape, dtype=x0.dtype)
        )

        return x


class DDPMSampler(DDIMSampler):
    def step(self, x, x0, epsilon, logSNR, logSNR_prev, eta=1):
        return super().step(x, x0, epsilon, logSNR, logSNR_prev, eta)


class LCMSampler(SimpleSampler):
    def step(self, x, x0, epsilon, logSNR, logSNR_prev):
        a_prev, b_prev = self.gdf.input_scaler(logSNR_prev)
        if len(a_prev.shape) == 1:
            a_prev, b_prev = a_prev.unsqueeze(-1).expand_as(x0), b_prev.unsqueeze(-1).expand_as(x0)
        return x0 * a_prev + paddle.randn_like(epsilon) * b_prev
