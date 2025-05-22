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
import paddle_aux  # noqa: F401


class BaseNoiseCond:
    def __init__(self, *args, shift=1, clamp_range=None, **kwargs):
        clamp_range = [-1000000000.0, 1000000000.0] if clamp_range is None else clamp_range
        self.shift = shift
        self.clamp_range = clamp_range
        self.setup(*args, **kwargs)

    def setup(self, *args, **kwargs):
        pass

    def cond(self, logSNR):
        raise NotImplementedError("this method needs to be overridden")

    def __call__(self, logSNR):
        if self.shift != 1:
            logSNR = logSNR.clone() + 2 * np.log(self.shift)
        return paddle.clip(self.cond(logSNR), min=self.clamp_range[0], max=self.clamp_range[1])


class CosineTNoiseCond(BaseNoiseCond):
    def setup(self, s=0.008, clamp_range=[0, 1]):
        self.s = paddle.to_tensor(data=[s])
        self.clamp_range = clamp_range
        self.min_var = paddle.square(paddle.cos(x=self.s / (1 + self.s) * np.pi * 0.5))

    def cond(self, logSNR):
        var = paddle.nn.functional.sigmoid(logSNR)
        var = paddle.clip(var, min=self.clamp_range[0], max=self.clamp_range[1])
        s, min_var = self.s, self.min_var
        t = ((var * min_var) ** 0.5).acos() / (np.pi * 0.5) * (1 + s) - s
        return t


class EDMNoiseCond(BaseNoiseCond):
    def cond(self, logSNR):
        return -logSNR / 8


class SigmoidNoiseCond(BaseNoiseCond):
    def cond(self, logSNR):
        return (-logSNR).sigmoid()


class LogSNRNoiseCond(BaseNoiseCond):
    def cond(self, logSNR):
        return logSNR


class EDMSigmaNoiseCond(BaseNoiseCond):
    def setup(self, sigma_data=1):
        self.sigma_data = sigma_data

    def cond(self, logSNR):
        return paddle.exp(x=-logSNR / 2) * self.sigma_data


class RectifiedFlowsNoiseCond(BaseNoiseCond):
    def cond(self, logSNR):
        _a = logSNR.exp() - 1
        _a[_a == 0] = 0.001
        a = 1 + (2 - (2**2 + 4 * _a) ** 0.5) / (2 * _a)
        return a


class PiecewiseLinearNoiseCond(BaseNoiseCond):
    def setup(self):
        self.x = None
        self.y = None

    def piecewise_linear(self, y, xs, ys):
        indices = len(xs) - 2 - paddle.searchsorted(sorted_sequence=ys.flip(axis=(-1,))[:-2], values=y)
        x_min, x_max = xs[indices], xs[indices + 1]
        y_min, y_max = ys[indices], ys[indices + 1]
        x = x_min + (x_max - x_min) * (y - y_min) / (y_max - y_min)
        return x

    def cond(self, logSNR):
        var = logSNR.sigmoid()
        t = self.piecewise_linear(var, self.x.to(var.place), self.y.to(var.place))
        return t


class StableDiffusionNoiseCond(PiecewiseLinearNoiseCond):
    def setup(self, linear_range=[0.00085, 0.012], total_steps=1000):
        self.total_steps = total_steps
        linear_range_sqrt = [(r**0.5) for r in linear_range]
        self.x = paddle.linspace(start=0, stop=1, num=total_steps + 1)
        alphas = 1 - (linear_range_sqrt[0] * (1 - self.x) + linear_range_sqrt[1] * self.x) ** 2
        self.y = alphas.cumprod(dim=-1)

    def cond(self, logSNR):
        return super().cond(logSNR).clip(min=0, max=1)


class DiscreteNoiseCond(BaseNoiseCond):
    def setup(self, noise_cond, steps=1000, continuous_range=[0, 1]):
        self.noise_cond = noise_cond
        self.steps = steps
        self.continuous_range = continuous_range

    def cond(self, logSNR):
        cond = self.noise_cond(logSNR)
        cond = (cond - self.continuous_range[0]) / (self.continuous_range[1] - self.continuous_range[0])
        return cond.mul(self.steps).astype(dtype="int64")
