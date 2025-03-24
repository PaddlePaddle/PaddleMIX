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
import paddle_aux  # noqa


class BaseSchedule:
    def __init__(self, *args, force_limits=True, discrete_steps=None, shift=1, **kwargs):
        self.setup(*args, **kwargs)
        self.limits = None
        self.discrete_steps = discrete_steps
        self.shift = shift
        if force_limits:
            self.reset_limits()

    def reset_limits(self, shift=1, disable=False):
        try:
            self.limits = None if disable else self(paddle.to_tensor(data=[1.0, 0.0]), shift=shift).tolist()
            return self.limits
        except Exception:
            print("WARNING: this schedule doesn't support t and will be unbounded")
            return None

    def setup(self, *args, **kwargs):
        raise NotImplementedError("this method needs to be overridden")

    def schedule(self, *args, **kwargs):
        raise NotImplementedError("this method needs to be overridden")

    def __call__(self, t, *args, shift=1, **kwargs):
        if isinstance(t, paddle.Tensor):
            batch_size = None
            if self.discrete_steps is not None:
                if t.dtype != "int64":
                    t = (t * (self.discrete_steps - 1)).round().astype(dtype="int64")
                t = t / (self.discrete_steps - 1)
            t = t.clip(min=0, max=1)
        else:
            batch_size = t
            t = None
        logSNR = self.schedule(t, batch_size, *args, **kwargs)
        if shift * self.shift != 1:
            logSNR += 2 * np.log(1 / (shift * self.shift))
        if self.limits is not None:
            logSNR = paddle.clip(logSNR, *self.limits)

        return logSNR


class CosineSchedule(BaseSchedule):
    def setup(self, s=0.008, clamp_range=[0.0001, 0.9999], norm_instead=False):
        self.s = paddle.to_tensor(data=[s])
        self.clamp_range = clamp_range
        self.norm_instead = norm_instead
        self.min_var = paddle.cos(x=self.s / (1 + self.s) * np.pi * 0.5) ** 2

    def schedule(self, t, batch_size):
        if t is None:
            t = (1 - paddle.rand(shape=[batch_size])).add(0.001).clip(min=0.001, max=1.0)
        s, min_var = self.s, self.min_var
        var = (paddle.cos(x=(s + t) / (1 + s) * np.pi * 0.5).clip(min=0, max=1) ** 2) / min_var

        if self.norm_instead:
            var = var * (self.clamp_range[1] - self.clamp_range[0]) + self.clamp_range[0]
        else:
            var = paddle.clip(var, min=self.clamp_range[0], max=self.clamp_range[1])
        logSNR = (var / (1 - var)).log()

        return logSNR


class CosineSchedule2(BaseSchedule):
    def setup(self, logsnr_range=[-15, 15]):
        self.t_min = np.arctan(np.exp(-0.5 * logsnr_range[1]))
        self.t_max = np.arctan(np.exp(-0.5 * logsnr_range[0]))

    def schedule(self, t, batch_size):
        if t is None:
            t = 1 - paddle.rand(shape=batch_size)
        return -2 * paddle.tan(self.t_min + t * (self.t_max - self.t_min)).log()


class SqrtSchedule(BaseSchedule):
    def setup(self, s=0.0001, clamp_range=[0.0001, 0.9999], norm_instead=False):
        self.s = s
        self.clamp_range = clamp_range
        self.norm_instead = norm_instead

    def schedule(self, t, batch_size):
        if t is None:
            t = 1 - paddle.rand(shape=batch_size)
        var = 1 - (t + self.s) ** 0.5
        if self.norm_instead:
            var = var * (self.clamp_range[1] - self.clamp_range[0]) + self.clamp_range[0]
        else:
            var = paddle.clip(var, min=self.clamp_range[0], max=self.clamp_range[1])
        logSNR = (var / (1 - var)).log()

        return logSNR


class RectifiedFlowsSchedule(BaseSchedule):
    def setup(self, logsnr_range=[-15, 15]):
        self.logsnr_range = logsnr_range

    def schedule(self, t, batch_size):
        if t is None:
            t = 1 - paddle.rand(shape=batch_size)
        logSNR = ((1 - t) ** 2 / t**2).log()
        logSNR = paddle.clip(logSNR, min=self.logsnr_range[0], max=self.logsnr_range[1])
        return logSNR


class EDMSampleSchedule(BaseSchedule):
    def setup(self, sigma_range=[0.002, 80], p=7):
        self.sigma_range = sigma_range
        self.p = p

    def schedule(self, t, batch_size):
        if t is None:
            t = 1 - paddle.rand(shape=batch_size)
        smin, smax, p = *self.sigma_range, self.p
        sigma = (smax ** (1 / p) + (1 - t) * (smin ** (1 / p) - smax ** (1 / p))) ** p
        logSNR = (1 / sigma**2).log()
        return logSNR


class EDMTrainSchedule(BaseSchedule):
    def setup(self, mu=-1.2, std=1.2):
        self.mu = mu
        self.std = std

    def schedule(self, t, batch_size):
        if t is not None:
            raise Exception("EDMTrainSchedule doesn't support passing timesteps: t")
        logSNR = -2 * (paddle.randn(shape=batch_size) * self.std - self.mu)
        return logSNR


class LinearSchedule(BaseSchedule):
    def setup(self, logsnr_range=[-10, 10]):
        self.logsnr_range = logsnr_range

    def schedule(self, t, batch_size):
        if t is None:
            t = 1 - paddle.rand(shape=batch_size)
        logSNR = t * (self.logsnr_range[0] - self.logsnr_range[1]) + self.logsnr_range[1]
        return logSNR


class PiecewiseLinearSchedule(BaseSchedule):
    def setup(self):
        self.x = None
        self.y = None

    def piecewise_linear(self, x, xs, ys):
        indices = paddle.searchsorted(sorted_sequence=xs[:-1], values=x) - 1
        x_min, x_max = xs[indices], xs[indices + 1]
        y_min, y_max = ys[indices], ys[indices + 1]
        var = y_min + (y_max - y_min) * (x - x_min) / (x_max - x_min)
        return var

    def schedule(self, t, batch_size):
        if t is None:
            t = 1 - paddle.rand(shape=batch_size)
        var = self.piecewise_linear(t, self.x.to(t.place), self.y.to(t.place))
        logSNR = (var / (1 - var)).log()
        return logSNR


class StableDiffusionSchedule(PiecewiseLinearSchedule):
    def setup(self, linear_range=[0.00085, 0.012], total_steps=1000):
        linear_range_sqrt = [(r**0.5) for r in linear_range]
        self.x = paddle.linspace(start=0, stop=1, num=total_steps + 1)
        alphas = 1 - (linear_range_sqrt[0] * (1 - self.x) + linear_range_sqrt[1] * self.x) ** 2
        self.y = alphas.cumprod(dim=-1)


class AdaptiveTrainSchedule(BaseSchedule):
    def setup(self, logsnr_range=[-10, 10], buckets=100, min_probs=0.0):
        th = paddle.linspace(start=logsnr_range[0], stop=logsnr_range[1], num=buckets + 1)
        self.bucket_ranges = paddle.to_tensor(data=[(th[i], th[i + 1]) for i in range(buckets)])
        self.bucket_probs = paddle.ones(shape=buckets)
        self.min_probs = min_probs

    def schedule(self, t, batch_size):
        if t is not None:
            raise Exception("AdaptiveTrainSchedule doesn't support passing timesteps: t")
        norm_probs = (self.bucket_probs + self.min_probs) / (self.bucket_probs + self.min_probs).sum()
        buckets = paddle.multinomial(x=norm_probs, num_samples=batch_size, replacement=True)
        ranges = self.bucket_ranges[buckets]
        logSNR = paddle.rand(shape=batch_size) * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
        return logSNR

    def update_buckets(self, logSNR, loss, beta=0.99):
        range_mtx = self.bucket_ranges.unsqueeze(axis=0).expand(shape=[logSNR.shape[0], -1, -1]).to(logSNR.place)
        range_mask = (range_mtx[:, :, 0] <= logSNR[:, None]) * (range_mtx[:, :, 1] > logSNR[:, None]).astype(
            dtype="float32"
        )
        range_idx = range_mask.argmax(axis=-1).cpu()
        self.bucket_probs[range_idx] = self.bucket_probs[range_idx] * beta + loss.detach().cpu() * (1 - beta)


class InterpolatedSchedule(BaseSchedule):
    def setup(self, scheduler1, scheduler2, shifts=[1.0, 1.0]):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.shifts = shifts

    def schedule(self, t, batch_size):
        if t is None:
            t = 1 - paddle.rand(shape=batch_size)
        t = t.clip(min=1e-07, max=1 - 1e-07)
        low_logSNR = self.scheduler1(t, shift=self.shifts[0])
        high_logSNR = self.scheduler2(t, shift=self.shifts[1])
        return low_logSNR * t + high_logSNR * (1 - t)
