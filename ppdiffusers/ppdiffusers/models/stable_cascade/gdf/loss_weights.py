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


class BaseLossWeight:
    def weight(self, logSNR):
        raise NotImplementedError("this method needs to be overridden")

    def __call__(self, logSNR, *args, shift=1, clamp_range=None, **kwargs):
        clamp_range = [-1000000000.0, 1000000000.0] if clamp_range is None else clamp_range
        if shift != 1:
            logSNR = logSNR.clone() + 2 * np.log(shift)
        return self.weight(logSNR, *args, **kwargs).clip(*clamp_range)


class ComposedLossWeight(BaseLossWeight):
    def __init__(self, div, mul):
        self.mul = [mul] if isinstance(mul, BaseLossWeight) else mul
        self.div = [div] if isinstance(div, BaseLossWeight) else div

    def weight(self, logSNR):
        prod, div = 1, 1
        for m in self.mul:
            prod *= m.weight(logSNR)
        for d in self.div:
            div *= d.weight(logSNR)
        return prod / div


class ConstantLossWeight(BaseLossWeight):
    def __init__(self, v=1):
        self.v = v

    def weight(self, logSNR):
        return paddle.ones_like(x=logSNR) * self.v


class SNRLossWeight(BaseLossWeight):
    def weight(self, logSNR):
        return logSNR.exp()


class P2LossWeight(BaseLossWeight):
    def __init__(self, k=1.0, gamma=1.0, s=1.0):
        self.k, self.gamma, self.s = k, gamma, s

    def weight(self, logSNR):
        return (self.k + (logSNR * self.s).exp()) ** -self.gamma


class SNRPlusOneLossWeight(BaseLossWeight):
    def weight(self, logSNR):
        return logSNR.exp() + 1


class MinSNRLossWeight(BaseLossWeight):
    def __init__(self, max_snr=5):
        self.max_snr = max_snr

    def weight(self, logSNR):
        return logSNR.exp().clip(max=self.max_snr)


class MinSNRPlusOneLossWeight(BaseLossWeight):
    def __init__(self, max_snr=5):
        self.max_snr = max_snr

    def weight(self, logSNR):
        return (logSNR.exp() + 1).clip(max=self.max_snr)


class TruncatedSNRLossWeight(BaseLossWeight):
    def __init__(self, min_snr=1):
        self.min_snr = min_snr

    def weight(self, logSNR):
        return logSNR.exp().clip(min=self.min_snr)


class SechLossWeight(BaseLossWeight):
    def __init__(self, div=2):
        self.div = div

    def weight(self, logSNR):
        return 1 / (logSNR / self.div).cosh()


class DebiasedLossWeight(BaseLossWeight):
    def weight(self, logSNR):
        return 1 / logSNR.exp().sqrt()


class SigmoidLossWeight(BaseLossWeight):
    def __init__(self, s=1):
        self.s = s

    def weight(self, logSNR):
        return (logSNR * self.s).sigmoid()


class AdaptiveLossWeight(BaseLossWeight):
    def __init__(self, logsnr_range=[-10, 10], buckets=300, weight_range=[1e-07, 10000000.0]):
        self.bucket_ranges = paddle.linspace(start=logsnr_range[0], stop=logsnr_range[1], num=buckets - 1)
        self.bucket_losses = paddle.ones(shape=buckets)
        self.weight_range = weight_range

    def weight(self, logSNR):
        indices = paddle.searchsorted(sorted_sequence=self.bucket_ranges.to(logSNR.place), values=logSNR)
        return (1 / self.bucket_losses.to(logSNR.place)[indices]).clip([*self.weight_range])

    def update_buckets(self, logSNR, loss, beta=0.99):
        indices = paddle.searchsorted(sorted_sequence=self.bucket_ranges.to(logSNR.place), values=logSNR).cpu()
        self.bucket_losses[indices] = self.bucket_losses[indices] * beta + loss.detach().cpu() * (1 - beta)
