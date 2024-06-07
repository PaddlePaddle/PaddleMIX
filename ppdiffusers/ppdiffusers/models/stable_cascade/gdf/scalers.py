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


class BaseScaler:
    def __init__(self):
        self.stretched_limits = None

    def setup_limits(self, schedule, input_scaler, stretch_max=True, stretch_min=True, shift=1):
        min_logSNR = schedule(paddle.ones(shape=[1]), shift=shift)
        max_logSNR = schedule(paddle.zeros(shape=[1]), shift=shift)
        min_a, max_b = [v.item() for v in input_scaler(min_logSNR)] if stretch_max else [0, 1]
        max_a, min_b = [v.item() for v in input_scaler(max_logSNR)] if stretch_min else [1, 0]
        self.stretched_limits = [min_a, max_a, min_b, max_b]
        return self.stretched_limits

    def stretch_limits(self, a, b):
        min_a, max_a, min_b, max_b = self.stretched_limits
        return (a - min_a) / (max_a - min_a), (b - min_b) / (max_b - min_b)

    def scalers(self, logSNR):
        raise NotImplementedError("this method needs to be overridden")

    def __call__(self, logSNR):
        a, b = self.scalers(logSNR)
        if self.stretched_limits is not None:
            a, b = self.stretch_limits(a, b)
        return a, b


class VPScaler(BaseScaler):
    def scalers(self, logSNR):
        a_squared = logSNR.sigmoid()
        a = a_squared.sqrt()
        b = (1 - a_squared).sqrt()
        return a, b


class LERPScaler(BaseScaler):
    def scalers(self, logSNR):
        _a = logSNR.exp() - 1
        _a[_a == 0] = 0.001
        a = 1 + (2 - (2**2 + 4 * _a) ** 0.5) / (2 * _a)
        b = 1 - a
        return a, b
