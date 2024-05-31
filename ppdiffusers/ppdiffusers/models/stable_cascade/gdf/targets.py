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


class EpsilonTarget:
    def __call__(self, x0, epsilon, logSNR, a, b):
        return epsilon

    def x0(self, noised, pred, logSNR, a, b):
        return (noised - pred * b) / a

    def epsilon(self, noised, pred, logSNR, a, b):
        return pred


class X0Target:
    def __call__(self, x0, epsilon, logSNR, a, b):
        return x0

    def x0(self, noised, pred, logSNR, a, b):
        return pred

    def epsilon(self, noised, pred, logSNR, a, b):
        return (noised - pred * a) / b


class VTarget:
    def __call__(self, x0, epsilon, logSNR, a, b):
        return a * epsilon - b * x0

    def x0(self, noised, pred, logSNR, a, b):
        squared_sum = a**2 + b**2
        return a / squared_sum * noised - b / squared_sum * pred

    def epsilon(self, noised, pred, logSNR, a, b):
        squared_sum = a**2 + b**2
        return b / squared_sum * noised + a / squared_sum * pred


class RectifiedFlowsTarget:
    def __call__(self, x0, epsilon, logSNR, a, b):
        return epsilon - x0

    def x0(self, noised, pred, logSNR, a, b):
        return noised - pred * b

    def epsilon(self, noised, pred, logSNR, a, b):
        return noised + pred * a
