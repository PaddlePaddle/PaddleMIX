# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


class AbstractDistribution:
    def sample(self):
        raise NotImplementedError()

    def mode(self):
        raise NotImplementedError()


class DiracDistribution(AbstractDistribution):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value

    def mode(self):
        return self.value


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = paddle.chunk(x=parameters, chunks=2, axis=1)
        self.logvar = paddle.clip(x=self.logvar, min=-30.0, max=20.0)
        self.deterministic = deterministic
        self.std = paddle.exp(x=(0.5 * self.logvar).astype("float32"))
        self.var = paddle.exp(x=self.logvar.astype("float32"))
        if self.deterministic:
            self.var = self.std = paddle.zeros_like(x=self.mean)

    def sample(self, noise=None):
        if noise is None:
            noise = paddle.randn(shape=self.mean.shape)
        x = self.mean + self.std * noise
        return x

    def kl(self, other=None):
        if self.deterministic:
            return paddle.to_tensor(data=[0.0], dtype="float32")
        elif other is None:
            return 0.5 * paddle.sum(x=paddle.pow(x=self.mean, y=2) + self.var - 1.0 - self.logvar, axis=[1, 2, 3])
        else:
            return 0.5 * paddle.sum(
                x=paddle.pow(x=self.mean - other.mean, y=2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                axis=[1, 2, 3],
            )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return paddle.to_tensor(data=[0.0], dtype="float32")
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * paddle.sum(x=logtwopi + self.logvar + paddle.pow(x=sample - self.mean, y=2) / self.var, axis=dims)

    def mode(self):
        return self.mean


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, paddle.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"
    logvar1, logvar2 = [(x if isinstance(x, paddle.Tensor) else paddle.to_tensor(data=x)) for x in (logvar1, logvar2)]
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + paddle.exp(x=(logvar1 - logvar2).astype("float32"))
        + (mean1 - mean2) ** 2 * paddle.exp(x=(-logvar2).astype("float32"))
    )
