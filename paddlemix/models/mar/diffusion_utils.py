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


import numpy as np
import paddle


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
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
    logvar1, logvar2 = [
        (x if isinstance(x, paddle.Tensor) else paddle.to_tensor(data=x).to(tensor)) for x in (logvar1, logvar2)
    ]
    return 0.5 * (
        -1.0 + logvar2 - logvar1 + paddle.exp(x=logvar1 - logvar2) + (mean1 - mean2) ** 2 * paddle.exp(x=-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + paddle.nn.functional.tanh(x=np.sqrt(2.0 / np.pi) * (x + 0.044715 * paddle.pow(x=x, y=3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert tuple(x.shape) == tuple(means.shape) == tuple(log_scales.shape)
    centered_x = x - means
    inv_stdv = paddle.exp(x=-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = paddle.log(x=cdf_plus.clip(min=1e-12))
    log_one_minus_cdf_min = paddle.log(x=(1.0 - cdf_min).clip(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = paddle.where(
        condition=x < -0.999,
        x=log_cdf_plus,
        y=paddle.where(
            condition=x > 0.999,
            x=log_one_minus_cdf_min,
            y=paddle.log(x=cdf_delta.clip(min=1e-12)),
        ),
    )
    assert tuple(log_probs.shape) == tuple(x.shape)
    return log_probs
