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

# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

import math

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, (paddle.Tensor, paddle.static.Variable, paddle.base.libpaddle.pir.Value)):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for paddle.exp().
    logvar1, logvar2 = [
        x
        if isinstance(x, (paddle.Tensor, paddle.static.Variable, paddle.base.libpaddle.pir.Value))
        else paddle.to_tensor(x, place=tensor.place)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0 + logvar2 - logvar1 + paddle.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * paddle.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + paddle.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * paddle.pow(x, 3))))


def continuous_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a continuous Gaussian distribution.
    :param x: the targets
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    centered_x = x - means
    inv_stdv = paddle.exp(-log_scales)
    normalized_x = centered_x * inv_stdv
    log_probs = paddle.distributions.Normal(paddle.zeros_like(x), paddle.ones_like(x)).log_prob(normalized_x)
    return log_probs


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
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = paddle.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = paddle.log(cdf_plus.clip(min=1e-12))
    log_one_minus_cdf_min = paddle.log((1.0 - cdf_min).clip(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = paddle.where(
        x < -0.999,
        log_cdf_plus,
        paddle.where(x > 0.999, log_one_minus_cdf_min, paddle.log(cdf_delta.clip(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def get_mesh(pp_idx=0):
    """
    To obtain ProcessMesh in auto parallel, you can choose to specify pipeline index.

    Args:
        pp_idx (int, optional, default 0): pipeline index, default to 0.

    Returns:
        ProcessMesh:  paddle.distributed.ProcessMesh object containing mesh information.
    """
    mesh = fleet.auto.get_mesh()
    if "pp" in mesh.dim_names:
        mesh = mesh.get_mesh_with_dim("pp")[pp_idx]
    return mesh


def get_layer_ipp(layer_index, num_layers):
    """
    Retrieve the position of the specified layer in the pipeline parallel network, and return None if it is not an pipeline parallel network.

    Args:
        layer_index (int): The layer index that needs to be queried is counted from 0.
        num_layers (int): The total number of transformer layers in the pipeline parallel network.

    Returns:
        Optional[int, bool]: If it is an pipeline parallel network, return the position of the specified layer and whether the input needs to be resharded; Otherwise, return None and False.
    """
    mesh = fleet.auto.get_mesh()
    if "pp" not in mesh.dim_names:
        return None, False
    else:
        pp_degree = mesh.get_dim_size("pp")
        layer_per_stage = math.ceil(num_layers / pp_degree)
        input_need_reshard = layer_index % layer_per_stage == 0
        return layer_index // layer_per_stage, input_need_reshard


def shard_w(w, pp_stage, placements):
    """
    sharding the given weight parameters and return the sharded weight parameters.

    Args:
        w (Parameter): The weight parameters to be sharded. If the parameter is not initialized, it will be initialized first.
        pp_stage (int): The current pipeline stage is used to determine which mesh to use for sharding.
        placements (list[paddle.distributed.Placement]): the placements describe how to place the tensor on ProcessMesh, it can
            be Shard, Replicate and Partial.

    Returns:
        Parameter: the weight parameters after sharding. Its name remains unchanged.
    """
    assert w._is_initialized()
    paran_name = w.name
    # print(f"shard w {paran_name}")
    w = dist.shard_tensor(w, get_mesh(pp_stage), placements)
    w.name = paran_name
    return w
