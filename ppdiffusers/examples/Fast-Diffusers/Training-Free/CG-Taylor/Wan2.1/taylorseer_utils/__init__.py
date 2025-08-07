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

import math
from typing import Dict

import paddle


def firstblock_derivative_approximation(cache_dic: Dict, current: Dict, feature: paddle.Tensor):
    """
    Compute derivative approximation.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current["block_activated_steps"][-1] - current["block_activated_steps"][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic["firstblock_max_order"]):
        if (cache_dic["cache"]["firstblock_hidden"].get(i, None) is not None) and (
            current["step"] > cache_dic["first_enhance"] - 2
        ):
            updated_taylor_factors[i + 1] = (
                updated_taylor_factors[i] - cache_dic["cache"]["firstblock_hidden"][i]
            ) / difference_distance
        else:
            break

    cache_dic["cache"]["firstblock_hidden"] = updated_taylor_factors


def firstblock_taylor_formula(cache_dic: Dict, current: Dict) -> paddle.Tensor:
    """
    Compute Taylor expansion error.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    x = current["step"] - current["block_activated_steps"][-1]

    output = 0

    for i in range(len(cache_dic["cache"]["firstblock_hidden"])):

        output += (1 / math.factorial(i)) * cache_dic["cache"]["firstblock_hidden"][i] * (x**i)

    return output


def step_uncond_derivative_approximation(cache_dic: Dict, current: Dict, feature: paddle.Tensor):
    """
    Compute derivative approximation.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic["max_order"]):
        if (cache_dic["cache"]["uncond_hidden"].get(i, None) is not None) and (
            current["step"] > cache_dic["first_enhance"] - 2
        ):
            updated_taylor_factors[i + 1] = (
                updated_taylor_factors[i] - cache_dic["cache"]["uncond_hidden"][i]
            ) / difference_distance
        else:
            break

    cache_dic["cache"]["uncond_hidden"] = updated_taylor_factors


def step_cond_derivative_approximation(cache_dic: Dict, current: Dict, feature: paddle.Tensor):
    """
    Compute derivative approximation.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic["max_order"]):
        if (cache_dic["cache"]["cond_hidden"].get(i, None) is not None) and (
            current["step"] > cache_dic["first_enhance"] - 2
        ):
            updated_taylor_factors[i + 1] = (
                updated_taylor_factors[i] - cache_dic["cache"]["cond_hidden"][i]
            ) / difference_distance
        else:
            break

    cache_dic["cache"]["cond_hidden"] = updated_taylor_factors


def derivative_approximation(cache_dic: Dict, current: Dict, feature: paddle.Tensor):
    """
    Compute derivative approximation.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic["max_order"]):
        if (
            cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]].get(i, None) is not None
        ) and (current["step"] > cache_dic["first_enhance"] - 2):
            updated_taylor_factors[i + 1] = (
                updated_taylor_factors[i]
                - cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]][i]
            ) / difference_distance
        else:
            break

    cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = updated_taylor_factors


def taylor_formula(derivative_dict: Dict, distance: int) -> paddle.Tensor:
    """
    Compute Taylor expansion error.

    :param derivative_dict: Derivative dictionary
    :param x: Current step
    """
    output = 0
    for i in range(len(derivative_dict)):
        output += (1 / math.factorial(i)) * derivative_dict[i] * (distance**i)

    return output


def taylor_cache_init(cache_dic: Dict, current: Dict):
    """
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if (current["step"] == 0) and (cache_dic["taylor_cache"]):
        cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = {}
