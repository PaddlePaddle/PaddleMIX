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

from ppdiffusers.models import FluxTransformer2DModel


def cache_init(self: FluxTransformer2DModel):
    """
    Initialization for cache.
    """
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1] = {}
    cache_index[-1] = {}
    cache_index["layer_index"] = {}
    cache_dic["block"] = {}
    cache_dic["block"][-1] = {}
    cache_dic["block"][-1]["double_stream"] = {}
    cache_dic["block"][-1]["single_stream"] = {}

    cache[-1]["double_stream"] = {}
    cache[-1]["single_stream"] = {}
    cache_dic["cache_counter"] = 0

    for j in range(self.config.num_layers):
        cache[-1]["double_stream"][j] = {}
        cache[-1]["double_stream"][j]["hidden_states"] = {}
        cache[-1]["double_stream"][j]["encoder_hidden_states"] = {}
        cache_index[-1][j] = {}
        cache_dic["block"][-1]["double_stream"][j] = {}
        cache_dic["block"][-1]["double_stream"][j]["hidden_states"] = {}
        cache_dic["block"][-1]["double_stream"][j]["encoder_hidden_states"] = {}

    for j in range(self.config.num_single_layers):
        cache[-1]["single_stream"][j] = {}
        cache[-1]["single_stream"][j]["hidden_states"] = {}
        cache_index[-1][j] = {}
        cache_dic["block"][-1]["single_stream"][j] = {}
        cache_dic["block"][-1]["single_stream"][j]["hidden_states"] = {}

    cache_dic["taylor_cache"] = False
    cache_dic["Delta-DiT"] = False

    mode = "Taylor"

    if mode == "original":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 1
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 3

    elif mode == "ToCa":
        cache_dic["cache_type"] = "attention"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.1
        cache_dic["fresh_threshold"] = 5
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 3

    elif mode == "Taylor":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 6
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["taylor_cache"] = True
        cache_dic["max_order"] = 1
        cache_dic["first_enhance"] = 3

    elif mode == "Delta":
        cache_dic["cache_type"] = "random"
        cache_dic["cache_index"] = cache_index
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 3
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["Delta-DiT"] = True
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 1

    current = {}
    current["activated_steps"] = [0]
    current["step"] = 0
    current["num_steps"] = self.num_steps

    return cache_dic, current
