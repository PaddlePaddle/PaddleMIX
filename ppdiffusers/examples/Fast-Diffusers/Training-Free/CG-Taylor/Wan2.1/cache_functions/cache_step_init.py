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


def cache_step_init(num_steps=50):
    """
    Initialization for cache.
    """
    cache_dic = {}
    cache = {}
    cache[-1] = {}
    cache[-1]["cond_stream"] = {}
    cache[-1]["uncond_stream"] = {}
    cache_dic["cache_counter"] = 0
    cache["cond_hidden"] = {}
    cache["uncond_hidden"] = {}
    cache["firstblock_hidden"] = {}

    cache_dic["taylor_cache"] = False
    cache_dic["Delta-DiT"] = False

    cache_dic["cache_type"] = "random"
    cache_dic["fresh_ratio_schedule"] = "ToCa"
    cache_dic["fresh_ratio"] = 0.0
    cache_dic["fresh_threshold"] = 1
    cache_dic["force_fresh"] = "global"

    mode = "Taylor"

    if mode == "original":
        cache_dic["cache"] = cache
        cache_dic["force_fresh"] = "global"
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 3

    elif mode == "ToCa":
        cache_dic["cache_type"] = "attention"
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio_schedule"] = "ToCa"
        cache_dic["fresh_ratio"] = 0.1
        cache_dic["fresh_threshold"] = 5
        cache_dic["force_fresh"] = "global"
        cache_dic["soft_fresh_weight"] = 0.0
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 3

    elif mode == "Taylor":
        cache_dic["cache"] = cache
        cache_dic["fresh_threshold"] = 5
        cache_dic["taylor_cache"] = True
        cache_dic["max_order"] = 1
        cache_dic["firstblock_max_order"] = 3
        cache_dic["first_enhance"] = 1

    elif mode == "Delta":
        cache_dic["cache"] = cache
        cache_dic["fresh_ratio"] = 0.0
        cache_dic["fresh_threshold"] = 3
        cache_dic["Delta-DiT"] = True
        cache_dic["max_order"] = 0
        cache_dic["first_enhance"] = 1

    current = {}
    current["activated_steps"] = [0]
    current["block_activated_steps"] = [0]
    current["step"] = 0
    current["num_steps"] = num_steps

    return cache_dic, current
