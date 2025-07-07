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
from functools import partial


def fn_LinearWarmup(warmup_steps, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0


def Scheduler_LinearWarmup(warmup_steps):
    return partial(fn_LinearWarmup, warmup_steps)


def fn_LinearWarmup_CosineDecay(warmup_steps, max_steps, multiplier_min, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        multiplier = 0.5 * (math.cos((step - warmup_steps) / (max_steps - warmup_steps) * math.pi) + 1)
        return max(multiplier, multiplier_min)


def Scheduler_LinearWarmup_CosineDecay(warmup_steps, max_steps, multiplier_min):
    return partial(fn_LinearWarmup_CosineDecay, warmup_steps, max_steps, multiplier_min)
