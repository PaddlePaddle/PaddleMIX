import paddle
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
        multiplier = 0.5 * (math.cos((step - warmup_steps) / (max_steps -
            warmup_steps) * math.pi) + 1)
        return max(multiplier, multiplier_min)


def Scheduler_LinearWarmup_CosineDecay(warmup_steps, max_steps, multiplier_min):
    return partial(fn_LinearWarmup_CosineDecay, warmup_steps, max_steps,
        multiplier_min)
