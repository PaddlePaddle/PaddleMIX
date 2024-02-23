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

import math

from paddle.optimizer.lr import LRScheduler

from paddlemix.utils.log import logger

__all__ = [
    "CosineDecayWithWarmup",
    "FilterParamsName",
]


class CosineDecayWithWarmup(LRScheduler):
    """
    Cosine learning rate decay
    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        eta_min(float): Minimum learning rate. Default: 0.0.
        warmup_start_lr(float): Initial learning rate of warm up. Default: 0.0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self, learning_rate, total_steps, eta_min=0.0, warmup_start_lr=0.0, last_step=-1, warmup=0, **kwargs):
        self.start_lr = learning_rate
        self.eta_min = eta_min
        self.warmup_start_lr = warmup_start_lr
        self.last_lr = self.start_lr
        self.last_step = last_step
        self.total_steps = total_steps

        if isinstance(warmup, int):
            self.warmup_steps = warmup
        elif isinstance(warmup, float):
            self.warmup_steps = int(warmup * total_steps)
        else:
            raise ValueError("Warmup expected a int or float number, but received: {}".format(type(warmup)))
        self.step_each_epoch = kwargs.get("step_each_epoch", None)
        if self.warmup_steps > 0:
            self.last_lr = self.warmup_start_lr
        super().__init__(learning_rate=self.last_lr, last_epoch=self.last_step)

    def step(self):
        global_cur_step = self.last_step + 1
        if global_cur_step < self.warmup_steps:
            self.last_lr = self.warmup_start_lr + (self.start_lr - self.warmup_start_lr) * global_cur_step / max(
                self.warmup_steps, 1
            )
        else:
            if self.step_each_epoch:
                self.last_lr = (self.start_lr - self.eta_min) * 0.5 * (
                    1.0 + math.cos(math.pi * global_cur_step // self.total_steps)
                ) + self.eta_min
            else:
                self.last_lr = (self.start_lr - self.eta_min) * 0.5 * (
                    1.0 + math.cos(math.pi * global_cur_step / self.total_steps)
                ) + self.eta_min
        self.last_step += 1

    def get_lr(self):
        return self.last_lr


class FilterParamsName(object):
    """
    FilterParamsName is a utility class to filter out some params from optimizer.
    """

    def __init__(self, non_wd_keyword=["bias", "ln", "bn"]):
        self.p_non_wd_name = []
        self.non_wd_keyword = non_wd_keyword

    def __call__(self, model):
        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in model.named_parameters():
            if p.stop_gradient:
                continue  # frozen weights
            if p.ndim < 2 or any([key in n for key in self.non_wd_keyword]):
                p_non_wd.append(p)
                self.p_non_wd_name.append(n)
            else:
                p_wd.append(p)
            num_parameters += p.numel()
        logger.info("number of trainable parameters: %d" % num_parameters)
        return p_wd, p_non_wd

    def _apply_decay_param_fun(self, name):
        return name not in self.p_non_wd_name
