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

from paddlevlp.utils.log import logger

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

    def __init__(
        self,
        learning_rate,
        epochs,
        eta_min=0.0,
        warmup_steps=0,
        warmup_start_lr=0.0,
        last_epoch=-1,
        step_each_epoch=1,
        **kwargs
    ):
        self.start_lr = learning_rate
        self.T_max = epochs
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.last_lr = self.start_lr
        self.cur_step = 0
        self.last_epoch = last_epoch
        self.step_each_epoch = step_each_epoch
        if self.warmup_steps > 0:
            self.last_lr = self.warmup_start_lr
        super().__init__(learning_rate=self.last_lr, last_epoch=self.last_epoch)

    def step(self):
        self.cur_step += 1
        cur_step_in_epoch = (self.cur_step - 2) % self.step_each_epoch
        if self.cur_step < self.warmup_steps and self.last_epoch == 0:
            self.last_lr = self.warmup_start_lr + (
                self.start_lr - self.warmup_start_lr
            ) * cur_step_in_epoch / max(self.warmup_steps, 1)
        else:
            self.last_lr = (self.start_lr - self.eta_min) * 0.5 * (
                1.0 + math.cos(math.pi * self.last_epoch / self.T_max)
            ) + self.eta_min
        self.last_epoch += 1

    def get_lr(self):
        return self.last_lr


class FilterParamsName(object):
    """
    FilterParamsName is a utility class to filter out some params from optimizer.
    """

    def __init__(self):
        self.p_non_wd_name = []

    def __call__(self, model):
        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in model.named_parameters():
            if p.stop_gradient:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                # print(n)
                p_non_wd.append(p)
                self.p_non_wd_name.append(n)
            else:
                p_wd.append(p)
            num_parameters += p.numel()
        logger.info("number of trainable parameters: %d" % num_parameters)
        return p_wd, p_non_wd

    def apply_decay_param_fun(self, name):
        return name not in self.p_non_wd_name
