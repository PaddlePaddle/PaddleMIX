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

import sys

import paddle
import paddle_aux

import ppdiffusers

from .loss_weights import *
from .noise_conditions import *
from .samplers import *
from .scalers import *
from .schedulers import *
from .targets import *


class GDF:
    def __init__(self, schedule, input_scaler, target, noise_cond, loss_weight, offset_noise=0):
        self.schedule = schedule
        self.input_scaler = input_scaler
        self.target = target
        self.noise_cond = noise_cond
        self.loss_weight = loss_weight
        self.offset_noise = offset_noise

    def setup_limits(self, stretch_max=True, stretch_min=True, shift=1):
        stretched_limits = self.input_scaler.setup_limits(
            self.schedule, self.input_scaler, stretch_max, stretch_min, shift
        )
        return stretched_limits

    def diffuse(self, x0, epsilon=None, t=None, shift=1, loss_shift=1, offset=None):
        if epsilon is None:
            epsilon = paddle.randn(shape=x0.shape, dtype=x0.dtype)

        if self.offset_noise > 0:
            if offset is None:
                offset = paddle.randn(
                    shape=[x0.shape[0], x0.shape[1]] + [1] * (len(x0.shape) - 2),
                )
            epsilon = epsilon + offset * self.offset_noise
        logSNR = self.schedule(x0.shape[0] if t is None else t, shift=shift)
        a, b = self.input_scaler(logSNR)
        if len(a.shape) == 1:
            a, b = a.reshape([-1, *([1] * (len(x0.shape) - 1))]), b.reshape([-1, *([1] * (len(x0.shape) - 1))])
        target = self.target(x0, epsilon, logSNR, a, b)
        return (
            x0 * a + epsilon * b,
            epsilon,
            target,
            logSNR,
            self.noise_cond(logSNR),
            self.loss_weight(logSNR, shift=loss_shift),
        )

    def undiffuse(self, x, logSNR, pred):
        a, b = self.input_scaler(logSNR)
        if len(a.shape) == 1:
            a, b = a.reshape([-1, *([1] * (len(x.shape) - 1))]), b.reshape([-1, *([1] * (len(x.shape) - 1))])
        return self.target.x0(x, pred, logSNR, a, b), self.target.epsilon(x, pred, logSNR, a, b)

    def sample(
        self,
        model,
        model_inputs,
        shape,
        unconditional_inputs=None,
        sampler=None,
        schedule=None,
        t_start=1.0,
        t_end=0.0,
        timesteps=20,
        x_init=None,
        cfg=3.0,
        cfg_t_stop=None,
        cfg_t_start=None,
        cfg_rho=0.7,
        sampler_params=None,
        shift=1,
        device="cpu",
    ):
        sampler_params = {} if sampler_params is None else sampler_params
        if sampler is None:
            sampler = DDPMSampler(self)  # noqa
        r_range = paddle.linspace(start=t_start, stop=t_end, num=timesteps + 1)
        schedule = self.schedule if schedule is None else schedule
        logSNR_range = (
            schedule(r_range, shift=shift)[:, None]
            .expand(shape=[-1, shape[0] if x_init is None else x_init.shape[0]])
            .to(device)
        )
        x = sampler.init_x(shape).to(device) if x_init is None else x_init.clone()
        if cfg is not None:
            if unconditional_inputs is None:
                unconditional_inputs = {k: paddle.zeros_like(x=v) for k, v in model_inputs.items()}
            model_inputs = {
                k: (
                    paddle.concat(x=[v, v_u], axis=0)
                    if isinstance(v, paddle.Tensor)
                    else [
                        (
                            paddle.concat(x=[vi, vi_u], axis=0)
                            if isinstance(vi, paddle.Tensor) and isinstance(vi_u, paddle.Tensor)
                            else None
                        )
                        for vi, vi_u in zip(v, v_u)
                    ]
                    if isinstance(v, list)
                    else {vk: paddle.concat(x=[v[vk], v_u.get(vk, paddle.zeros_like(x=v[vk]))], axis=0) for vk in v}
                    if isinstance(v, dict)
                    else None
                )
                for (k, v), (k_u, v_u) in zip(model_inputs.items(), unconditional_inputs.items())
            }
        for i in range(0, timesteps):
            noise_cond = self.noise_cond(logSNR_range[i])
            if (
                cfg is not None
                and (cfg_t_stop is None or r_range[i].item() >= cfg_t_stop)
                and (cfg_t_start is None or r_range[i].item() <= cfg_t_start)
            ):
                cfg_val = cfg
                if isinstance(cfg_val, (list, tuple)):
                    assert len(cfg_val) == 2, "cfg must be a float or a list/tuple of length 2"
                    cfg_val = cfg_val[0] * r_range[i].item() + cfg_val[1] * (1 - r_range[i].item())

                pred, pred_unconditional = model(
                    paddle.concat(x=[x, x], axis=0), noise_cond.repeat(2), **model_inputs
                ).chunk(chunks=2)

                pred_cfg = paddle.lerp(pred_unconditional, pred, paddle.to_tensor(cfg_val, dtype=paddle.float32))
                if cfg_rho > 0:
                    std_pos, std_cfg = pred.std(), pred_cfg.std()
                    pred = cfg_rho * (pred_cfg * std_pos / (std_cfg + 1e-9)) + pred_cfg * (1 - cfg_rho)
                else:
                    pred = pred_cfg
            else:
                pred = model(x, noise_cond, **model_inputs)

            x0, epsilon = self.undiffuse(x, logSNR_range[i], pred)
            x = sampler(x, x0, epsilon, logSNR_range[i], logSNR_range[i + 1], **sampler_params)
            altered_vars = yield x0, x, pred
            if altered_vars is not None:
                cfg = altered_vars.get("cfg", cfg)
                cfg_rho = altered_vars.get("cfg_rho", cfg_rho)
                sampler = altered_vars.get("sampler", sampler)
                model_inputs = altered_vars.get("model_inputs", model_inputs)
                x = altered_vars.get("x", x)
                x_init = altered_vars.get("x_init", x_init)
