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

import importlib
from inspect import isfunction

import numpy as np
import paddle
import paddle.nn as nn


class DiffusionWrapper(nn.Layer):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)

        self.conditioning_key = conditioning_key

        for key in self.conditioning_key:
            if "concat" in key or "crossattn" in key or "hybrid" in key or "film" in key or "noncond" in key:
                continue
            else:
                raise ValueError("The conditioning key %s is illegal" % key)

        self.being_verbosed_once = False

    def forward(self, x, t, cond_dict: dict = {}):
        # x with condition (or maybe not)
        xc = x

        y = None
        context_list, attn_mask_list = [], []

        conditional_keys = cond_dict.keys()

        for key in conditional_keys:
            if "concat" in key:
                xc = paddle.concat([x, cond_dict[key].unsqueeze(1)], axis=1)
            elif "film" in key:
                if y is None:
                    y = cond_dict[key].squeeze(1)
                else:
                    y = paddle.concat([y, cond_dict[key].squeeze(1)], axis=-1)
            elif "crossattn" in key:
                # assert context is None, "You can only have one context matrix, got %s" % (cond_dict.keys())
                if isinstance(cond_dict[key], dict):
                    for k in cond_dict[key].keys():
                        if "crossattn" in k:
                            context, attn_mask = cond_dict[key][
                                k
                            ]  # crossattn_audiomae_pooled: paddle.Size([12, 128, 768])
                else:
                    assert (
                        len(cond_dict[key]) == 2
                    ), "The context condition for %s you returned should have two element, one context one mask" % (
                        key
                    )
                    context, attn_mask = cond_dict[key]

                # The input to the UNet model is a list of context matrix
                context_list.append(context)
                attn_mask_list.append(attn_mask)

            elif (
                "noncond" in key
            ):  # If you use loss function in the conditional module, include the keyword "noncond" in the return dictionary
                continue
            else:
                raise NotImplementedError()

        out = self.diffusion_model(xc, t, context_list=context_list, y=y, context_attn_mask_list=attn_mask_list)

        return out


def instantiate_from_config(config):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package="paddlemix.models.audioldm2"), cls)


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        tmp = float(total_params * 1.0e-6)
        print(f"{model.__class__.__name__} has {tmp:.2f} M params.")
    return total_params


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = paddle.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype="float64") ** 2

    elif schedule == "cosine":
        timesteps = paddle.arange(n_timestep + 1, dtype="float64") / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = paddle.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = paddle.linspace(linear_start, linear_end, n_timestep, dtype="float64")
    elif schedule == "sqrt":
        betas = paddle.linspace(linear_start, linear_end, n_timestep, dtype="float64") ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(t, -1)
    return out.reshape((b,) + ((1,) * (len(x_shape) - 1)))


def noise_like(shape, repeat=False):
    repeat_noise = lambda: paddle.randn((1, *shape[1:])).repeat_interleave(repeats=shape[0], axis=0)
    noise = lambda: paddle.randn(shape)
    return repeat_noise() if repeat else noise()


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self
