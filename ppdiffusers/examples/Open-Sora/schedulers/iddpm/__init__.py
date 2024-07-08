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

from functools import partial

import paddle

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


class IDDPM(SpacedDiffusion):
    def __init__(
        self,
        num_sampling_steps=None,
        timestep_respacing=None,
        noise_schedule="linear",
        use_kl=False,
        sigma_small=False,
        predict_xstart=False,
        learn_sigma=True,
        rescale_learned_sigmas=False,
        diffusion_steps=1000,
        cfg_scale=4.0,
        cfg_channel=None,
    ):
        betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE
        if num_sampling_steps is not None:
            assert timestep_respacing is None
            timestep_respacing = str(num_sampling_steps)
        if timestep_respacing is None or timestep_respacing == "":
            timestep_respacing = [diffusion_steps]
        super().__init__(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=(gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X),
            model_var_type=(
                (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            # rescale_timesteps=rescale_timesteps,
        )

        self.cfg_scale = cfg_scale
        self.cfg_channel = cfg_channel

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        additional_args=None,
        mask=None,
    ):
        n = len(prompts)
        z = paddle.concat(x=[z, z], axis=0)
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n).astype(model_args["y"].dtype)
        model_args["y"] = paddle.concat(x=[model_args["y"], y_null], axis=0)
        if additional_args is not None:
            model_args.update(additional_args)

        forward = partial(forward_with_cfg, model, cfg_scale=self.cfg_scale, cfg_channel=self.cfg_channel)
        samples = self.p_sample_loop(
            forward,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_args,
            progress=True,
            mask=mask,
        )
        samples, _ = samples.chunk(chunks=2, axis=0)
        return samples


def forward_with_cfg(model, x, timestep, y, cfg_scale, cfg_channel=None, **kwargs):
    # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    half = x[: len(x) // 2]
    combined = paddle.concat(x=[half, half], axis=0)
    if "x_mask" in kwargs and kwargs["x_mask"] is not None:
        if len(kwargs["x_mask"]) != len(x):
            kwargs["x_mask"] = paddle.concat(x=[kwargs["x_mask"], kwargs["x_mask"]], axis=0)
    model_out = model.forward(combined, timestep, y, **kwargs)
    model_out = model_out["x"] if isinstance(model_out, dict) else model_out
    if cfg_channel is None:
        cfg_channel = model_out.shape[1] // 2
    eps, rest = model_out[:, :cfg_channel], model_out[:, cfg_channel:]

    cond_eps, uncond_eps = paddle.split(eps, round(eps.shape[0] / (len(eps) // 2)), axis=0)

    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = paddle.concat(x=[half_eps, half_eps], axis=0)
    return paddle.concat(x=[eps, rest], axis=1)
