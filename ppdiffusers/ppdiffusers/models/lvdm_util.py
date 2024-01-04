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
import random
from inspect import isfunction

import numpy as np
import paddle
from einops import repeat


def make_interp_mask_with_bothsidescond(t, device, n_interp1, n_interp2):
    """1: cond frames
    0: generated frames
    """
    mask = paddle.zeros(shape=[t])
    mask[:n_interp1] = 1
    mask[t - n_interp2 :] = 1
    return mask


def make_interp_mask_with_framestride(t, device, frame_stride):
    """1: cond frames
    0: generated frames
    """
    mask = paddle.zeros(shape=[t])
    for i in range(0, t, frame_stride):
        mask[i] = 1
    return mask


def random_temporal_masking(
    input_shape, p_interp, p_pred, device, n_interp1=1, n_interp2=1, n_prevs=[1], interp_frame_stride=None
):
    """return mask for masking input, where 1 indicates given real image as condition,
    0 indicates noisy samples.
    """
    if p_pred == 0.0:
        n_prevs = None
    b, c, t, h, w = input_shape
    mask = paddle.zeros(shape=[b, t])
    for i in range(b):
        r = random.random()
        if r < p_interp:
            if interp_frame_stride is not None:
                mask[i] = make_interp_mask_with_framestride(t, device, interp_frame_stride)
            else:
                mask[i] = make_interp_mask_with_bothsidescond(t, device, n_interp1, n_interp2)
        elif p_interp <= r < p_interp + p_pred:
            n_pred = random.choice(n_prevs)
            mask[(i), :n_pred] = 1
        else:
            pass
    mask = mask.unsqueeze(axis=1).unsqueeze(axis=3).unsqueeze(axis=4)
    mask = mask.tile(repeat_times=[1, 1, 1, h, w])
    return mask


def make_beta_schedule(schedule, n_timestep, linear_start=0.0001, linear_end=0.02, cosine_s=0.008):
    if schedule == "linear":
        betas = (
            paddle.linspace(start=linear_start**0.5, stop=linear_end**0.5, num=n_timestep).astype("float64") ** 2
        )
    elif schedule == "cosine":
        timesteps = paddle.arange(end=n_timestep + 1).astype("float64") / n_timestep + cosine_s
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = paddle.cos(x=alphas).pow(y=2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule == "sqrt_linear":
        betas = paddle.linspace(start=linear_start, stop=linear_end, num=n_timestep).astype("float64")
    elif schedule == "sqrt":
        betas = paddle.linspace(start=linear_start, stop=linear_end, num=n_timestep).astype("float64") ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == "uniform":
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == "quad":
        ddim_timesteps = (np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f"Selected timesteps for ddim sampler: {steps_out}")
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f"Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}")
        print(
            f"For the chosen value of eta, which is {eta}, this results in the following sigma_t schedule for ddim sampler {sigmas}"
        )
    return sigmas, alphas, alphas_prev


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.take_along_axis(axis=-1, indices=t)
    return out.reshape([b, *((1,) * (len(x_shape) - 1))])


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = paddle.exp(
            x=(-math.log(max_period) * paddle.arange(start=0, end=half).astype("float32") / half).astype("float32")
        )
        args = timesteps[:, (None)].astype(dtype="float32") * freqs[None]
        embedding = paddle.concat(x=[paddle.cos(x=args), paddle.sin(x=args)], axis=-1)
        if dim % 2:
            embedding = paddle.concat(x=[embedding, paddle.zeros_like(x=embedding[:, :1])], axis=-1)
    else:
        embedding = repeat(timesteps, "b -> b d", d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().scale_(scale=scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(axis=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


def Normalize(in_channels):
    return paddle.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, epsilon=1e-06, weight_attr=None, bias_attr=None
    )


def identity(*args, **kwargs):
    return paddle.nn.Identity()


def nonlinearity(type="silu"):
    if type == "silu":
        return paddle.nn.Silu()
    elif type == "leaky_relu":
        return paddle.nn.LeakyReLU()


class GEGLU(paddle.nn.Layer):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = paddle.nn.Linear(in_features=dim_in, out_features=dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(chunks=2, axis=-1)
        return x * paddle.nn.functional.gelu(x=gate)


class SiLU(paddle.nn.Layer):
    def forward(self, x):
        return x * paddle.nn.functional.sigmoid(x=x)


class GroupNorm32(paddle.nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.astype(dtype="float32")).astype(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return paddle.nn.Conv1D(*args, **kwargs)
    elif dims == 2:
        return paddle.nn.Conv2D(*args, **kwargs)
    elif dims == 3:
        return paddle.nn.Conv3D(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return paddle.nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return paddle.nn.AvgPool1D(*args, **kwargs, exclusive=False)
    elif dims == 2:
        return paddle.nn.AvgPool1D(*args, **kwargs, exclusive=False)
    elif dims == 3:
        return paddle.nn.AvgPool1D(*args, **kwargs, exclusive=False)
    raise ValueError(f"unsupported dimensions: {dims}")


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: paddle.randn(shape=(1, *shape[1:])).tile(
        repeat_times=[shape[0], *((1,) * (len(shape) - 1))]
    )
    noise = lambda: paddle.randn(shape=shape)
    return repeat_noise() if repeat else noise()


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(min=-std, max=std)
    return tensor


def exists(val):
    return val is not None


def uniq(arr):
    return {el: (True) for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
