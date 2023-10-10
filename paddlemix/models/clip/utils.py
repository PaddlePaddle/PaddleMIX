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

import collections
from itertools import repeat

import paddle


def is_model_parrallel():
    if paddle.distributed.get_world_size() > 1:
        hcg = paddle.distributed.fleet.get_hybrid_communicate_group()
        if hcg.get_model_parallel_world_size() > 1:
            return True
        else:
            return False
    else:
        return False


def params_normal_(tensor, mean=0.0, std=1.0):
    origin_dtype = paddle.get_default_dtype()
    paddle.set_default_dtype("float32")
    with paddle.no_grad():
        normal = paddle.normal(mean=mean, std=std, shape=tensor.shape)
        if origin_dtype != "float32":
            normal = normal.astype(origin_dtype)
        tensor.set_value(normal)
    paddle.set_default_dtype(origin_dtype)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, min=-2, max=2):
    origin_dtype = paddle.get_default_dtype()
    paddle.set_default_dtype("float32")
    with paddle.no_grad():
        normal = paddle.normal(mean=mean, std=std, shape=tensor.shape)
        trunc = paddle.clip(normal, min=min, max=max)
        if origin_dtype != "float32":
            trunc = trunc.astype(origin_dtype)
        tensor.set_value(trunc)
    paddle.set_default_dtype(origin_dtype)
    return tensor


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite: bool = False, need_grad_norm: bool = False):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        need_grad_norm (bool): if True, total norm clipped will be return and it is
            only used for tensorboard. Default: False.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    paddle_dtype = paddle.get_default_dtype()
    if len(grads) == 0:
        return paddle.to_tensor([0.0])
    if norm_type == float("inf"):
        norms = [g.detach().abs().max() for g in grads]
        total_norm = norms[0] if len(norms) == 1 else paddle.max(paddle.stack(norms))
    else:
        total_norm = paddle.norm(
            paddle.stack(
                [
                    paddle.norm(g.detach(), norm_type)
                    if g.dtype == paddle_dtype
                    else paddle.norm(g.detach().cast(paddle_dtype), norm_type)
                    for g in grads
                ]
            ),
            norm_type,
        )
    if error_if_nonfinite and paddle.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = paddle.clip(clip_coef, max=1.0)
    clip_coef_clamped_low_precison = None
    for g in grads:
        if g.dtype == paddle.float32:
            g.detach().multiply_(clip_coef_clamped)
        else:
            clip_coef_clamped_low_precison = (
                clip_coef_clamped.cast(g.dtype)
                if clip_coef_clamped_low_precison is None
                else clip_coef_clamped_low_precison
            )
            g.detach().multiply_(clip_coef_clamped_low_precison)
    if need_grad_norm:
        total_norm_clip = paddle.norm(
            paddle.stack(
                [
                    paddle.norm(g.detach(), norm_type)
                    if g.dtype == paddle_dtype
                    else paddle.norm(g.detach().cast(paddle_dtype), norm_type)
                    for g in grads
                ]
            ),
            norm_type,
        )
        return total_norm_clip
    return total_norm


def clip_grad_norm(
    parameters, max_norm, norm_type=2.0, error_if_nonfinite: bool = False, need_grad_norm: bool = False
):
    return clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite, need_grad_norm)
