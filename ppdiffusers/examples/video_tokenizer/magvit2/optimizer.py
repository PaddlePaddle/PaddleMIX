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

import paddle


def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []

    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params


def get_optimizer(
    params,
    lr=0.0001,
    lr_scheduler=None,
    wd=0.01,
    betas=(0.9, 0.99),
    eps=1e-08,
    filter_by_requires_grad=False,
    group_wd_params=True,
    **kwargs
):
    max_grad_norm = kwargs.pop("max_grad_norm", None)

    if filter_by_requires_grad:
        params = [t for t in params if not t.stop_gradient]

    if wd == 0:
        return paddle.optimizer.Adam(
            parameters=params,
            learning_rate=lr_scheduler,
            beta1=betas[0],
            beta2=betas[1],
            epsilon=eps,
            grad_clip=paddle.nn.ClipGradByGlobalNorm(max_grad_norm) if max_grad_norm is not None else None,
        )

    if not group_wd_params:
        return paddle.optimizer.AdamW(
            parameters=params,
            learning_rate=lr_scheduler,
            beta1=betas[0],
            beta2=betas[1],
            epsilon=eps,
            weight_decay=wd,
            grad_clip=paddle.nn.ClipGradByGlobalNorm(max_grad_norm) if max_grad_norm is not None else None,
        )
    wd_params, no_wd_params = separate_weight_decayable_params(params)

    params = [{"params": wd_params, "weight_decay": wd}, {"params": no_wd_params, "weight_decay": 0.0}]

    return paddle.optimizer.AdamW(
        parameters=params,
        learning_rate=lr_scheduler,
        beta1=betas[0],
        beta2=betas[1],
        epsilon=eps,
        weight_decay=wd,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(max_grad_norm) if max_grad_norm is not None else None,
    )
