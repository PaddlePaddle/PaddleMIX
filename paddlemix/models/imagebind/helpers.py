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

from types import SimpleNamespace

import einops
import numpy as np
import paddle

ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
)


class Normalize(paddle.nn.Layer):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return paddle.nn.functional.normalize(x=x, axis=self.dim, p=2)


class LearnableLogitScaling(paddle.nn.Layer):
    def __init__(
        self,
        logit_scale_init: float = 1 / 0.07,
        learnable: bool = True,
        max_logit_scale: float = 100,
    ) -> None:
        super().__init__()
        self.max_logit_scale = max_logit_scale
        self.logit_scale_init = logit_scale_init
        self.learnable = learnable
        log_logit_scale = paddle.ones(shape=[]) * np.log(self.logit_scale_init)
        if learnable:
            self.log_logit_scale = paddle.create_parameter(
                shape=log_logit_scale.shape if log_logit_scale.dim() != 0 else [1],
                dtype=log_logit_scale.dtype,
                default_initializer=paddle.nn.initializer.Assign(value=log_logit_scale),
            )
        else:
            self.register_buffer("log_logit_scale", log_logit_scale)

    def forward(self, x):
        return paddle.clip(x=self.log_logit_scale.exp(), max=self.max_logit_scale).unsqueeze(0) * x


class EinOpsRearrange(paddle.nn.Layer):
    def __init__(self, rearrange_expr: str, **kwargs) -> None:
        super().__init__()
        self.rearrange_expr = rearrange_expr
        self.kwargs = kwargs

    def forward(self, x):
        assert isinstance(x, paddle.Tensor)
        return einops.rearrange(x, self.rearrange_expr, **self.kwargs)


class VerboseNNModule(paddle.nn.Layer):
    """
    Wrapper around nn.Module that prints registered buffers and parameter names.
    """

    @staticmethod
    def get_readable_tensor_repr(name: str, tensor: paddle.Tensor) -> str:
        st = (
            "("
            + name
            + "): "
            + "tensor("
            + str(tuple(tensor[1].shape))
            + ", requires_grad="
            + str(not tensor[1].stop_gradient)
            + ")\n"
        )
        return st


def cast_if_src_dtype(tensor: paddle.Tensor, src_dtype: paddle.dtype, tgt_dtype: paddle.dtype):
    updated = False
    if tensor.dtype == src_dtype:
        tensor = tensor.cast(tgt_dtype)
        updated = True
    return tensor, updated


class QuickGELU(paddle.nn.Layer):
    def forward(self, x: paddle.Tensor):
        return x * paddle.nn.functional.sigmoid(x=1.702 * x)


class SelectElement(paddle.nn.Layer):
    def __init__(self, index) -> None:
        super().__init__()
        self.index = index

    def forward(self, x):
        assert x.ndim >= 3
        return x[:, (self.index), (...)]


class SelectEOSAndProject(paddle.nn.Layer):
    """
    Text Pooling used in OpenCLIP
    """

    def __init__(self, proj: paddle.nn.Layer) -> None:
        super().__init__()
        self.proj = proj

    def forward(self, x, seq_len):
        assert x.ndim == 3
        x = x[paddle.arange(end=x.shape[0]).astype("int64"), seq_len]
        x = self.proj(x[None, :])
        return x
