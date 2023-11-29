# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..initializer import normal_, zeros_
from ..loaders import (
    PatchedLoraProjection,
    text_encoder_attn_modules,
    text_encoder_mlp_modules,
)
from ..utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def adjust_lora_scale_text_encoder(text_encoder, lora_scale: float = 1.0):
    for _, attn_module in text_encoder_attn_modules(text_encoder):
        if isinstance(attn_module.q_proj, PatchedLoraProjection):
            attn_module.q_proj.lora_scale = lora_scale
            attn_module.k_proj.lora_scale = lora_scale
            attn_module.v_proj.lora_scale = lora_scale
            attn_module.out_proj.lora_scale = lora_scale

    for _, mlp_module in text_encoder_mlp_modules(text_encoder):
        if isinstance(mlp_module.linear1, PatchedLoraProjection):
            mlp_module.linear1.lora_scale = lora_scale
            mlp_module.linear2.lora_scale = lora_scale


class LoRALinearLayer(nn.Layer):
    r"""
    A linear layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        device (`torch.device`, `optional`, defaults to `None`):
            The device to use for the layer's weights.
        dtype (`torch.dtype`, `optional`, defaults to `None`):
            The dtype to use for the layer's weights.
    """

    def __init__(self, in_features, out_features, rank=4, network_alpha=None, device=None, dtype=None):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias_attr=False)
        self.up = nn.Linear(rank, out_features, bias_attr=False)
        if device is not None:
            self.down.to(device=device)
            self.up.to(device=device)
        if dtype is not None:
            self.down.to(dtype=dtype)
            self.up.to(dtype=dtype)

        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        normal_(self.down.weight, std=1 / rank)
        zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.cast(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.cast(orig_dtype)


class LoRAConv2dLayer(nn.Layer):
    r"""
    A convolutional layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        kernel_size (`int` or `tuple` of two `int`, `optional`, defaults to 1):
            The kernel size of the convolution.
        stride (`int` or `tuple` of two `int`, `optional`, defaults to 1):
            The stride of the convolution.
        padding (`int` or `tuple` of two `int` or `str`, `optional`, defaults to 0):
            The padding of the convolution.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
    """

    def __init__(
        self, in_features, out_features, rank=4, kernel_size=(1, 1), stride=(1, 1), padding=0, network_alpha=None
    ):
        super().__init__()

        self.down = nn.Conv2D(
            in_features, rank, kernel_size=kernel_size, stride=stride, padding=padding, bias_attr=False
        )
        # according to the official kohya_ss trainer kernel_size are always fixed for the up layer
        # # see: https://github.com/bmaltais/kohya_ss/blob/2accb1305979ba62f5077a23aabac23b4c37e935/networks/lora_diffusers.py#L129
        self.up = nn.Conv2D(rank, out_features, kernel_size=(1, 1), stride=(1, 1), bias_attr=False)

        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        normal_(self.down.weight, std=1 / rank)
        zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.cast(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.cast(orig_dtype)


class LoRACompatibleConv(nn.Conv2D):
    """
    A convolutional layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRAConv2dLayer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer: Optional[LoRAConv2dLayer]):
        self.lora_layer = lora_layer

    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        if self.lora_layer is None:
            return

        dtype = self.weight.dtype

        w_orig = self.weight.astype(paddle.get_default_dtype())
        w_up = self.lora_layer.up.weight.astype(paddle.get_default_dtype())
        w_down = self.lora_layer.down.weight.astype(paddle.get_default_dtype())

        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        fusion = paddle.mm(w_up.flatten(start_axis=1), w_down.flatten(start_axis=1))
        fusion = fusion.reshape((w_orig.shape))
        fused_weight = w_orig + (lora_scale * fusion)

        if safe_fusing and paddle.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        out_0 = fused_weight.cast(dtype=dtype)
        self.weight = self.create_parameter(
            shape=out_0.shape,
            default_initializer=nn.initializer.Assign(out_0),
        )

        # we can drop the lora layer now
        self.lora_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self._lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return

        fused_weight = self.weight
        dtype = fused_weight.dtype

        self.w_up = self.w_up.astype(paddle.get_default_dtype())
        self.w_down = self.w_down.astype(paddle.get_default_dtype())

        fusion = paddle.mm(self.w_up.flatten(start_axis=1), self.w_down.flatten(start_axis=1))
        fusion = fusion.reshape((fused_weight.shape))
        unfused_weight = fused_weight.astype(paddle.get_default_dtype()) - (self._lora_scale * fusion)
        out_0 = unfused_weight.cast(dtype=dtype)
        self.weight = self.create_parameter(
            shape=out_0.shape,
            default_initializer=nn.initializer.Assign(out_0),
        )

        self.w_up = None
        self.w_down = None

    def forward(self, hidden_states, scale: float = 1.0):
        if self.lora_layer is None:
            # make sure to the functional Conv2D function as otherwise torch.compile's graph will break
            # see: https://github.com/huggingface/diffusers/pull/4315
            return F.conv2d(
                hidden_states, self.weight, self.bias, self._stride, self._padding, self._dilation, self._groups
            )
        else:
            original_outputs = F.conv2d(
                hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
            return original_outputs + (scale * self.lora_layer(hidden_states))


class LoRACompatibleLinear(nn.Linear):
    """
    A Linear layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRALinearLayer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

    def set_lora_layer(self, lora_layer: Optional[LoRALinearLayer]):
        self.lora_layer = lora_layer

    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        if self.lora_layer is None:
            return

        dtype = self.weight.dtype

        w_orig = self.weight.astype(paddle.get_default_dtype())
        w_up = self.lora_layer.up.weight.astype(paddle.get_default_dtype())
        w_down = self.lora_layer.down.weight.astype(paddle.get_default_dtype())

        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        fused_weight = w_orig + (lora_scale * paddle.bmm(w_up.T[None, :], w_down.T[None, :])[0]).T

        if safe_fusing and paddle.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        out_0 = fused_weight.cast(dtype=dtype)
        self.weight = self.create_parameter(
            shape=out_0.shape,
            default_initializer=nn.initializer.Assign(out_0),
        )

        # we can drop the lora layer now
        self.lora_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self._lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return

        fused_weight = self.weight
        dtype = fused_weight.dtype

        w_up = self.w_up.astype(paddle.get_default_dtype())
        w_down = self.w_down.astype(paddle.get_default_dtype())

        unfused_weight = (
            fused_weight.astype(paddle.get_default_dtype())
            - (self._lora_scale * paddle.bmm(w_up.T[None, :], w_down.T[None, :])[0]).T
        )
        out_0 = unfused_weight.cast(dtype=dtype)
        self.weight = self.create_parameter(
            shape=out_0.shape,
            default_initializer=nn.initializer.Assign(out_0),
        )

        self.w_up = None
        self.w_down = None

    def forward(self, hidden_states, scale: float = 1.0):
        # breakpoint()
        if self.lora_layer is None:
            # return super().forward(hidden_states)
            return nn.functional.linear(
                hidden_states,
                self.weight,
                self.bias,
            )
        else:
            return super().forward(hidden_states) + (scale * self.lora_layer(hidden_states))
