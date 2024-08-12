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


# IMPORTANT:                                                      #
###################################################################
# ----------------------------------------------------------------#
# This file is deprecated and will be removed soon                #
# (as soon as PEFT will become a required dependency for LoRA)    #
# ----------------------------------------------------------------#
###################################################################

import contextlib
from typing import Optional, Tuple, Union

import paddle
from paddle import nn

from ppdiffusers.transformers import CLIPTextModel, CLIPTextModelWithProjection

from ..utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def text_encoder_attn_modules(text_encoder):
    attn_modules = []

    if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
        for i, layer in enumerate(text_encoder.text_model.encoder.layers):
            name = f"text_model.encoder.layers.{i}.self_attn"
            mod = layer.self_attn
            attn_modules.append((name, mod))
    else:
        raise ValueError(f"do not know how to get attention modules for: {text_encoder.__class__.__name__}")

    return attn_modules


def text_encoder_mlp_modules(text_encoder):
    mlp_modules = []

    if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
        for i, layer in enumerate(text_encoder.text_model.encoder.layers):
            mlp_mod = layer.mlp
            name = f"text_model.encoder.layers.{i}.mlp"
            mlp_modules.append((name, mlp_mod))
    else:
        raise ValueError(f"do not know how to get mlp modules for: {text_encoder.__class__.__name__}")

    return mlp_modules


def adjust_lora_scale_text_encoder(text_encoder, lora_scale: float = 1.0):
    for _, attn_module in text_encoder_attn_modules(text_encoder):
        if isinstance(attn_module.q_proj, PatchedLoraProjection):
            attn_module.q_proj.lora_scale = lora_scale
            attn_module.k_proj.lora_scale = lora_scale
            attn_module.v_proj.lora_scale = lora_scale
            attn_module.out_proj.lora_scale = lora_scale

    for _, mlp_module in text_encoder_mlp_modules(text_encoder):
        if isinstance(mlp_module.fc1, PatchedLoraProjection):
            mlp_module.fc1.lora_scale = lora_scale
            mlp_module.fc2.lora_scale = lora_scale


class PatchedLoraProjection(nn.Layer):
    def __init__(self, regular_linear_layer, lora_scale=1, network_alpha=None, rank=4, dtype=None):
        super().__init__()
        from ..models.lora import LoRALinearLayer

        self.regular_linear_layer = regular_linear_layer

        if dtype is None:
            dtype = self.regular_linear_layer.weight.dtype

        self.lora_linear_layer = LoRALinearLayer(
            self.regular_linear_layer.in_features,
            self.regular_linear_layer.out_features,
            network_alpha=network_alpha,
            dtype=dtype,
            rank=rank,
        )

        self.lora_scale = lora_scale

    # overwrite PyTorch's `state_dict` to be sure that only the 'regular_linear_layer' weights are saved
    # when saving the whole text encoder model and when LoRA is unloaded or fused
    def state_dict(self, destination=None, include_sublayers=True, structured_name_prefix="", use_hook=True):
        if self.lora_linear_layer is None:
            return self.regular_linear_layer.state_dict(
                destination=destination,
                include_sublayers=include_sublayers,
                structured_name_prefix=structured_name_prefix,
                use_hook=use_hook,
            )

        return super().state_dict(
            destination=destination,
            include_sublayers=include_sublayers,
            structured_name_prefix=structured_name_prefix,
            use_hook=use_hook,
        )

    def _fuse_lora(self, lora_scale=1.0, safe_fusing=False):
        if self.lora_linear_layer is None:
            return

        dtype = self.regular_linear_layer.weight.dtype

        w_orig = self.regular_linear_layer.weight.cast("float32")
        w_up = self.lora_linear_layer.up.weight.cast("float32")
        w_down = self.lora_linear_layer.down.weight.cast("float32")

        if self.lora_linear_layer.network_alpha is not None:
            w_up = w_up * self.lora_linear_layer.network_alpha / self.lora_linear_layer.rank

        fused_weight = w_orig + (lora_scale * paddle.matmul(w_down[None, :], w_up[None, :])[0])

        if safe_fusing and paddle.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        self.regular_linear_layer.weight.copy_(fused_weight.cast(dtype=dtype), False)

        # we can drop the lora layer now
        self.lora_linear_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self.lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return

        fused_weight = self.regular_linear_layer.weight
        dtype = fused_weight.dtype

        w_up = self.w_up.cast("float32")
        w_down = self.w_down.cast("float32")

        unfused_weight = fused_weight.cast("float32") - (
            self.lora_scale * paddle.matmul(w_down[None, :], w_up[None, :])[0]
        )
        self.regular_linear_layer.weight.copy_(unfused_weight.cast(dtype=dtype), False)

        self.w_up = None
        self.w_down = None

    def forward(self, input):
        if self.lora_scale is None:
            self.lora_scale = 1.0
        if self.lora_linear_layer is None:
            return self.regular_linear_layer(input)
        return self.regular_linear_layer(input) + (self.lora_scale * self.lora_linear_layer(input))


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
        dtype (`torch.dtype`, `optional`, defaults to `None`):
            The dtype to use for the layer's weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        dtype: Optional[paddle.dtype] = None,
    ):
        super().__init__()
        if dtype is not None:
            ctx = paddle.dtype_guard(dtype)
        else:
            ctx = contextlib.nullcontext()
        with ctx:
            self.down = nn.Linear(in_features, rank, bias_attr=False)
            self.up = nn.Linear(rank, out_features, bias_attr=False)

        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
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
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        kernel_size: Union[int, Tuple[int, int]] = (1, 1),
        stride: Union[int, Tuple[int, int]] = (1, 1),
        padding: Union[int, Tuple[int, int], str] = 0,
        network_alpha: Optional[float] = None,
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

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
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
        self.in_channels = self._in_channels
        self.out_channels = self._out_channels
        self.kernel_size = self._kernel_size
        self.lora_layer = lora_layer
        self.data_format = kwargs.get("data_format", "NCHW")

    def set_lora_layer(self, lora_layer: Optional[LoRAConv2dLayer]):
        self.lora_layer = lora_layer

    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        if self.lora_layer is None:
            return

        dtype = self.weight.dtype

        w_orig = self.weight.cast("float32")
        w_up = self.lora_layer.up.weight.cast("float32")
        w_down = self.lora_layer.down.weight.cast("float32")

        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        fusion = paddle.matmul(w_up.flatten(start_axis=1), w_down.flatten(start_axis=1))
        fusion = fusion.reshape(w_orig.shape)
        fused_weight = w_orig + (lora_scale * fusion)

        if safe_fusing and paddle.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        self.weight.copy_(fused_weight.cast(dtype=dtype), False)

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

        w_up = self.w_up.cast("float32")
        w_down = self.w_down.cast("float32")

        fusion = paddle.matmul(w_up.flatten(start_axis=1), w_down.flatten(start_axis=1))
        fusion = fusion.reshape(fused_weight.shape)
        unfused_weight = fused_weight.cast("float32") - (self._lora_scale * fusion)
        self.weight.copy_(unfused_weight.cast(dtype=dtype), False)

        self.w_up = None
        self.w_down = None

    def forward(self, hidden_states: paddle.Tensor, scale: float = 1.0) -> paddle.Tensor:
        if self.lora_layer is None:
            # make sure to the functional Conv2D function as otherwise torch.compile's graph will break
            # see: https://github.com/huggingface/diffusers/pull/4315
            return nn.functional.conv2d(
                hidden_states,
                self.weight,
                self.bias,
                self._stride,
                self._padding,
                self._dilation,
                self._groups,
                data_format=self.data_format,
            )
        else:
            original_outputs = nn.functional.conv2d(
                hidden_states,
                self.weight,
                self.bias,
                self._stride,
                self._padding,
                self._dilation,
                self._groups,
                data_format=self.data_format,
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

        w_orig = self.weight.cast("float32")
        w_up = self.lora_layer.up.weight.cast("float32")
        w_down = self.lora_layer.down.weight.cast("float32")

        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        fused_weight = w_orig + (lora_scale * paddle.matmul(w_down[None, :], w_up[None, :])[0])

        if safe_fusing and paddle.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )
        self.weight.copy_(fused_weight.cast(dtype=dtype), False)

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

        w_up = self.w_up.cast("float32")
        w_down = self.w_down.cast("float32")

        unfused_weight = fused_weight.cast("float32") - (
            self._lora_scale * paddle.matmul(w_down[None, :], w_up[None, :])[0]
        )
        self.weight.copy_(unfused_weight.cast(dtype=dtype), False)

        self.w_up = None
        self.w_down = None

    def forward(self, hidden_states: paddle.Tensor, scale: float = 1.0) -> paddle.Tensor:
        if self.lora_layer is None:
            return nn.functional.linear(
                hidden_states,
                self.weight,
                self.bias,
            )
        else:
            out = super().forward(hidden_states) + (scale * self.lora_layer(hidden_states))
            return out
