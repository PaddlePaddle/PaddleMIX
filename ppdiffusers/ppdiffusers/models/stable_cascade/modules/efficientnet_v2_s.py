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

import copy
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import paddle
import paddle.nn as nn
from paddle import Tensor
from paddle.nn import (
    AdaptiveAvgPool2D,
    BatchNorm,
    BatchNorm2D,
    Conv2D,
    Dropout,
    GroupNorm,
    Layer,
    Linear,
    ReLU,
    Sequential,
    Sigmoid,
    Silu,
)
from paddle.nn.initializer import Constant, KaimingNormal, Uniform
from paddle.utils.download import get_weights_path_from_url

__all__ = ["EfficientNet", "EfficientNet_V2_S_Weights", "efficientnet_v2_s"]


class SqueezeExcitation(paddle.nn.Layer):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation`` and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input feature maps
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[[Tensor], Tensor], optional): ``delta`` activation. Default: ReLU
        scale_activation (Callable[[Tensor], Tensor], optional): ``sigma`` activation. Default: Sigmoid
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[[Tensor], Tensor] = ReLU(),
        scale_activation: Callable[[Tensor], Tensor] = Sigmoid(),
    ) -> None:
        super(SqueezeExcitation, self).__init__()
        self.avgpool = AdaptiveAvgPool2D(1)
        self.fc1 = Conv2D(in_channels=input_channels, out_channels=squeeze_channels, kernel_size=1)
        self.fc2 = Conv2D(in_channels=squeeze_channels, out_channels=input_channels, kernel_size=1)
        self.activation = activation
        self.scale_activation = scale_activation

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return scale * input


def stochastic_depth(input, p, mode, training=True):
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_  used for randomly dropping residual
    branches of residual architectures.

    Args:
        input (paddle.Tensor): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training (bool): apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        paddle.Tensor: The randomly zeroed tensor.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = paddle.empty(size, dtype=input.dtype)
    survival_rate = paddle.to_tensor(survival_rate, dtype=input.dtype)
    paddle.assign(paddle.bernoulli(paddle.broadcast_to(survival_rate, noise.shape)), noise)
    if survival_rate > 0.0:
        noise /= survival_rate
    return input * noise


class StochasticDepth(Layer):
    """
    See :func:`stochastic_depth`.
    """

    def __init__(self, p: float, mode: str) -> None:
        super(StochasticDepth, self).__init__()
        self.p = p
        self.mode = mode

    def forward(self, input):
        return stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self):
        s = f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"
        return s


def _make_ntuple(value, n):
    """Helper function to create a tuple of size n with the given value."""
    if isinstance(value, int):
        return (value,) * n
    return value


class ConvNormActivation(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 1,
        padding: Optional[Union[int, Sequence[int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., paddle.nn.Layer]] = BatchNorm,
        activation_layer: Optional[Callable[..., paddle.nn.Layer]] = ReLU,
        dilation: Union[int, Sequence[int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., Conv2D] = Conv2D,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        else:
            padding = _make_ntuple(padding, len(kernel_size))

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias_attr=False if bias is None else bias,
            )
        ]

        if norm_layer is not None:
            norm_layer_instance = norm_layer(out_channels, use_global_stats=True)
            layers.append(norm_layer_instance)

        if activation_layer is not None:
            layers.append(activation_layer)

        super(ConvNormActivation, self).__init__(*layers)
        self.out_channels = out_channels


class Conv2DNormActivation(ConvNormActivation):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., paddle.nn.Layer]] = BatchNorm,
        activation_layer: Optional[Callable[..., paddle.nn.Layer]] = ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            Conv2D,
        )


class EfficientNet_V2_S_Weights:
    IMAGENET1K_V1 = "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth"

    def __init__(self, url: str, transforms: Callable[..., Any], meta: Dict[str, Any]) -> None:
        self.url = url
        self.transforms = transforms
        self.meta = meta

    def state_dict(self, progress: bool = True, check_hash: bool = False) -> Dict[str, Any]:
        path = get_weights_path_from_url(self.url, progress=progress, check_hash=check_hash)
        return paddle.load(path)

    @classmethod
    def verify(cls, weights):
        if weights is None:
            return None
        if not isinstance(weights, EfficientNet_V2_S_Weights):
            raise ValueError(f"weights must be an instance of EfficientNet_V2_S_Weights, but got {type(weights)}")
        return weights


@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., paddle.nn.Layer]

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)


class MBConvConfig(_MBConvConfig):
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Optional[Callable[..., paddle.nn.Layer]] = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult)
        out_channels = self.adjust_channels(out_channels, width_mult)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = MBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class FusedMBConvConfig(_MBConvConfig):
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., paddle.nn.Layer]] = None,
    ) -> None:
        if block is None:
            block = FusedMBConv
        super().__init__(expand_ratio, kernel, stride, input_channels, out_channels, num_layers, block)


class MBConv(Layer):
    def __init__(
        self,
        cnf,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., Layer],
        se_layer: Callable[..., Layer] = SqueezeExcitation,
    ) -> None:
        super(MBConv, self).__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers = []
        activation_layer = nn.Silu()

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                Conv2DNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2DNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=nn.Silu()))

        # project
        layers.append(
            Conv2DNormActivation(
                expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input) -> paddle.Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class FusedMBConv(Layer):
    def __init__(
        self,
        cnf: "FusedMBConvConfig",
        stochastic_depth_prob: float,
        norm_layer: Callable[..., Layer],
    ) -> None:
        super(FusedMBConv, self).__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[Layer] = []
        activation_layer = nn.Silu()

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            # fused expand and project
            layers.append(
                Conv2DNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )
            # project
            layers.append(
                Conv2DNormActivation(
                    expanded_channels,
                    cnf.out_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=None,
                )
            )
        else:
            layers.append(
                Conv2DNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet(Layer):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., paddle.nn.Layer]] = None,
        last_channel: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, _MBConvConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")
        if norm_layer is None:
            norm_layer = BatchNorm2D
        layers: List[paddle.nn.Layer] = []
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2DNormActivation(
                3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=Silu()
            )
        )
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[paddle.nn.Layer] = []
            for _ in range(cnf.num_layers):
                block_cnf = copy.copy(cnf)
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1
            layers.append(Sequential(*stage))
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = last_channel if last_channel is not None else 4 * lastconv_input_channels
        layers.append(
            Conv2DNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=Silu(),
            )
        )
        self.features = Sequential(*layers)
        self.avgpool = AdaptiveAvgPool2D(output_size=1)
        self.classifier = Sequential(
            Dropout(p=dropout), Linear(in_features=lastconv_output_channels, out_features=num_classes)
        )

        for m in self.sublayers():
            if isinstance(m, Conv2D):
                KaimingNormal()(m.weight)
                if m.bias is not None:
                    Constant(value=0.0)(m.bias)
            elif isinstance(m, (BatchNorm2D, GroupNorm)):
                Constant(value=1.0)(m.weight)
                Constant(value=0.0)(m.bias)
            elif isinstance(m, Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                Uniform(low=-init_range, high=init_range)(m.weight)
                Constant(value=0.0)(m.bias)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = paddle.flatten(x=x, start_axis=1)
        x = self.classifier(x)
        return x


def _make_divisible(value: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def _efficientnet(
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
    dropout: float,
    last_channel: Optional[int],
    weights: Optional[EfficientNet_V2_S_Weights],
    progress: bool,
    **kwargs: Any
) -> EfficientNet:
    if weights is not None:
        kwargs["num_classes"] = len(weights.meta["categories"])
    model = EfficientNet(inverted_residual_setting, dropout, last_channel=last_channel, **kwargs)
    if weights is not None:
        model.set_state_dict(weights.state_dict(progress=progress, check_hash=True))
    return model


def _efficientnet_conf(
    arch: str, **kwargs: Any
) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]
    if arch.startswith("efficientnet_v2_s"):
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
    else:
        raise ValueError(f"Unsupported model type {arch}")
    return inverted_residual_setting, last_channel


def efficientnet_v2_s(
    *, weights: Optional[EfficientNet_V2_S_Weights] = None, progress: bool = True, **kwargs: Any
) -> EfficientNet:
    weights = EfficientNet_V2_S_Weights.verify(weights)
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
    return _efficientnet(
        inverted_residual_setting,
        kwargs.pop("dropout", 0.2),
        last_channel,
        weights,
        progress,
        norm_layer=partial(BatchNorm2D, epsilon=0.001),
        **kwargs,
    )
