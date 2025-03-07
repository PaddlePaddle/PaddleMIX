# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
import paddle.nn.functional as F
from paddle import nn

from ..utils import USE_PEFT_BACKEND
from .lora import LoRACompatibleLinear

ACTIVATION_FUNCTIONS = {
    "swish": nn.Silu(),
    "silu": nn.Silu(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}


def get_activation(act_fn: str) -> nn.Layer:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Layer: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")

class FP32SiLU(nn.Layer):
    r"""
    SiLU activation function with input upcasted to paddle.float32.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        return F.silu(inputs.astype(paddle.float32)).astype(inputs.dtype)

class GELU(nn.Layer):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool=True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias_attr=bias)
        self.approximate = approximate

    def gelu(self, gate: paddle.Tensor) -> paddle.Tensor:
        return F.gelu(gate, approximate=self.approximate != "none")

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class GEGLU(nn.Layer):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool=True):
        super().__init__()
        linear_cls = LoRACompatibleLinear if not USE_PEFT_BACKEND else nn.Linear

        self.proj = linear_cls(dim_in, dim_out * 2, bias_attr=bias)

    def gelu(self, gate: paddle.Tensor) -> paddle.Tensor:
        return F.gelu(gate)

    def forward(self, hidden_states, scale: float = 1.0):
        args = () if USE_PEFT_BACKEND else (scale,)
        hidden_states, gate = self.proj(hidden_states, *args).chunk(2, axis=-1)
        return hidden_states * self.gelu(gate)


class ApproximateGELU(nn.Layer):
    r"""
    The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
    [paper](https://arxiv.org/abs/1606.08415).

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool=True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias_attr=bias)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.proj(x)
        return x * F.sigmoid(1.702 * x)

class LinearActivation(paddle.nn.Layer):
    def __init__(
        self, dim_in: int, dim_out: int, bias: bool = True, activation: str = "silu"
    ):
        super().__init__()
        self.proj = paddle.nn.Linear(in_features=dim_in, out_features=dim_out, bias_attr=bias)
        self.activation = get_activation(activation)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        return self.activation(hidden_states)