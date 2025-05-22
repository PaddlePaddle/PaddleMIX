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

import math
import sys

import paddle

from paddlemix.models.diffsinger.utils import paddle_aux


class PositionalEncoding(paddle.nn.Layer):
    """Positional encoding.
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = paddle.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(paddle.to_tensor(data=0.0).expand(shape=[1, max_len]))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.shape[1] >= x.shape[1]:
                if self.pe.dtype != x.dtype or self.pe.place != x.place:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.place)
                return
        if self.reverse:
            position = paddle.arange(start=x.shape[1] - 1, end=-1, step=-1.0, dtype="float32").unsqueeze(axis=1)
        else:
            position = paddle.arange(start=0, end=x.shape[1], dtype="float32").unsqueeze(axis=1)
        div_term = paddle.exp(
            x=paddle.arange(start=0, end=self.d_model, step=2, dtype="float32") * -(math.log(10000.0) / self.d_model)
        )
        pe = (
            paddle.stack(x=[paddle.sin(x=position * div_term), paddle.cos(x=position * div_term)], axis=2)
            .view(-1, self.d_model)
            .unsqueeze(axis=0)
        )
        self.pe = pe.to(device=x.place, dtype=x.dtype)

    def forward(self, x: paddle.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.shape[1]]
        return self.dropout(x)


class ScaledPositionalEncoding(PositionalEncoding):
    """Scaled positional encoding module.
    See Sec. 3.2  https://arxiv.org/abs/1809.08895
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(d_model=d_model, dropout_rate=dropout_rate, max_len=max_len)
        self.alpha = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.to_tensor(data=1.0))

    def reset_parameters(self):
        """Reset parameters."""
        self.alpha.data = paddle.to_tensor(data=1.0)

    def forward(self, x):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x + self.alpha * self.pe[:, : x.shape[1]]
        return self.dropout(x)


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class."""
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self, x):
        """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[:, : x.shape[1]]
        return self.dropout(x) + self.dropout(pos_emb)
