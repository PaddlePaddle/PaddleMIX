# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn as nn


class AttentionPool(nn.Layer):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        out_0 = paddle.create_parameter(
            shape=(paddle.randn(shape=[spacial_dim + 1, embed_dim]) / embed_dim**0.5).shape,
            dtype=(paddle.randn(shape=[spacial_dim + 1, embed_dim]) / embed_dim**0.5).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.randn(shape=[spacial_dim + 1, embed_dim]) / embed_dim**0.5
            ),
        )
        out_0.stop_gradient = not True
        self.positional_embedding = out_0

        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.output_dim = output_dim or embed_dim

        # Initialize MultiHeadAttention and Linear layers
        self.attention = nn.MultiHeadAttention(embed_dim, num_heads)
        self.c_proj = nn.Linear(embed_dim, self.output_dim)

        # Initialize q_proj, k_proj, v_proj for loading pretrained weights
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Load pretrained weights into q_proj, k_proj, v_proj
        self._initialize_attention_weights()

    def _initialize_attention_weights(self):
        with paddle.no_grad():
            self.attention.q_proj.weight = self.q_proj.weight
            self.attention.k_proj.weight = self.k_proj.weight
            self.attention.v_proj.weight = self.v_proj.weight
            self.attention.q_proj.bias = self.q_proj.bias
            self.attention.k_proj.bias = self.k_proj.bias
            self.attention.v_proj.bias = self.v_proj.bias

            self.attention.out_proj.weight = self.c_proj.weight
            self.attention.out_proj.bias = self.c_proj.bias

    def forward(self, x):
        x = paddle.concat(x=[x.mean(axis=1, keepdim=True), x], axis=1)
        x = x + self.positional_embedding[(None), :, :].to(dtype=x.dtype)
        query = x[:, :1]
        key = x
        value = x
        attn_output = self.attention(query, key, value)
        # attn_output = self.c_proj(attn_output)
        return attn_output.squeeze(axis=1)
