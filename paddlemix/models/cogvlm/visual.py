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

from argparse import Namespace

import paddle
from paddle.incubate.nn.memory_efficient_attention import memory_efficient_attention
from paddlenlp.transformers.activations import ACT2FN


class PatchEmbedding(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.proj = paddle.nn.Conv2D(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        out_1 = paddle.create_parameter(
            shape=paddle.zeros(shape=[1, config.hidden_size]).shape,
            dtype=paddle.zeros(shape=[1, config.hidden_size]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=[1, config.hidden_size])),
        )
        out_1.stop_gradient = not True
        self.cls_embedding = out_1
        self.position_embedding = paddle.nn.Embedding(
            num_embeddings=config.num_positions, embedding_dim=config.hidden_size
        )

    def forward(self, images):
        x = self.proj(images)
        x = x.flatten(start_axis=2)
        perm_2 = list(range(x.ndim))
        perm_2[1] = 2
        perm_2[2] = 1
        x = x.transpose(perm=perm_2)
        cls_token = self.cls_embedding.expand(shape=[x.shape[0], -1, -1])
        x = paddle.concat(x=(cls_token, x), axis=1)
        x += self.position_embedding.weight.unsqueeze(axis=0)
        return x


class Attention(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        head_dim = config.hidden_size // config.num_heads
        self.scale = head_dim**-0.5
        self.query_key_value = paddle.nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size * 3)
        self.dense = paddle.nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.output_dropout = paddle.nn.Dropout(p=config.dropout_prob)

    def forward(self, x):
        B, L, _ = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.reshape([B, L, 3, self.num_heads, -1]).transpose(perm=[2, 0, 1, 3, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = memory_efficient_attention(q, k, v, scale=self.scale)
        output = self.dense(out.reshape([B, L, -1]))
        output = self.output_dropout(output)
        return output

    def attention(self, q, k, v):
        x = k
        perm_3 = list(range(x.ndim))
        perm_3[-2] = -1
        perm_3[-1] = -2
        attn_weights = paddle.matmul(x=q * self.scale, y=x.transpose(perm=perm_3))
        attn_weights = paddle.nn.functional.softmax(attn_weights, axis=-1)
        output = paddle.matmul(x=attn_weights, y=v)
        return output


class MLP(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = paddle.nn.Linear(in_features=config.hidden_size, out_features=config.intermediate_size)
        self.fc2 = paddle.nn.Linear(in_features=config.intermediate_size, out_features=config.hidden_size)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class TransformerLayer(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = paddle.nn.LayerNorm(normalized_shape=config.hidden_size, epsilon=config.layer_norm_eps)
        self.attention = Attention(config)
        self.mlp = MLP(config)
        self.post_attention_layernorm = paddle.nn.LayerNorm(
            normalized_shape=config.hidden_size, epsilon=config.layer_norm_eps
        )

    def forward(self, hidden_states):
        attention_input = hidden_states
        attention_output = self.input_layernorm(self.attention(attention_input))
        hidden_states = attention_input + attention_output
        mlp_input = hidden_states
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))
        output = mlp_input + mlp_output
        return output


class Transformer(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.layers = paddle.nn.LayerList(
            sublayers=[TransformerLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states):
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class GLU(paddle.nn.Layer):
    def __init__(self, config, in_features):
        super().__init__()
        self.linear_proj = paddle.nn.Linear(in_features=in_features, out_features=config.hidden_size, bias_attr=False)
        self.norm1 = paddle.nn.LayerNorm(normalized_shape=config.hidden_size)
        self.act1 = paddle.nn.GELU()
        self.act2 = paddle.nn.functional.silu
        self.dense_h_to_4h = paddle.nn.Linear(
            in_features=config.hidden_size, out_features=config.intermediate_size, bias_attr=False
        )
        self.gate_proj = paddle.nn.Linear(
            in_features=config.hidden_size, out_features=config.intermediate_size, bias_attr=False
        )
        self.dense_4h_to_h = paddle.nn.Linear(
            in_features=config.intermediate_size, out_features=config.hidden_size, bias_attr=False
        )

    def forward(self, x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)
        return x


class EVA2CLIPModel(paddle.nn.Layer):
    def __init__(self, config):
        super().__init__()
        vision_config = Namespace(**config.vision_config)
        self.patch_embedding = PatchEmbedding(vision_config)
        self.transformer = Transformer(vision_config)
        self.linear_proj = GLU(config, in_features=vision_config.hidden_size)
        out_2 = paddle.create_parameter(
            shape=paddle.zeros(shape=[1, 1, config.hidden_size]).shape,
            dtype=paddle.zeros(shape=[1, 1, config.hidden_size]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=[1, 1, config.hidden_size])),
        )
        out_2.stop_gradient = not True
        self.boi = out_2
        out_3 = paddle.create_parameter(
            shape=paddle.zeros(shape=[1, 1, config.hidden_size]).shape,
            dtype=paddle.zeros(shape=[1, 1, config.hidden_size]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=[1, 1, config.hidden_size])),
        )
        out_3.stop_gradient = not True
        self.eoi = out_3

    def forward(self, images):
        x = self.patch_embedding(images)
        x = self.transformer(x)
        x = x[:, 1:]
        x = self.linear_proj(x)
        boi = self.boi.expand(shape=[x.shape[0], -1, -1])
        eoi = self.eoi.expand(shape=[x.shape[0], -1, -1])
        x = paddle.concat(x=(boi, x, eoi), axis=1)
        return x
