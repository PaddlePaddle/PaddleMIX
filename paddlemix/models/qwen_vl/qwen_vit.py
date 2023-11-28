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

import math
import os
from functools import partial
from typing import Callable, Dict, Optional, Union

import paddle
from paddlenlp.transformers.configuration_utils import PretrainedConfig

from paddlemix.models.model_utils import MixPretrainedModel
from paddlemix.utils.log import logger


def get_abs_pos(abs_pos, tgt_size):
    src_size = int(math.sqrt(abs_pos.shape[0]))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype
    if src_size != tgt_size:
        return (
            paddle.nn.functional.interpolate(
                x=abs_pos.astype(dtype="float32").reshape([1, src_size, src_size, -1]).transpose(perm=[0, 3, 1, 2]),
                size=(tgt_size, tgt_size),
                mode="bicubic",
                align_corners=False,
            )
            .transpose(perm=[0, 2, 3, 1])
            .flatten(start_axis=0, stop_axis=2)
            .astype(dtype=dtype)
        )
    else:
        return abs_pos


class VisualAttention(paddle.nn.Layer):
    """self-attention layer class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, embed_dim, num_heads, bias=True, kdim=None, vdim=None):
        super(VisualAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0
        self.hidden_size_per_attention_head = embed_dim // num_heads
        self.num_attention_heads_per_partition = num_heads
        self.hidden_size_per_partition = embed_dim

        assert self._qkv_same_embed_dim, "Only Support SelfAttention Currently"
        self.in_proj = paddle.nn.Linear(in_features=embed_dim, out_features=3 * embed_dim)
        self.out_proj = paddle.nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)

    def forward(self, query, key, value, attn_mask=None):

        sq, b, _ = query.shape
        assert query is key, "Only Support Self-Attention Currently"
        sk = sq
        mixed_x_layer = self.in_proj(query)
        new_tensor_shape = mixed_x_layer.shape[:-1] + [
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        ]
        mixed_x_layer = mixed_x_layer.reshape(new_tensor_shape)

        query_layer, key_layer, value_layer = mixed_x_layer.split(
            new_tensor_shape[-1] // self.hidden_size_per_attention_head, axis=-1
        )

        x = query_layer.reshape([sq, b * self.num_attention_heads_per_partition, self.hidden_size_per_attention_head])
        perm_3 = list(range(x.ndim))
        perm_3[0] = 1
        perm_3[1] = 0
        query_layer = x.transpose(perm=perm_3)
        x = key_layer.reshape([sk, b * self.num_attention_heads_per_partition, self.hidden_size_per_attention_head])
        perm_4 = list(range(x.ndim))
        perm_4[0] = 1
        perm_4[1] = 0
        key_layer = x.transpose(perm=perm_4)
        q_scaled = query_layer / self.norm_factor
        if attn_mask is not None:
            x = key_layer
            perm_5 = list(range(x.ndim))
            perm_5[-2] = -1
            perm_5[-1] = -2
            attention_probs = paddle.add(1 * attn_mask, 1 * paddle.bmm(q_scaled, x.transpose(perm=perm_5)))
        else:
            x = key_layer
            perm_6 = list(range(x.ndim))
            perm_6[-2] = -1
            perm_6[-1] = -2
            attention_probs = paddle.bmm(x=q_scaled, y=x.transpose(perm=perm_6))
        attention_probs = paddle.nn.functional.softmax(attention_probs, axis=-1)

        x = value_layer.reshape([sk, b * self.num_attention_heads_per_partition, self.hidden_size_per_attention_head])
        perm_7 = list(range(x.ndim))
        perm_7[0] = 1
        perm_7[1] = 0
        value_layer = x.transpose(perm=perm_7)
        context_layer = paddle.bmm(x=attention_probs, y=value_layer)

        context_layer = context_layer.reshape(
            [b, self.num_attention_heads_per_partition, sq, self.hidden_size_per_attention_head]
        )
        context_layer = context_layer.transpose(perm=[2, 0, 1, 3])
        new_context_layer_shape = context_layer.shape[:-2] + [self.hidden_size_per_partition]
        context_layer = context_layer.reshape(new_context_layer_shape)
        output = self.out_proj(context_layer)
        return output


class VisualAttentionBlock(paddle.nn.Layer):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = paddle.nn.GELU,
        norm_layer: Callable = paddle.nn.LayerNorm,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        self.ln_1 = norm_layer(d_model)
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.attn = VisualAttention(d_model, n_head)
        self.mlp = paddle.nn.Sequential(
            *[
                ("c_fc", paddle.nn.Linear(in_features=d_model, out_features=mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", paddle.nn.Linear(in_features=mlp_width, out_features=d_model)),
            ]
        )

    def attention(
        self,
        q_x: paddle.Tensor,
        k_x: Optional[paddle.Tensor] = None,
        v_x: Optional[paddle.Tensor] = None,
        attn_mask: Optional[paddle.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.astype(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, attn_mask=attn_mask)

    def forward(
        self,
        q_x: paddle.Tensor,
        k_x: Optional[paddle.Tensor] = None,
        v_x: Optional[paddle.Tensor] = None,
        attn_mask: Optional[paddle.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerBlock(paddle.nn.Layer):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = paddle.nn.GELU,
        norm_layer: Callable = paddle.nn.LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = paddle.nn.LayerList(
            sublayers=[
                VisualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer, norm_layer=norm_layer)
                for _ in range(layers)
            ]
        )

    def get_cast_dtype(self) -> paddle.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def get_cast_device(self) -> str:
        return self.resblocks[0].mlp.c_fc.weight.device

    def forward(self, x: paddle.Tensor, attn_mask: Optional[paddle.Tensor] = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)

        return x


class VisionTransformerConfig(PretrainedConfig):

    model_type = "vision_transformer"
    attribute_map: Dict[str, str] = {}

    def __init__(
        self,
        image_size: int = 448,
        patch_size: int = 14,
        width: int = 1664,
        layers: int = 48,
        heads: int = 16,
        mlp_ratio: float = 4.9231,
        output_dim: int = 4096,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.output_dim = output_dim

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        if "visual" in config_dict:
            config_dict = config_dict["visual"]

        return cls.from_dict(config_dict, **kwargs)


class VisionTransformerPretrainedModel(MixPretrainedModel):
    """
    See :class:`~paddlemix.models.model_utils.MixPretrainedModel` for more details.
    """

    model_config_file = "config.json"
    config_class = VisionTransformerConfig
    resource_files_names = {"model_state": "model_state_vision.pdparams"}
    base_model_prefix = "vision_transformer"


class VisionTransformer(VisionTransformerPretrainedModel):
    def __init__(self, config: VisionTransformerConfig):
        super().__init__(config)

        image_height, image_width = self.image_size = config.image_size, config.image_size
        patch_height, patch_width = self.patch_size = config.patch_size, config.patch_size
        self.grid_size = (image_height // patch_height, image_width // patch_width)

        self.conv1 = paddle.nn.Conv2D(
            in_channels=3,
            out_channels=config.width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias_attr=False,
        )
        scale = config.width**-0.5
        out_4 = paddle.create_parameter(
            shape=(scale * paddle.randn(shape=[256, config.width])).shape,
            dtype=(scale * paddle.randn(shape=[256, config.width])).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(scale * paddle.randn(shape=[256, config.width])),
        )
        out_4.stop_gradient = not True
        self.positional_embedding = out_4
        norm_layer = partial(paddle.nn.LayerNorm, epsilon=1e-06)
        act_layer = paddle.nn.GELU
        self.ln_pre = norm_layer(config.width)
        self.transformer = TransformerBlock(
            config.width, config.layers, config.heads, config.mlp_ratio, act_layer=act_layer, norm_layer=norm_layer
        )

    def forward(self, x: paddle.Tensor):
        x = x.astype(dtype=self.conv1.weight.dtype)

        x = self.conv1(x)

        x = x.reshape([x.shape[0], x.shape[1], self.grid_size[0] * self.grid_size[1]])
        x = x.transpose(perm=[0, 2, 1])
        x = x + get_abs_pos(self.positional_embedding, x.shape[1])

        x = self.ln_pre(x)
        x = x.transpose(perm=[1, 0, 2])

        x = self.transformer(x)
        x = x.transpose(perm=[1, 0, 2])

        return x
