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

import os
from typing import Union

import paddle
from paddle import nn
from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.utils.log import logger

from paddlemix.models.model_utils import MixPretrainedModel

from .text_model import LayerNorm, LayerNormFp32, QuickGELU, ResidualAttentionBlock
from .utils import params_normal_


class MultimodalTransformerConfig(PretrainedConfig):

    model_type = "multimodal_transformer"

    def __init__(
        self,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        context_length: int = 76,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        quick_gelu: bool = False,
        cast_dtype: Union[str, paddle.dtype] = None,
        vocab_size: int = 512,
        xattn: bool = False,
        fusedlinear: bool = False,
        flash_attn: bool = False,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

        self.width = width
        self.layers = layers
        self.heads = heads
        self.context_length = context_length
        self.mlp_ratio = mlp_ratio
        self.ls_init_value = ls_init_value
        self.quick_gelu = quick_gelu
        self.cast_dtype = cast_dtype
        self.output_dim = vocab_size
        self.xattn = xattn
        self.fusedlinear = fusedlinear
        self.flash_attn = flash_attn

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        if "multimodal_cfg" in config_dict:
            config_dict = config_dict["multimodal_cfg"]

        return cls.from_dict(config_dict, **kwargs)


class MultimodalTransformerPretrainedModel(MixPretrainedModel):
    """
    See :class:`~paddlemix.models.model_utils.MixPretrainedModel` for more details.
    """

    model_config_file = "config.json"
    config_class = MultimodalTransformerConfig
    resource_files_names = {"model_state": "model_state_multimodal.pdparams"}
    base_model_prefix = "multimodal_transformer"


class MultimodalTransformer(MultimodalTransformerPretrainedModel):
    def __init__(self, config):

        super().__init__(config)
        width = config.width
        self.context_length = config.context_length
        act_layer = QuickGELU if config.quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if config.cast_dtype in ("float16", "bfloat16") else LayerNorm

        self.enable_recompute = False
        self.resblocks = paddle.nn.LayerList(
            sublayers=[
                ResidualAttentionBlock(
                    width,
                    config.heads,
                    config.mlp_ratio,
                    ls_init_value=config.ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    xattn=config.xattn,
                    fusedlinear=config.fusedlinear,
                    flash_attn=config.flash_attn,
                )
                for _ in range(config.layers)
            ]
        )

        self.cross_attn = paddle.nn.LayerList(
            [
                ResidualAttentionBlock(
                    width,
                    config.heads,
                    config.mlp_ratio,
                    ls_init_value=config.ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    is_cross_attention=True,
                    fusedlinear=config.fusedlinear,
                    flash_attn=config.flash_attn,
                )
                for _ in range(config.layers)
            ]
        )

        self.register_buffer("attn_mask", self.build_attention_mask(), persistable=False)

        self.ln_final = norm_layer(width)
        init_data = paddle.empty([width, config.output_dim])
        self.text_projection = self.create_parameter(
            shape=[width, config.output_dim], default_initializer=paddle.nn.initializer.Assign(init_data)
        )

    def init_parameters(self):
        proj_std = (self.transformer.width**-0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5

        for block in self.transformer.resblocks:
            block.attn.in_proj_weight = params_normal_(block.attn.in_proj_weight, std=attn_std)
            block.attn.out_proj.weight = params_normal_(block.attn.out_proj.weight, std=proj_std)
            block.mlp.c_fc.weight = params_normal_(block.mlp.c_fc.weight, std=fc_std)
            block.mlp.c_proj.weight = params_normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.cross_attn:
            block.attn.in_proj_weight = params_normal_(block.attn.in_proj_weight, std=attn_std)
            block.attn.out_proj.weight = params_normal_(block.attn.out_proj.weight, std=proj_std)
            block.mlp.c_fc.weight = params_normal_(block.mlp.c_fc.weight, std=fc_std)
            block.mlp.c_proj.weight = params_normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            self.text_projection = params_normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        mask = paddle.empty([self.context_length, self.context_length])
        mask.fill_(float("-inf"))
        mask = paddle.triu(mask, 1)
        return mask

    def forward(self, image_embs, text_embs):
        seq_len = text_embs.shape[1]

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
            text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        x = self.ln_final(text_embs)

        if self.text_projection is not None:
            x = x @ self.text_projection

        return x

    # @paddle.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.enable_recompute = enable
