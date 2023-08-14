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
""" GroundingDino model configuration"""
import os
from typing import Union

from paddlenlp.transformers.configuration_utils import PretrainedConfig

from paddlemix.utils.log import logger

__all__ = ["GroundingDinoConfig"]


class GroundingDinoConfig(PretrainedConfig):

    model_type = "groundingdino"

    def __init__(
        self,
        modelname="groundingdino",
        backbone="swin_T_224_1k",
        position_embedding="sine",
        pe_temperatureH=20,
        pe_temperatureW=20,
        return_interm_indices=[1, 2, 3],
        backbone_freeze_keywords=None,
        enc_layers=6,
        dec_layers=6,
        pre_norm=False,
        dim_feedforward=2048,
        hidden_dim=256,
        dropout=0.0,
        nheads=8,
        num_queries=900,
        query_dim=4,
        num_patterns=0,
        num_feature_levels=4,
        enc_n_points=4,
        dec_n_points=4,
        two_stage_type="standard",
        two_stage_bbox_embed_share=False,
        two_stage_class_embed_share=False,
        transformer_activation="relu",
        dec_pred_bbox_embed_share=True,
        dn_box_noise_scale=1.0,
        dn_label_noise_ratio=0.5,
        dn_label_coef=1.0,
        dn_bbox_coef=1.0,
        embed_init_tgt=True,
        dn_labelbook_size=2000,
        max_text_len=256,
        text_encoder_type="bert-base-uncased",
        use_text_enhancer=True,
        use_fusion_layer=True,
        use_checkpoint=False,
        use_transformer_ckpt=False,
        use_text_cross_attention=True,
        text_dropout=0.0,
        fusion_dropout=0.0,
        fusion_droppath=0.1,
        sub_sentence_present=True,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)
        self.modelname = modelname
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.pe_temperatureH = pe_temperatureH
        self.pe_temperatureW = pe_temperatureW
        self.return_interm_indices = return_interm_indices
        self.backbone_freeze_keywords = backbone_freeze_keywords
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.pre_norm = pre_norm
        self.dim_feedforward = dim_feedforward
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.nheads = nheads
        self.num_queries = num_queries
        self.query_dim = query_dim
        self.num_patterns = num_patterns
        self.num_feature_levels = num_feature_levels
        self.enc_n_points = enc_n_points
        self.dec_n_points = dec_n_points
        self.two_stage_type = two_stage_type
        self.two_stage_bbox_embed_share = two_stage_bbox_embed_share
        self.two_stage_class_embed_share = two_stage_class_embed_share
        self.transformer_activation = transformer_activation
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_label_coef = dn_label_coef
        self.dn_bbox_coef = dn_bbox_coef
        self.embed_init_tgt = embed_init_tgt
        self.dn_labelbook_size = dn_labelbook_size
        self.max_text_len = max_text_len
        self.text_encoder_type = text_encoder_type
        self.use_text_enhancer = use_text_enhancer
        self.use_fusion_layer = use_fusion_layer
        self.use_checkpoint = use_checkpoint
        self.use_transformer_ckpt = use_transformer_ckpt
        self.use_text_cross_attention = use_text_cross_attention
        self.text_dropout = text_dropout
        self.fusion_dropout = fusion_dropout
        self.fusion_droppath = fusion_dropout
        self.sub_sentence_present = sub_sentence_present

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)
