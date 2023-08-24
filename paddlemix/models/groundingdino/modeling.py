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

import copy
from typing import List

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import BertModel, RobertaModel
from paddlenlp.transformers.model_utils import register_base_model
from paddlenlp.utils.initializer import constant_, xavier_uniform_

from paddlemix.models.model_utils import MixPretrainedModel

from .backbone import build_backbone
from .bertwarper import BertModelWarper
from .configuration import GroundingDinoConfig
from .transformer import build_transformer
from .utils import MLP, ContrastiveEmbed, inverse_sigmoid

__all__ = [
    "GroundingDinoModel",
    "GroundingDinoPretrainedModel",
]


class GroundingDinoPretrainedModel(MixPretrainedModel):
    """
    See :class:`paddlemix.models.model_utils.MixPretrainedModel` for more details.
    """

    model_config_file = "config.json"
    config_class = GroundingDinoConfig
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "grounddingDino"


@register_base_model
class GroundingDinoModel(GroundingDinoPretrainedModel):
    """
    Args:
        config (:class:`GroundingDinoConfig`):
            An instance of BertConfig used to construct BertModel.
    """

    def __init__(self, config: GroundingDinoConfig):
        super(GroundingDinoModel, self).__init__(config)

        self.query_dim = config.query_dim
        self.backbone = build_backbone(config)
        self.transformer = build_transformer(config)
        self.hidden_dim = hidden_dim = self.transformer.d_model
        self.num_feature_levels = config.num_feature_levels
        self.nheads = config.nheads
        self.max_text_len = config.max_text_len
        self.sub_sentence_present = config.sub_sentence_present

        # bert
        if config.text_encoder_type == "bert-base-uncased":
            self.bert = BertModel.from_pretrained(config.text_encoder_type)
        elif config.text_encoder_type == "roberta-base":
            self.bert = RobertaModel.from_pretrained(config.text_encoder_type)
        else:
            raise ValueError("Unknown text_encoder_type {}".format(config.text_encoder_type))
        self.bert.pooler.dense.weight.stop_gradient = True
        self.bert.pooler.dense.bias.stop_gradient = True
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias_attr=True)
        constant_(self.feat_map.bias, 0)
        xavier_uniform_(self.feat_map.weight)

        # prepare input projection layers
        if config.num_feature_levels > 1:
            num_backbone_outs = len(self.backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2D(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(config.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2D(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.LayerList(input_proj_list)
        else:
            # assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.LayerList(
                [
                    nn.Sequential(
                        nn.Conv2D(self.backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        # prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        constant_(_bbox_embed.layers[-1].weight, 0)
        constant_(_bbox_embed.layers[-1].bias, 0)

        if config.dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(self.transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for i in range(self.transformer.num_decoder_layers)]
        class_embed_layerlist = [_class_embed for i in range(self.transformer.num_decoder_layers)]
        self.bbox_embed = nn.LayerList(box_embed_layerlist)
        self.class_embed = nn.LayerList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        assert config.two_stage_type in [
            "no",
            "standard",
        ], "unknown param {} of two_stage_type".format(config.two_stage_type)
        if config.two_stage_type != "no":
            if config.two_stage_bbox_embed_share:
                assert config.dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if config.two_stage_class_embed_share:
                assert config.dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            xavier_uniform_(proj[0].weight, gain=1)
            constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    def forward(
        self,
        x: paddle.Tensor,
        m: paddle.Tensor,
        input_ids: paddle.Tensor,
        attention_mask: paddle.Tensor,
        text_self_attention_masks: paddle.Tensor,
        position_ids: paddle.Tensor = None,
        targets: List = None,
    ):

        tokenized = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            # import ipdb; ipdb.set_trace()
            tokenized_for_encoder = tokenized

        bert_output = self.bert(**tokenized_for_encoder)  # bs, 195, 768

        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        text_token_mask = tokenized["attention_mask"].cast(paddle.bool)  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[:, : self.max_text_len, : self.max_text_len]

        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

        features, feat_masks, poss = self.backbone(x, m)

        srcs = []
        masks = []
        for l, src in enumerate(features):
            # src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(feat_masks[l])
            # assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    # src = self.input_proj[l](features[-1].tensors)
                    src = self.input_proj[l](features[-1])
                else:
                    src = self.input_proj[l](srcs[-1])
                # m = samples.mask
                mask = F.interpolate(m[None].cast(paddle.float32), size=src.shape[-2:]).cast(paddle.bool)[0]
                # pos_l = self.backbone[1](NestedTensor(src, mask)).cast(src.dtype)
                pos_l = self.backbone[1](mask).cast(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        # input_query_bbox = input_query_label = attn_mask = dn_meta = None
        input_query_bbox = input_query_label = attn_mask = None
        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
        )

        # deformable-detr-like anchor update
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = F.sigmoid(layer_outputs_unsig)
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = paddle.stack(outputs_coord_list)

        # output
        outputs_class = paddle.stack(
            [layer_cls_embed(layer_hs, text_dict) for layer_cls_embed, layer_hs in zip(self.class_embed, hs)]
        )

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}

        return out
