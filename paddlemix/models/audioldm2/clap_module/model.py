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

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddlenlp.transformers import BartModel, BertModel, RobertaModel

from .htsat_model import create_htsat_model


class MLPLayers(nn.Layer):
    def __init__(self, units=[512, 512, 512], nonlin=nn.ReLU(), dropout=0.1):
        super(MLPLayers, self).__init__()
        self.nonlin = nonlin
        self.dropout = dropout

        sequence = []
        for u0, u1 in zip(units[:-1], units[1:]):
            sequence.append(nn.Linear(u0, u1))
            sequence.append(self.nonlin)
            sequence.append(nn.Dropout(self.dropout))
        sequence = sequence[:-2]

        self.sequential = nn.Sequential(*sequence)

    def forward(self, X):
        X = self.sequential(X)
        return X


class ResidualAttentionBlock(nn.Layer):
    def __init__(self, d_model: int, n_head: int, act_layer: Callable = nn.GELU):
        super().__init__()

        self.attn = nn.MultiHeadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        )
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: paddle.Tensor, attn_mask: Optional[paddle.Tensor] = None):
        return self.attn(x, x, x, attn_mask=attn_mask)[0]

    def forward(self, x: paddle.Tensor, attn_mask: Optional[paddle.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Layer):
    def __init__(self, width: int, layers: int, heads: int, act_layer: Callable = nn.GELU):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.LayerList(
            [ResidualAttentionBlock(width, heads, act_layer=act_layer) for _ in range(layers)]
        )

    def forward(self, x: paddle.Tensor, attn_mask: Optional[paddle.Tensor] = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x


# Audio Config Class
@dataclass
class CLAPAudioCfg:
    model_type: str = "HTSAT"
    model_name: str = "base"
    sample_rate: int = 48000
    audio_length: int = 1024
    window_size: int = 1024
    hop_size: int = 480
    fmin: int = 50
    fmax: int = 14000
    class_num: int = 527
    mel_bins: int = 64
    clip_samples: int = 480000


@dataclass
class CLAPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    model_type: str = "roberta"


class CLAP(nn.Layer):
    def __init__(
        self,
        embed_dim: int,
        audio_cfg: CLAPAudioCfg,
        text_cfg: CLAPTextCfg,
        quick_gelu: bool = False,
        enable_fusion: bool = False,
        fusion_type: str = "None",
        joint_embed_shape: int = 512,
        mlp_act: str = "relu",
    ):
        super().__init__()
        if isinstance(audio_cfg, dict):
            audio_cfg = CLAPAudioCfg(**audio_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLAPTextCfg(**text_cfg)

        self.audio_cfg = audio_cfg
        self.text_cfg = text_cfg
        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type
        self.joint_embed_shape = joint_embed_shape
        self.mlp_act = mlp_act

        self.context_length = text_cfg.context_length

        act_layer = nn.GELU

        if mlp_act == "relu":
            mlp_act_layer = nn.ReLU()
        elif mlp_act == "gelu":
            mlp_act_layer = nn.GELU()
        else:
            raise NotImplementedError

        # audio branch
        # audio branch parameters
        if audio_cfg.model_type == "PANN":
            raise ValueError("PANN has not been implemented.")
        elif audio_cfg.model_type == "HTSAT":
            self.audio_branch = create_htsat_model(audio_cfg, enable_fusion, fusion_type)
        else:
            logging.error(f"Model config for {audio_cfg.model_type} not found")
            raise RuntimeError(f"Model config for {audio_cfg.model_type} not found.")

        # text branch
        # text branch parameters
        if text_cfg.model_type == "transformer":
            self.text_branch = Transformer(
                width=text_cfg.width,
                layers=text_cfg.layers,
                heads=text_cfg.heads,
                act_layer=act_layer,
            )
            self.vocab_size = text_cfg.vocab_size
            self.token_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
            positional_embedding = paddle.empty([self.context_length, text_cfg.width])
            self.positional_embedding = paddle.create_parameter(
                shape=positional_embedding.shape,
                dtype=str(positional_embedding.numpy().dtype),
                default_initializer=nn.initializer.Assign(positional_embedding),
            )
            # self.ln_final = LayerNorm(text_cfg.width)
            self.ln_final = nn.LayerNorm(text_cfg.width)
            self.text_transform = MLPLayers(
                units=[
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                ],
                dropout=0.1,
            )
            self.text_projection = nn.Sequential(
                nn.Linear(text_cfg.width, self.joint_embed_shape),
                mlp_act_layer,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
            )
        elif text_cfg.model_type == "bert":
            self.text_branch = BertModel.from_pretrained("bert-base-uncased")
            self.text_transform = MLPLayers(
                units=[
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                ],
                dropout=0.1,
            )
            self.text_projection = nn.Sequential(
                nn.Linear(768, self.joint_embed_shape),
                mlp_act_layer,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
            )
        elif text_cfg.model_type == "roberta":
            self.text_branch = RobertaModel.from_pretrained("roberta-base")
            self.text_transform = MLPLayers(
                units=[
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                ],
                dropout=0.1,
            )
            self.text_projection = nn.Sequential(
                nn.Linear(768, self.joint_embed_shape),
                mlp_act_layer,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
            )
        elif text_cfg.model_type == "bart":
            self.text_branch = BartModel.from_pretrained("bart-base")
            self.text_transform = MLPLayers(
                units=[
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                    self.joint_embed_shape,
                ],
                dropout=0.1,
            )
            self.text_projection = nn.Sequential(
                nn.Linear(768, self.joint_embed_shape),
                mlp_act_layer,
                nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
            )
        else:
            logging.error(f"Model config for {text_cfg.model_type} not found")
            raise RuntimeError(f"Model config for {text_cfg.model_type} not found.")
        self.text_branch_type = text_cfg.model_type
        # text branch parameters

        # audio branch parameters
        self.audio_transform = MLPLayers(
            units=[
                self.joint_embed_shape,
                self.joint_embed_shape,
                self.joint_embed_shape,
            ],
            dropout=0.1,
        )

        # below here is text branch parameters

        self.audio_projection = nn.Sequential(
            nn.Linear(embed_dim, self.joint_embed_shape),
            mlp_act_layer,
            nn.Linear(self.joint_embed_shape, self.joint_embed_shape),
        )

        self.logit_scale_a = paddle.create_parameter(
            [], "float32", default_initializer=nn.initializer.Assign(paddle.ones([]) * np.log(1 / 0.07))
        )
        self.logit_scale_t = paddle.create_parameter(
            [], "float32", default_initializer=nn.initializer.Assign(paddle.ones([]) * np.log(1 / 0.07))
        )
        self.register_buffer("attn_mask", self.build_attention_mask(), persistable=False)

    def build_attention_mask(self):

        mask = paddle.empty([self.context_length, self.context_length]) * float("-inf")
        # mask.fill_(float("-inf"))
        mask = paddle.triu(mask, 1)  # zero out the lower diagonal
        # mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_audio(self, audio):
        return self.audio_branch(audio, mixup_lambda=None)  # mix lambda needs to add

    def encode_text(self, text):
        if self.text_branch_type == "transformer":
            x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding
            x = x.transpose([1, 0, 2])  # NLD -> LND
            x = self.text_branch(x, attn_mask=self.attn_mask)
            x = x.transpose([1, 0, 2])  # LND -> NLD
            x = self.ln_final(x)

            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = self.text_projection(x[paddle.arange(x.shape[0]), text.argmax(axis=-1)])
        elif self.text_branch_type == "bert":
            x = self.text_branch(
                input_ids=text["input_ids"],
                attention_mask=text["attention_mask"],
                token_type_ids=text["token_type_ids"],
                return_dict=True,
            )["pooler_output"]
            x = self.text_projection(x)
        elif self.text_branch_type == "roberta":
            x = self.text_branch(
                input_ids=text["input_ids"],
                attention_mask=text["attention_mask"],
                return_dict=True,
            )["pooler_output"]
            x = self.text_projection(x)
        elif self.text_branch_type == "bart":
            x = paddle.mean(
                self.text_branch(
                    input_ids=text["input_ids"],
                    attention_mask=text["attention_mask"],
                    return_dict=True,
                )["encoder_last_hidden_state"],
                axis=1,
            )
            x = self.text_projection(x)
        else:
            logging.error(f"Model type {self.text_branch_type} not found")
            raise RuntimeError(f"Model type {self.text_branch_type} not found.")
        return x

    def forward(self, audio, text):
        """Forward audio and text into the CLAP

        Parameters
        ----------
        audio: paddle.Tensor (batch_size, audio_length)
            the time-domain audio input / the batch of mel_spec and longer list.
        text: paddle.Tensor () // need to add
            the text token input
        """

        if audio is None and text is None:
            # a hack to get the logit scale
            return self.logit_scale_a.exp(), self.logit_scale_t.exp()
        elif audio is None:
            return self.encode_text(text)
        elif text is None:
            return self.audio_projection(self.encode_audio(audio)["embedding"])
        audio_features = self.audio_projection(self.encode_audio(audio)["embedding"])
        audio_features = F.normalize(audio_features, axis=-1)

        text_features = self.encode_text(text)
        text_features = F.normalize(text_features, axis=-1)

        audio_features_mlp = self.audio_transform(audio_features)
        text_features_mlp = self.text_transform(text_features)
        # Four outputs: audio features (basic & MLP), text features (basic & MLP)
        return (
            audio_features,
            text_features,
            audio_features_mlp,
            text_features_mlp,
            self.logit_scale_a.exp(),
            self.logit_scale_t.exp(),
        )

    def get_logit_scale(self):
        return self.logit_scale_a.exp(), self.logit_scale_t.exp()

    def get_text_embedding(self, data):
        """Get the text embedding from the model

        Parameters
        ----------
        data: paddle.Tensor
            a tensor of text embedding

        Returns
        ----------
        text_embed: paddle.Tensor
            a tensor of text_embeds (N, D)

        """
        text_embeds = self.encode_text(data)
        text_embeds = F.normalize(text_embeds, axis=-1)

        return text_embeds

    def get_audio_embedding(self, data):
        """Get the audio embedding from the model

        Parameters
        ----------
        data: a list of dict
            the audio input dict list from 'get_audio_feature' method

        Returns
        ----------
        audio_embed: paddle.Tensor
            a tensor of audio_embeds (N, D)

        """
        audio_embeds = self.audio_projection(self.encode_audio(data)["embedding"])
        audio_embeds = F.normalize(audio_embeds, axis=-1)

        return audio_embeds
