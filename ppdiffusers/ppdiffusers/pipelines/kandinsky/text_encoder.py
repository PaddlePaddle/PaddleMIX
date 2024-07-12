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

import paddle.nn as nn
from paddlenlp.utils.converter import StateDictNameMapping

from ppdiffusers.transformers import PretrainedModel, XLMRobertaConfig, XLMRobertaModel


class MCLIPConfig(XLMRobertaConfig):
    model_type = "M-CLIP"

    def __init__(self, transformerDimSize=1024, imageDimSize=768, **kwargs):
        self.transformerDimensions = transformerDimSize
        self.numDims = imageDimSize
        super().__init__(**kwargs)


class MultilingualCLIP(PretrainedModel):
    config_class = MCLIPConfig

    @classmethod
    def _get_name_mappings(cls, config):
        mappings = []
        model_mappings = [
            ["embeddings.word_embeddings.weight", "embeddings.word_embeddings.weight"],
            ["embeddings.position_ids", "embeddings.position_ids"],
            ["embeddings.position_embeddings.weight", "embeddings.position_embeddings.weight"],
            ["embeddings.token_type_embeddings.weight", "embeddings.token_type_embeddings.weight"],
            ["embeddings.LayerNorm.weight", "embeddings.LayerNorm.weight"],
            ["embeddings.LayerNorm.bias", "embeddings.LayerNorm.bias"],
            ["pooler.dense.weight", "pooler.dense.weight", "transpose"],
            ["pooler.dense.bias", "pooler.dense.bias"],
            # for TokenClassification
        ]
        for layer_index in range(config.num_hidden_layers):
            for name in [
                "attention.self.query",
                "attention.self.key",
                "attention.self.value",
                "attention.output.dense",
                "attention.output.LayerNorm",
                "intermediate.dense",
                "output.dense",
                "output.LayerNorm",
            ]:
                action = None if "LayerNorm" in name else "transpose"
                model_mappings.extend(
                    [
                        [
                            f"encoder.layer.{layer_index}.{name}.weight",
                            f"encoder.layer.{layer_index}.{name}.weight",
                            action,
                        ],
                        [
                            f"encoder.layer.{layer_index}.{name}.bias",
                            f"encoder.layer.{layer_index}.{name}.bias",
                        ],
                    ]
                )

        torch_prefix = "transformer."
        paddle_prefix = "transformer."

        # add prefix
        for mapping in model_mappings:
            mapping[0] = torch_prefix + mapping[0]
            mapping[1] = paddle_prefix + mapping[1]

        model_mappings.extend(
            [
                ["LinearTransformation.weight", "LinearTransformation.weight", "transpose"],
                ["LinearTransformation.bias", "LinearTransformation.bias"],
            ]
        )
        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(model_mappings)]
        return mappings

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.transformer = XLMRobertaModel(config)
        self.LinearTransformation = nn.Linear(in_features=config.transformerDimensions, out_features=config.numDims)

    def forward(self, input_ids, attention_mask):
        embs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)[0]
        attention_mask = attention_mask.cast(embs.dtype)
        embs2 = (embs * attention_mask.unsqueeze(2)).sum(axis=1) / attention_mask.sum(axis=1)[:, None]
        return self.LinearTransformation(embs2), embs
