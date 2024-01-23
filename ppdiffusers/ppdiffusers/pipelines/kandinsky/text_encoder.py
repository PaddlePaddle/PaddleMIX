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

from ppdiffusers.transformers import PretrainedModel, XLMRobertaConfig, XLMRobertaModel


class MCLIPConfig(XLMRobertaConfig):
    model_type = "M-CLIP"

    def __init__(self, transformerDimSize=1024, imageDimSize=768, **kwargs):
        self.transformerDimensions = transformerDimSize
        self.numDims = imageDimSize
        super().__init__(**kwargs)


class MultilingualCLIP(PretrainedModel):
    config_class = MCLIPConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.transformer = XLMRobertaModel(config)
        self.LinearTransformation = nn.Linear(in_features=config.transformerDimensions, out_features=config.numDims)

    def forward(self, input_ids, attention_mask):
        embs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)[0]
        embs2 = (embs * attention_mask.unsqueeze(2)).sum(axis=1) / attention_mask.sum(axis=1)[:, None]
        return self.LinearTransformation(embs2), embs
