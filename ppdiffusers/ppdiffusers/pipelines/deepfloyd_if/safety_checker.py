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

import numpy as np
import paddle
import paddle.nn as nn
from paddlenlp.utils.converter import StateDictNameMapping

from ppdiffusers.transformers import (
    CLIPConfig,
    CLIPPretrainedModel,
    CLIPVisionModelWithProjection,
)

from ...utils import logging

logger = logging.get_logger(__name__)


class IFSafetyChecker(CLIPPretrainedModel):
    config_class = CLIPConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    _deprecated_dict = {
        "key": ".transformer.",
        "name_mapping": {
            # common
            ".transformer.": ".encoder.",
            ".positional_embedding.": ".embeddings.position_embedding.",
            ".linear1.": ".mlp.fc1.",
            ".linear2.": ".mlp.fc2.",
            ".norm1.": ".layer_norm1.",
            ".norm2.": ".layer_norm2.",
            ".class_embedding": ".embeddings.class_embedding",
            ".conv1.weight": ".embeddings.patch_embedding.weight",
            ".ln_pre.": ".pre_layrnorm.",
            ".ln_post.": ".post_layernorm.",
            # projection
            "vision_projection": "visual_projection.weight",
        },
    }

    @classmethod
    def _get_name_mappings(cls, config):
        mappings = []
        model_type = config.get("model_type", "clip")
        num_layer_key = "num_hidden_layers"
        num_vision_layer = 0

        if model_type in ["clip", "clip_vision_model"]:
            vision_config = config.get("vision_config")
            if vision_config:
                num_vision_layer = vision_config.get(num_layer_key, 0)
            else:
                num_vision_layer = config.get(num_layer_key, 0)

        hard_mappings = []
        safety_checker_layer_mappings = [
            ["p_head.weight", "p_head.weight", "transpose"],
            ["w_head.weight", "w_head.weight", "transpose"],
            # vision
            [
                "vision_model.vision_model.embeddings.class_embedding",
                "vision_model.vision_model.embeddings.class_embedding",
            ],
            [
                "vision_model.vision_model.embeddings.patch_embedding.weight",
                "vision_model.vision_model.embeddings.patch_embedding.weight",
            ],
            [
                "vision_model.vision_model.embeddings.position_embedding.weight",
                "vision_model.vision_model.embeddings.position_embedding.weight",
            ],
            ["vision_model.vision_model.pre_layrnorm.weight", "vision_model.vision_model.pre_layrnorm.weight"],
            ["vision_model.vision_model.pre_layrnorm.bias", "vision_model.vision_model.pre_layrnorm.bias"],
            ["vision_model.vision_model.post_layernorm.weight", "vision_model.vision_model.post_layernorm.weight"],
            ["vision_model.vision_model.post_layernorm.bias", "vision_model.vision_model.post_layernorm.bias"],
        ]

        hard_mappings.extend(safety_checker_layer_mappings)
        for layer_index in range(num_vision_layer):
            for name in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.out_proj",
                "mlp.fc1",
                "mlp.fc2",
                "layer_norm1",
                "layer_norm2",
            ]:
                action = None if "norm" in name else "transpose"
                hard_mappings.extend(
                    [
                        [
                            f"vision_model.vision_model.encoder.layers.{layer_index}.{name}.weight",
                            f"vision_model.vision_model.encoder.layers.{layer_index}.{name}.weight",
                            action,
                        ],
                        [
                            f"vision_model.vision_model.encoder.layers.{layer_index}.{name}.bias",
                            f"vision_model.vision_model.encoder.layers.{layer_index}.{name}.bias",
                        ],
                    ]
                )

        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(hard_mappings)]
        return mappings

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModelWithProjection(config.vision_config)

        self.p_head = nn.Linear(config.vision_config.projection_dim, 1)
        self.w_head = nn.Linear(config.vision_config.projection_dim, 1)

    @paddle.no_grad()
    def forward(self, clip_input, images, p_threshold=0.5, w_threshold=0.5):
        image_embeds = self.vision_model(clip_input)[0]

        nsfw_detected = self.p_head(image_embeds)
        nsfw_detected = nsfw_detected.flatten()
        nsfw_detected = nsfw_detected > p_threshold
        nsfw_detected = nsfw_detected.tolist()

        if any(nsfw_detected):
            logger.warning(
                "Potential NSFW content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed."
            )

        for idx, nsfw_detected_ in enumerate(nsfw_detected):
            if nsfw_detected_:
                images[idx] = np.zeros(images[idx].shape)

        watermark_detected = self.w_head(image_embeds)
        watermark_detected = watermark_detected.flatten()
        watermark_detected = watermark_detected > w_threshold
        watermark_detected = watermark_detected.tolist()

        if any(watermark_detected):
            logger.warning(
                "Potential watermarked content was detected in one or more images. A black image will be returned instead."
                " Try again with a different prompt and/or seed."
            )

        for idx, watermark_detected_ in enumerate(watermark_detected):
            if watermark_detected_:
                images[idx] = np.zeros(images[idx].shape)

        return images, nsfw_detected, watermark_detected
