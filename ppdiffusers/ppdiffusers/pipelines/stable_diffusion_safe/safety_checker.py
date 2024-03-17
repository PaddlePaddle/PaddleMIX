# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from paddlenlp.utils.converter import StateDictNameMapping

from ppdiffusers.transformers import CLIPConfig, CLIPPretrainedModel, CLIPVisionModel

from ...utils import logging

logger = logging.get_logger(__name__)


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return paddle.matmul(normalized_image_embeds, normalized_text_embeds.t())


class SafeStableDiffusionSafetyChecker(CLIPPretrainedModel):
    config_class = CLIPConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    _deprecated_dict = {
        "key": ".transformer.",
        "name_mapping": {
            # common
            "clip.": "vision_model.",
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
            ["visual_projection.weight", "visual_projection.weight", "transpose"],
            ["concept_embeds", "concept_embeds"],
            ["special_care_embeds", "special_care_embeds"],
            ["concept_embeds_weights", "concept_embeds_weights"],
            ["special_care_embeds_weights", "special_care_embeds_weights"],
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

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias_attr=False)

        self.register_buffer(
            "concept_embeds",
            paddle.ones([17, config.projection_dim]),
        )
        self.register_buffer("special_care_embeds", paddle.ones([3, config.projection_dim]))

        self.register_buffer("concept_embeds_weights", paddle.ones((17,)))
        self.register_buffer("special_care_embeds_weights", paddle.ones((3,)))

    @paddle.no_grad()
    def forward(self, clip_input, images):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        special_cos_dist = (
            cosine_distance(image_embeds, self.special_care_embeds).cast(dtype=paddle.float32).cpu().numpy()
        )
        cos_dist = cosine_distance(image_embeds, self.concept_embeds).cast(dtype=paddle.float32).cpu().numpy()

        result = []
        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}

            # increase this value to create a stronger `nfsw` filter
            # at the cost of increasing the possibility of filtering benign images
            adjustment = 0.0

            for concept_idx in range(len(special_cos_dist[0])):
                concept_cos = special_cos_dist[i][concept_idx]
                concept_threshold = self.special_care_embeds_weights[concept_idx].item()
                result_img["special_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["special_scores"][concept_idx] > 0:
                    result_img["special_care"].append({concept_idx, result_img["special_scores"][concept_idx]})
                    adjustment = 0.01

            for concept_idx in range(len(cos_dist[0])):
                concept_cos = cos_dist[i][concept_idx]
                concept_threshold = self.concept_embeds_weights[concept_idx].item()
                result_img["concept_scores"][concept_idx] = round(concept_cos - concept_threshold + adjustment, 3)
                if result_img["concept_scores"][concept_idx] > 0:
                    result_img["bad_concepts"].append(concept_idx)

            result.append(result_img)

        has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]

        return images, has_nsfw_concepts

    @paddle.no_grad()
    def forward_fastdeploy(self, clip_input: paddle.Tensor, images: paddle.Tensor):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        # increase this value to create a stronger `nsfw` filter
        # at the cost of increasing the possibility of filtering benign images
        adjustment = 0.0

        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        # special_scores = special_scores.round(decimals=3)
        special_care = paddle.any(special_scores > 0, axis=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).expand(-1, cos_dist.shape[1])

        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment
        # concept_scores = concept_scores.round(decimals=3)
        has_nsfw_concepts = paddle.any(concept_scores > 0, axis=1)

        return images, has_nsfw_concepts
