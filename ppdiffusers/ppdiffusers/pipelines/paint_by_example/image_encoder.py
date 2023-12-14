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
from paddle import nn
from paddlenlp.utils.converter import StateDictNameMapping

from ppdiffusers.transformers import (
    CLIPPreTrainedModel,
    CLIPVisionConfig,
    CLIPVisionModel,
)

from ...models.attention import BasicTransformerBlock
from ...utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PaintByExampleImageEncoder(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig

    @classmethod
    def _get_name_mappings(cls, config: CLIPVisionConfig):
        num_vision_layer = config.num_hidden_layers
        hard_mappings = [
            # other
            ["final_layer_norm.weight", "final_layer_norm.weight"],
            ["proj_out.weight", "proj_out.weight", "transpose"],
            ["proj_out.bias", "proj_out.bias"],
            ["uncond_vector", "uncond_vector"],
            # model prefix
            ["model.vision_model.embeddings.class_embedding", "vision_model.embeddings.class_embedding"],
            ["model.vision_model.embeddings.patch_embedding.weight", "vision_model.embeddings.patch_embedding.weight"],
            [
                "model.vision_model.embeddings.position_embedding.weight",
                "model.vision_model.embeddings.position_embedding.weight",
            ],
            ["model.vision_model.pre_layrnorm.weight", "model.vision_model.pre_layrnorm.weight"],
            ["model.vision_model.pre_layrnorm.bias", "model.vision_model.pre_layrnorm.bias"],
            ["model.vision_model.post_layernorm.weight", "model.vision_model.post_layernorm.weight"],
            ["model.vision_model.post_layernorm.bias", "model.vision_model.post_layernorm.bias"],
        ]
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
                # model prefix
                hard_mappings.extend(
                    [
                        [
                            f"model.vision_model.encoder.layers.{layer_index}.{name}.weight",
                            f"model.vision_model.encoder.layers.{layer_index}.{name}.weight",
                            action,
                        ],
                        [
                            f"model.vision_model.encoder.layers.{layer_index}.{name}.bias",
                            f"model.vision_model.encoder.layers.{layer_index}.{name}.bias",
                        ],
                    ]
                )
        num_mapper_layer = (config.num_hidden_layers + 1) // 5
        for layer_index in range(num_mapper_layer):
            # mapper prefix
            for name in [
                "attn1.to_q",
                "attn1.to_k",
                "attn1.to_v",
                "attn1.to_out",
                "ff.net.0.proj",
                "ff.net.2",
                "norm1",
                "norm3",
            ]:
                action = None if "norm" in name else "transpose"
                # model prefix
                hard_mappings.extend(
                    [
                        [
                            f"mapper.blocks.{layer_index}.{name}.weight",
                            f"mapper.blocks.{layer_index}.{name}.weight",
                            action,
                        ],
                        [
                            f"mapper.blocks.{layer_index}.{name}.bias",
                            f"mapper.blocks.{layer_index}.{name}.bias",
                        ],
                    ]
                )
        mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(hard_mappings)]
        return mappings

    def __init__(self, config: CLIPVisionConfig, proj_size=None):
        super().__init__(config)
        self.proj_size = proj_size or getattr(config, "projection_dim", 768)

        self.model = CLIPVisionModel(config)
        self.mapper = PaintByExampleMapper(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        self.proj_out = nn.Linear(config.hidden_size, self.proj_size)

        # uncondition for scaling
        self.uncond_vector = nn.Parameter(paddle.randn((1, 1, self.proj_size)))

    def forward(self, pixel_values, return_uncond_vector=False):
        clip_output = self.model(pixel_values=pixel_values)
        latent_states = clip_output.pooler_output
        latent_states = self.mapper(latent_states[:, None])
        latent_states = self.final_layer_norm(latent_states)
        latent_states = self.proj_out(latent_states)
        if return_uncond_vector:
            return latent_states, self.uncond_vector

        return latent_states


class PaintByExampleMapper(nn.Layer):
    def __init__(self, config):
        super().__init__()
        num_layers = (config.num_hidden_layers + 1) // 5
        hid_size = config.hidden_size
        num_heads = 1
        self.blocks = nn.LayerList(
            [
                BasicTransformerBlock(hid_size, num_heads, hid_size, activation_fn="gelu", attention_bias=True)
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)

        return hidden_states
