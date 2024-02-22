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

from paddlenlp.transformers.clap.modeling import ClapPreTrainedModel
from paddlenlp.transformers.conversion_utils import (
    StateDictNameMapping,
    init_name_mappings,
)
from paddlenlp.transformers.speecht5.modeling import SpeechT5HifiGan

from ppdiffusers.transformers.model_utils import ModuleUtilsMixin


def clap_get_name_mappings(cls, config):
    mappings = []

    model_type = config.get("model_type", "clap")

    text_num_layer_key = "num_hidden_layers"
    audio_num_layer_key = "num_hidden_layers"
    num_text_layer = 0
    num_audio_layer = 0

    if model_type in ["clap", "clap_text_model"]:
        text_config = config.get("text_config")
        if text_config:
            num_text_layer = text_config.get(text_num_layer_key, 0)
        else:
            num_text_layer = config.get(audio_num_layer_key, 0)

    if model_type in ["clap", "clap_audio_model"]:
        audio_config = config.get("audio_config")
        if audio_config:
            num_audio_layer = audio_config.get(audio_num_layer_key, 0)
            audio_depths = audio_config.get("depths", [])
            enable_patch_layer_norm = audio_config.get("enable_patch_layer_norm", True)
        else:
            num_audio_layer = config.get(audio_num_layer_key, 0)
            audio_depths = []
            enable_patch_layer_norm = True
        assert len(audio_depths) > 0, "audio_depths is empty"

    has_text_layer = num_text_layer > 0
    has_text_projection_layer = has_text_layer and (
        "ClapModel" in (config.architectures or [])
        or "ClapTextModelWithProjection" in (config.architectures or [])
        or cls.__name__ in ["ClapModel", "ClapTextModelWithProjection"]
    )

    has_audio_layer = num_audio_layer > 0
    has_audio_projection_layer = has_audio_layer and (
        "ClapModel" in (config.architectures or [])
        or "ClapAudioModelWithProjection" in (config.architectures or [])
        or cls.__name__ in ["ClapModel", "ClapAudioModelWithProjection"]
    )

    if model_type == "clap":
        hard_mappings = [["logit_scale_a", "logit_scale_a"], ["logit_scale_t", "logit_scale_t"]]
    else:
        hard_mappings = []

    # text model
    if has_text_layer:
        text_model_layer_mappings = [
            ["text_model.embeddings.word_embeddings.weight", "text_model.embeddings.word_embeddings.weight"],
            ["text_model.embeddings.position_embeddings.weight", "text_model.embeddings.position_embeddings.weight"],
            [
                "text_model.embeddings.token_type_embeddings.weight",
                "text_model.embeddings.token_type_embeddings.weight",
            ],
            ["text_model.embeddings.LayerNorm.weight", "text_model.embeddings.LayerNorm.weight"],
            ["text_model.embeddings.LayerNorm.bias", "text_model.embeddings.LayerNorm.bias"],
            ["text_model.pooler.dense.weight", "text_model.pooler.dense.weight", "transpose"],
            ["text_model.pooler.dense.bias", "text_model.pooler.dense.bias"],
        ]

        if has_text_projection_layer:
            text_model_layer_mappings.extend(
                [
                    ["text_projection.linear1.weight", "text_projection.linear1.weight", "transpose"],
                    ["text_projection.linear1.bias", "text_projection.linear1.bias"],
                    ["text_projection.linear2.weight", "text_projection.linear2.weight", "transpose"],
                    ["text_projection.linear2.bias", "text_projection.linear2.bias"],
                ]
            )

        hard_mappings.extend(text_model_layer_mappings)

        for layer_index in range(num_text_layer):
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
                hard_mappings.extend(
                    [
                        [
                            f"text_model.encoder.layer.{layer_index}.{name}.weight",
                            f"text_model.encoder.layer.{layer_index}.{name}.weight",
                            action,
                        ],
                        [
                            f"text_model.encoder.layer.{layer_index}.{name}.bias",
                            f"text_model.encoder.layer.{layer_index}.{name}.bias",
                        ],
                    ]
                )

    # audio model
    if has_audio_layer:
        audio_model_layer_mappings = [
            # patch_embed
            ["audio_model.audio_encoder.patch_embed.proj.weight", "audio_model.audio_encoder.patch_embed.proj.weight"],
            ["audio_model.audio_encoder.patch_embed.proj.bias", "audio_model.audio_encoder.patch_embed.proj.bias"],
            # batch_norm
            ["audio_model.audio_encoder.batch_norm.weight", "audio_model.audio_encoder.batch_norm.weight"],
            ["audio_model.audio_encoder.batch_norm.bias", "audio_model.audio_encoder.batch_norm.bias"],
            ["audio_model.audio_encoder.batch_norm.running_mean", "audio_model.audio_encoder.batch_norm._mean"],
            ["audio_model.audio_encoder.batch_norm.running_var", "audio_model.audio_encoder.batch_norm._variance"],
            # norm
            ["audio_model.audio_encoder.norm.weight", "audio_model.audio_encoder.norm.weight"],
            ["audio_model.audio_encoder.norm.bias", "audio_model.audio_encoder.norm.bias"],
        ]
        if enable_patch_layer_norm:
            audio_model_layer_mappings.extend(
                [
                    [
                        "audio_model.audio_encoder.patch_embed.norm.weight",
                        "audio_model.audio_encoder.patch_embed.norm.weight",
                    ],
                    [
                        "audio_model.audio_encoder.patch_embed.norm.bias",
                        "audio_model.audio_encoder.patch_embed.norm.bias",
                    ],
                ]
            )

        if has_audio_projection_layer:
            audio_model_layer_mappings.extend(
                [
                    ["audio_projection.linear1.weight", "audio_projection.linear1.weight", "transpose"],
                    ["audio_projection.linear1.bias", "audio_projection.linear1.bias"],
                    ["audio_projection.linear2.weight", "audio_projection.linear2.weight", "transpose"],
                    ["audio_projection.linear2.bias", "audio_projection.linear2.bias"],
                ]
            )
        hard_mappings.extend(audio_model_layer_mappings)
        for layer_index, depth in zip(range(num_audio_layer), audio_depths):
            for i in range(depth):
                for name in [
                    "attention.self.relative_position_bias_table",
                    "attention.self.relative_position_index",
                    "attention.self.query",
                    "attention.self.key",
                    "attention.self.value",
                    "attention.output.dense",
                    "intermediate.dense",
                    "output.dense",
                    "layernorm_before",
                    "layernorm_after",
                ]:
                    if "relative_position" in name:
                        hard_mappings.extend(
                            [
                                [
                                    f"audio_model.audio_encoder.layers.{layer_index}.blocks.{i}.{name}",
                                    f"audio_model.audio_encoder.layers.{layer_index}.blocks.{i}.{name}",
                                ]
                            ]
                        )
                    else:
                        action = None if "norm" in name else "transpose"
                        hard_mappings.extend(
                            [
                                [
                                    f"audio_model.audio_encoder.layers.{layer_index}.blocks.{i}.{name}.weight",
                                    f"audio_model.audio_encoder.layers.{layer_index}.blocks.{i}.{name}.weight",
                                    action,
                                ],
                                [
                                    f"audio_model.audio_encoder.layers.{layer_index}.blocks.{i}.{name}.bias",
                                    f"audio_model.audio_encoder.layers.{layer_index}.blocks.{i}.{name}.bias",
                                ],
                            ]
                        )
            hard_mappings.extend(
                [
                    [
                        f"audio_model.audio_encoder.layers.{layer_index}.downsample.reduction.weight",
                        f"audio_model.audio_encoder.layers.{layer_index}.downsample.reduction.weight",
                        "transpose",
                    ],
                ]
            )
            if layer_index < num_audio_layer - 1:
                hard_mappings.extend(
                    [
                        [
                            f"audio_model.audio_encoder.layers.{layer_index}.downsample.norm.weight",
                            f"audio_model.audio_encoder.layers.{layer_index}.downsample.norm.weight",
                        ],
                        [
                            f"audio_model.audio_encoder.layers.{layer_index}.downsample.norm.bias",
                            f"audio_model.audio_encoder.layers.{layer_index}.downsample.norm.bias",
                        ],
                    ]
                )

    mappings = [StateDictNameMapping(*mapping, index=index) for index, mapping in enumerate(hard_mappings)]
    return mappings


def speecht5hifigan_get_name_mappings(cls, config):
    mappings = []
    model_mappings = [
        "mean",
        "scale",
        "conv_pre.weight",
        "conv_pre.bias",
        "conv_post.weight",
        "conv_post.bias",
    ]
    resblocks_index = 0
    for i, _ in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
        model_mappings.extend(
            [
                f"upsampler.{i}.weight",
                f"upsampler.{i}.bias",
            ]
        )
        for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes):
            for d in range(len(dilation)):
                model_mappings.extend(
                    [
                        f"resblocks.{resblocks_index}.convs1.{d}.weight",
                        f"resblocks.{resblocks_index}.convs1.{d}.bias",
                        f"resblocks.{resblocks_index}.convs2.{d}.weight",
                        f"resblocks.{resblocks_index}.convs2.{d}.bias",
                    ]
                )
            resblocks_index += 1
    init_name_mappings(model_mappings)
    mappings = [StateDictNameMapping(*mapping) for mapping in model_mappings]
    return mappings


ClapPreTrainedModel._get_name_mappings = classmethod(clap_get_name_mappings)
SpeechT5HifiGan._get_name_mappings = classmethod(speecht5hifigan_get_name_mappings)

try:
    SpeechT5HifiGan.dtype = ModuleUtilsMixin.dtype
    ClapPreTrainedModel.get_extended_attention_mask = ModuleUtilsMixin.get_extended_attention_mask
    ClapPreTrainedModel.dtype = ModuleUtilsMixin.dtype
except Exception:
    pass
