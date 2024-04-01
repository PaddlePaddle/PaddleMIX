# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Optional, Union

from paddlenlp.transformers.configuration_utils import (
    PretrainedConfig,
    convert_to_legacy_config,
    flatten_model_config,
)
from paddlenlp.utils.log import logger

__all__ = ["CLIPVisionConfig"]


class Old2NewPretrainedConfig(PretrainedConfig):
    old_config_dict = [
        "image_resolution",
        "vision_layers",
        "vision_heads",
        "vision_embed_dim",
        "vision_patch_size",
        "vision_mlp_ratio",
        "vision_hidden_act",
        "max_text_length",
        "vocab_size",
        "text_embed_dim",
        "text_heads",
        "text_layers",
        "text_hidden_act",
        "projection_dim",
        "initializer_range",
        "initializer_factor",
        "logit_scale_init_value",
        "init_class",
    ]
    text_name_mapping = {
        "max_text_length": "max_position_embeddings",
        "vocab_size": "vocab_size",
        "text_embed_dim": "hidden_size",
        "text_heads": "num_attention_heads",
        "text_layers": "num_hidden_layers",
        "text_hidden_act": "hidden_act",
        "initializer_range": "initializer_range",
        "initializer_factor": "initializer_factor",
        "projection_dim": "projection_dim",
    }
    vision_name_mapping = {
        "image_resolution": "image_size",
        "vision_layers": "num_hidden_layers",
        "vision_heads": "num_attention_heads",
        "vision_embed_dim": "hidden_size",
        "vision_patch_size": "patch_size",
        "vision_hidden_act": "hidden_act",
        "initializer_range": "initializer_range",
        "initializer_factor": "initializer_factor",
        "projection_dim": "projection_dim",
    }

    @classmethod
    def from_dict(cls, config_dict, **kwargs) -> "PretrainedConfig":
        """
        Instantiates a [`PretrainedConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the [`~PretrainedConfig.get_config_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`PretrainedConfig`]: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        # Those arguments may be passed along for our internal telemetry.
        # We remove them so they don't appear in `return_unused_kwargs`.
        # convert local config to legacy config
        # do standard config map: there are some old-school pretrained-config not refactored.
        config_dict = convert_to_legacy_config(cls.attribute_map, config_dict)
        config_dict = flatten_model_config(config_dict)

        # check old_config?
        is_old_config = "vision_layers" in config_dict or "text_layers" in config_dict
        if is_old_config:
            # convert to new_config
            old_config_dict = {}
            for old_name in cls.old_config_dict:
                value = config_dict.pop(old_name, None)
                if value is not None:
                    old_config_dict[old_name] = value

            # convert text config
            if cls.model_type in ["clip", "clip_text_model"]:
                text_config = {}
                for old_name, new_name in cls.text_name_mapping.items():
                    old_value = old_config_dict.get(old_name, None)
                    if old_value is not None:
                        text_config[new_name] = old_value
                if "hidden_size" in text_config:
                    text_config["intermediate_size"] = 4 * text_config["hidden_size"]

                if cls.model_type == "clip":
                    config_dict["text_config_dict"] = text_config
                else:
                    config_dict.update(text_config)

            # convert vision config
            if cls.model_type in ["clip", "clip_vision_model"]:
                vision_config = {}
                for old_name, new_name in cls.vision_name_mapping.items():
                    old_value = old_config_dict.get(old_name, None)
                    if old_value is not None:
                        vision_config[new_name] = old_value
                if "hidden_size" in vision_config:
                    radio = old_config_dict.get("vision_mlp_ratio", 4)
                    vision_config["intermediate_size"] = radio * vision_config["hidden_size"]
                if cls.model_type == "clip":
                    config_dict["vision_config_dict"] = vision_config
                else:
                    config_dict.update(vision_config)

            if cls.model_type == "clip":
                # convert common config
                if "projection_dim" in old_config_dict:
                    config_dict["projection_dim"] = old_config_dict["projection_dim"]
                if "logit_scale_init_value" in old_config_dict:
                    config_dict["logit_scale_init_value"] = old_config_dict["logit_scale_init_value"]

        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

        # Update config with kwargs if needed
        if "num_labels" in kwargs and "id2label" in kwargs:
            num_labels = kwargs["num_labels"]
            id2label = kwargs["id2label"] if kwargs["id2label"] is not None else []
            if len(id2label) != num_labels:
                raise ValueError(
                    f"You passed along `num_labels={num_labels }` with an incompatible id to label map: "
                    f"{kwargs['id2label']}. Since those arguments are inconsistent with each other, you should remove "
                    "one of them."
                )
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                if key != "dtype":
                    to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Model config {config}")
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config


class CLIPVisionConfig(Old2NewPretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CLIPModel`]. It is used to instantiate an CLIP
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the CLIP
    [openai/clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 32):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
        layer_norm_eps (`float`, *optional*,
            defaults to 1e-5): The epsilon used by the layer normalization layers.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        initializer_factor (`float``, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).

    Example:

    ```python
    >>> from paddlenlp.transformers import CLIPVisionConfig, CLIPVisionModel

    >>> # Initializing a CLIPVisionConfig with openai/clip-vit-base-patch32 style configuration
    >>> configuration = CLIPVisionConfig()

    >>> # Initializing a CLIPVisionModel (with random weights) from the openai/clip-vit-base-patch32 style configuration
    >>> model = CLIPVisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "clip_vision_model"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=32,
        hidden_act="quick_gelu",
        layer_norm_eps=0.00001,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        **kwargs
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        from_hf_hub: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs
    ) -> PretrainedConfig:
        kwargs.update({"from_hf_hub": from_hf_hub, "cache_dir": cache_dir})
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from CLIPConfig
        if config_dict.get("model_type") == "clip":
            projection_dim = config_dict.get("projection_dim", None)
            config_dict = config_dict["vision_config"]
            if projection_dim is not None:
                config_dict["projection_dim"] = projection_dim

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)
