# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The Salesforce Team Authors and The HuggingFace Team. All rights reserved.
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
""" BLIP-2 model configuration"""
import copy
import os
from typing import Union

from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.utils.log import logger

__all__ = [
    "Blip2VisionConfig",
    "Blip2QFormerConfig",
    "Blip2Config",
]


class Blip2VisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VisionTransformer`]. It is used to instantiate a
    BLIP-2 visual encoder according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        embed_dim (`int`, *optional*, defaults to 1408):
            Dimensionality of the encoder layers and the pooler layer.
        depth (`int`, *optional*, defaults to 39):
            Number of hidden layers in the Transformer encoder.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries and values in the self-attention layers.
        mlp_ratio (`int`, *optional*, defaults to 14):
            The ratio of mlp hidden dim
        epsilon (`float`, *optional*, defaults to 1e-6):
            epsilon for nn.LayerNorm
        drop_rate (`float`, *optional*, defaults to 0):
            dropout rate for nn.Dropout
    ```"""
    model_type = "blip_2_vision_model"

    def __init__(
        self,
        img_size=224,
        patch_size=14,
        embed_dim=1408,
        depth=39,
        num_heads=16,
        mlp_ratio=4.3637,
        qkv_bias=True,
        drop_rate=0,
        epsilon=1e-6,
        gradient_checkpointing=False,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.epsilon = epsilon
        self.gradient_checkpointing = gradient_checkpointing

        self.in_chans = kwargs.get("in_chans", 3)
        self.class_num = kwargs.get("class_num", 1000)
        self.qk_scale = kwargs.get("qk_scale", None)
        self.attn_drop_rate = kwargs.get("attn_drop_rate", 0.0)
        self.drop_path_rate = kwargs.get("drop_path_rate", 0.0)
        self.norm_layer = kwargs.get("norm_layer", "nn.LayerNorm")
        self.use_fusedlinear = kwargs.get("use_fusedlinear", False)
        self.use_flash_attn = kwargs.get("use_flash_attn", False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from Blip2Config
        if config_dict.get("model_type") == "blip-2":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class Blip2QFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Blip2QFormerModel`]. It is used to instantiate a
    BLIP-2 Querying Transformer (Q-Former) model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the BLIP-2
    [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) architecture. Configuration objects
    inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the documentation from
    [`PretrainedConfig`] for more information.
    Note that [`Blip2QFormerModel`] is very similar to [`BertLMHeadModel`] with interleaved cross-attention.
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the Q-Former model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling the model.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional*, defaults to `"absolute"`):
            Type of position embedding. Choose one of `"absolute"`, `"relative_key"`, `"relative_key_query"`. For
            positional embeddings use `"absolute"`. For more information on `"relative_key"`, please refer to
            [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155).
            For more information on `"relative_key_query"`, please refer to *Method 4* in [Improve Transformer Models
            with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658).
        classifier_dropout (`float`, *optional*):
            The dropout ratio for the classification head.
        cross_attention_frequency (`int`, *optional*, defaults to 2):
            The frequency of adding cross-attention to the Transformer layers.
        encoder_hidden_size (`int`, *optional*, defaults to 1408):
            The hidden size of the hidden states for cross-attention.
    """
    model_type = "blip_2_qformer"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        classifier_dropout=None,
        cross_attention_frequency=2,
        encoder_hidden_size=1408,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.classifier_dropout = classifier_dropout
        self.cross_attention_frequency = cross_attention_frequency
        self.encoder_hidden_size = encoder_hidden_size
        self.dropout = kwargs.pop("dropout", None)
        self.use_fusedlinear = kwargs.get("use_fusedlinear", False)
        self.use_flash_attn = kwargs.get("use_flash_attn", False)
        self.tokenizer_name = kwargs.get("tokenizer_name", None)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the qformer config dict if we are loading from Blip2Config
        if config_dict.get("model_type") == "blip-2":
            config_dict = config_dict["qformer_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class Blip2Config(PretrainedConfig):
    r"""
    [`Blip2Config`] is the configuration class to store the configuration of a [`Blip2ForConditionalGeneration`]. It is
    used to instantiate a BLIP-2 model according to the specified arguments, defining the vision model, Q-Former model
    and language model configs. Instantiating a configuration with the defaults will yield a similar configuration to
    that of the BLIP-2 [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Blip2VisionConfig`].
        qformer_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Blip2QFormerConfig`].
        text_config (`str`, *optional*):
            the name or path of language model
        kwargs (*optional*):
            Dictionary of keyword arguments.
    """

    model_type = "blip-2"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        qformer_config=None,
        text_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the Blip2VisionConfig with default values.")

        if qformer_config is None:
            qformer_config = {}
            logger.info("qformer_config is None. Initializing the Blip2QFormerConfig with default values.")

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the text config with default values (`OPTConfig`).")
        self.vision_config = Blip2VisionConfig(**vision_config)
        self.qformer_config = Blip2QFormerConfig(**qformer_config)
        self.text_config = text_config
        self.initializer_factor = 1.0
        self.initializer_range = 0.02
        self.freeze_vit = kwargs.get("freeze_vit", True)
        self.mp_degree = kwargs.get("mp_gree", 1)
        self.use_decoder_only_language_model = kwargs.get("use_decoder_only_language_model", True)

    @classmethod
    def from_vision_qformer_text_configs(
        cls,
        vision_config: Blip2VisionConfig,
        qformer_config: Blip2QFormerConfig,
        text_config: PretrainedConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`Blip2Config`] (or a derived class) from a BLIP-2 vision model, Q-Former and language model
        configurations.
        Returns:
            [`Blip2Config`]: An instance of a configuration object
        """

        return cls(
            vision_config=vision_config.to_dict(),
            qformer_config=qformer_config.to_dict(),
            text_config=text_config.to_dict(),
            **kwargs,
        )

    def to_dict(self, saving_file=False):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].
        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["qformer_config"] = self.qformer_config.to_dict()
        output["text_config"] = self.text_config
        output["model_type"] = self.__class__.model_type
        return output
