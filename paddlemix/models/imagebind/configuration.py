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
import os
from typing import Union

from paddlenlp.transformers.clip.configuration import CLIPTextConfig, CLIPVisionConfig
from paddlenlp.transformers.configuration_utils import PretrainedConfig

from paddlemix.utils.log import logger

__all__ = [
    "ImageBindVisionConfig",
    "ImageBindTextConfig",
    "ImageBindConfig",
    "ImageBindAudioConfig",
]


class ImageBindVisionConfig(CLIPVisionConfig):

    model_type = "imagebind_vision_model"

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
        **kwargs,
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


class ImageBindTextConfig(CLIPTextConfig):

    model_type = "imagebind_text_model"

    def __init__(
        self,
        vocab_size=49408,
        hidden_size=512,
        intermediate_size=2048,
        projection_dim=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=0.00001,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.projection_dim = projection_dim
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout


class ImageBindAudioConfig(PretrainedConfig):

    model_type = "imagebind_audio_model"

    def __init__(
        self,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the qformer config dict if we are loading from Blip2Config
        if config_dict.get("model_type") == "imagebind":
            config_dict = config_dict["audio_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ImageBindDepthConfig(PretrainedConfig):

    model_type = "imagebind_depth_model"

    def __init__(
        self,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the qformer config dict if we are loading from Blip2Config
        if config_dict.get("model_type") == "imagebind":
            config_dict = config_dict["depth_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ImageBindThermalConfig(PretrainedConfig):

    model_type = "imagebind_thermal_model"

    def __init__(
        self,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the qformer config dict if we are loading from Blip2Config
        if config_dict.get("model_type") == "imagebind":
            config_dict = config_dict["thermal_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ImageBindIMUConfig(PretrainedConfig):

    model_type = "imagebind_imu_model"

    def __init__(
        self,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the qformer config dict if we are loading from Blip2Config
        if config_dict.get("model_type") == "imagebind":
            config_dict = config_dict["imu_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class ImageBindConfig(PretrainedConfig):

    model_type = "imagebind"
    is_composition = True

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        audio_config=None,
        depth_config=None,
        thermal_config=None,
        imu_config=None,
        projection_dim=512,
        logit_scale_init_value=2.6592,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

        # If `_config_dict` exist, we use them for the backward compatibility.
        text_config_dict = kwargs.pop("text_config_dict", None)
        vision_config_dict = kwargs.pop("vision_config_dict", None)
        audio_config_dict = kwargs.pop("audio_config_dict", None)
        depth_config_dict = kwargs.pop("depth_config_dict", None)
        thermal_config_dict = kwargs.pop("thermal_config_dict", None)
        imu_config_dict = kwargs.pop("imu_config_dict", None)
        if text_config_dict is not None:
            text_config = text_config_dict
        if vision_config_dict is not None:
            vision_config = vision_config_dict
        if audio_config_dict is not None:
            audio_config = audio_config_dict
        if depth_config_dict is not None:
            depth_config = depth_config_dict
        if thermal_config_dict is not None:
            thermal_config = thermal_config_dict
        if imu_config_dict is not None:
            imu_config = imu_config_dict

        if text_config is None:
            text_config = {}
            logger.info("text_config is None. Initializing the ImageBindTextConfig with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the ImageBindVisionConfig with default values.")

        if audio_config is None:
            audio_config = {}
            logger.info("audio_config is None. initializing the ImageBindAudioConfig with default values.")

        if depth_config is None:
            depth_config = {}
            logger.info("depth_config is None. initializing the ImageBindDepthConfig with default values.")

        if thermal_config is None:
            thermal_config = {}
            logger.info("thermal_config is None. initializing the ImageBindThermalConfig with default values.")

        if imu_config is None:
            imu_config = {}
            logger.info("imu_config is None. initializing the ImageBindIMUConfig with default values.")

        # text_config["projection_dim"] = projection_dim
        # vision_config["projection_dim"] = projection_dim
        self.text_config = CLIPTextConfig(**text_config)
        self.vision_config = CLIPVisionConfig(**vision_config)
        self.audio_config = ImageBindAudioConfig(**audio_config)
        self.depth_config = CLIPTextConfig(**depth_config)
        self.thermal_config = CLIPVisionConfig(**thermal_config)
        self.imu_config = ImageBindAudioConfig(**imu_config)

        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(
        cls,
        text_config: ImageBindTextConfig,
        vision_config: ImageBindVisionConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`ImageBindConfig`] (or a derived class) from clip text model configuration and clip vision model
        configuration.

        Returns:
            [`ImageBindConfig`]: An instance of a configuration object
        """

        return cls(
            text_config=text_config.to_dict(),
            vision_config=vision_config.to_dict(),
            **kwargs,
        )

    def to_dict(self, saving_file=False):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["text_config"] = self.text_config.to_dict()
        output["vision_config"] = self.vision_config.to_dict()
        output["audio_config"] = self.audio_config.to_dict()
        output["depth_config"] = self.depth_config.to_dict()
        output["thermal_config"] = self.thermal_config.to_dict()
        output["imu_config"] = self.imu_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
