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

import paddle

""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import copy
import math
import os
from typing import Union

import numpy as np
from paddlenlp.transformers.configuration_utils import PretrainedConfig

from paddlemix.models.model_utils import MixPretrainedModel
from paddlemix.utils.log import logger

from .loss import ClipLoss
from .text_model import TextTransformer, TextTransformerConfig
from .vit_model import EVAVisionTransformer, EVAVisionTransformerConfig


class EVACLIPConfig(PretrainedConfig):

    model_type = "evaclip"

    def __init__(
        self,
        embed_dim=None,
        vision_cfg={},
        text_cfg={},
        fusedlinear=False,
        flash_attn=False,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

        self.vision_config = vision_cfg
        self.text_config = text_cfg
        if embed_dim is not None:
            self.vision_config["embed_dim"] = embed_dim
            self.text_config["embed_dim"] = embed_dim

        if fusedlinear:
            self.vision_config["fusedlinear"] = True
            self.text_config["fusedlinear"] = True

        if flash_attn:
            self.vision_config["xattn"] = False
            self.vision_config["flash_attn"] = True
            self.text_config["flash_attn"] = True

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike] = None,
        pretrained_vismodel_name_or_path: Union[str, os.PathLike] = None,
        pretrained_textmodel_name_or_path: Union[str, os.PathLike] = None,
        **kwargs,
    ) -> "PretrainedConfig":
        assert pretrained_model_name_or_path is not None or (
            pretrained_vismodel_name_or_path is not None and pretrained_textmodel_name_or_path is not None
        ), (
            f"Either `pretrained_model_name_or_path` or (`pretrained_vismodel_name_or_path` and `pretrained_textmodel_name_or_path`) must be set, but"
            f"received `pretrained_model_name_or_path={pretrained_model_name_or_path}` and `pretrained_vismodel_name_or_path={pretrained_vismodel_name_or_path}`, "
            f"`pretrained_textmodel_name_or_path={pretrained_textmodel_name_or_path}`"
        )
        config_dict = {}
        if pretrained_model_name_or_path is not None:
            config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

            if (
                "model_type" in config_dict
                and hasattr(cls, "model_type")
                and config_dict["model_type"] != cls.model_type
            ):
                logger.warning(
                    f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                    f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
                )
        if pretrained_vismodel_name_or_path is not None:
            visual_config_dict, kwargs = cls.get_config_dict(pretrained_vismodel_name_or_path, **kwargs)

            if "model_type" in visual_config_dict and visual_config_dict["model_type"] != "evavision_transformer":
                logger.warning(
                    f"You are using a model of type {visual_config_dict['model_type']} to instantiate a model of type "
                    f"evavision_transformer. This is not supported for all configurations of models and can yield errors."
                )
            config_dict["vision_cfg"] = visual_config_dict
        if pretrained_textmodel_name_or_path is not None:
            text_config_dict, kwargs = cls.get_config_dict(pretrained_textmodel_name_or_path, **kwargs)
            config_dict["text_cfg"] = text_config_dict

            if "model_type" in text_config_dict and text_config_dict["model_type"] != "text_transformer":
                logger.warning(
                    f"You are using a model of type {text_config_dict['model_type']} to instantiate a model of type "
                    f"text_transformer. This is not supported for all configurations of models and can yield errors."
                )

        return cls.from_dict(config_dict, **kwargs)

    def to_dict(self, saving_file=False):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["text_cfg"] = self.text_config
        output["vision_cfg"] = self.vision_config
        output["model_type"] = self.__class__.model_type
        return output


class EVACLIPPretrainedModel(MixPretrainedModel):
    """
    See :class:`paddlemix.models.model_utils.MixPretrainedModel` for more details.
    """

    model_config_file = "config.json"
    config_class = EVACLIPConfig
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "evaclip"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path=None,
        pretrained_vismodel_name_or_path=None,
        pretrained_textmodel_name_or_path=None,
        from_hf_hub: bool = False,
        subfolder: str = "",
        *args,
        **kwargs,
    ):
        assert pretrained_model_name_or_path is not None or (
            pretrained_vismodel_name_or_path is not None and pretrained_textmodel_name_or_path is not None
        ), (
            f"Either `pretrained_model_name_or_path` or (`pretrained_vismodel_name_or_path` and `pretrained_textmodel_name_or_path`) must be set, but"
            f"received `pretrained_model_name_or_path={pretrained_model_name_or_path}` and `pretrained_vismodel_name_or_path={pretrained_vismodel_name_or_path}`, "
            f"`pretrained_textmodel_name_or_path={pretrained_textmodel_name_or_path}`"
        )

        if pretrained_model_name_or_path is not None:
            return super().from_pretrained(
                pretrained_model_name_or_path,
                from_hf_hub=from_hf_hub,
                subfolder=subfolder,
                *args,
                **kwargs,
            )
        else:
            config_dict = {
                "vision_cfg": pretrained_vismodel_name_or_path,
                "text_cfg": pretrained_textmodel_name_or_path,
            }
            config = EVACLIPConfig.from_dict(config_dict)
            return cls(config, *args, **kwargs)


class EVACLIP(EVACLIPPretrainedModel):
    def __init__(
        self,
        config,
        disable_text=False,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=True,
        data_world_rank=0,
        data_world_size=1,
        enable_recompute=False,
    ):
        super().__init__(config)
        if isinstance(config.vision_config, str):
            self.visual = EVAVisionTransformer.from_pretrained(config.vision_config)
            if not disable_text:
                self.text = TextTransformer.from_pretrained(config.text_config)
        else:
            vision_config = EVAVisionTransformerConfig(**config.vision_config)
            text_config = TextTransformerConfig(**config.text_config)
            self.visual = EVAVisionTransformer(vision_config)
            if not disable_text:
                self.text = TextTransformer(text_config)
        init_data = paddle.ones(shape=[1]) * np.log(1 / 0.07)
        self.logit_scale = self.create_parameter(
            shape=[1], default_initializer=paddle.nn.initializer.Assign(init_data)
        )

        self.loss = ClipLoss(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=data_world_rank,
            world_size=data_world_size,
        )

        if enable_recompute:
            self.visual.set_grad_checkpointing(True)
            if not disable_text:
                self.text.set_grad_checkpointing(True)

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def no_weight_decay(self):
        return {"logit_scale"}

    @paddle.no_grad()
    def clip_scale(self):
        share_buffer = self.logit_scale.clip(0, math.log(100))
        self.logit_scale.copy_(share_buffer, True)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        out = paddle.nn.functional.normalize(x=features, axis=-1) if normalize else features
        return out

    def encode_text(self, text, text_features=None, normalize: bool = False):
        if text_features is not None:
            # directly use text_features if given
            return paddle.nn.functional.normalize(x=text_features, axis=-1) if normalize else text_features
        features = self.text(text)
        return paddle.nn.functional.normalize(x=features, axis=-1) if normalize else features

    def forward(self, image, input_ids=None, text_emb=None, skiploss=False, **kwargs):
        self.clip_scale()
        text = input_ids
        text_features = text_emb
        image_features = self.encode_image(image, normalize=True)
        if text is not None or text_features is not None:
            text_features = self.encode_text(text, text_features=text_features, normalize=True)
        else:
            return paddle.to_tensor(0), image_features

        if skiploss:
            return image_features, text_features, self.logit_scale.exp()

        loss_itc, logits_per_image, logits_per_text, labels = self.loss(
            (image_features, text_features, self.logit_scale.exp())
        )
        return loss_itc, image_features, text_features, self.logit_scale.exp()
