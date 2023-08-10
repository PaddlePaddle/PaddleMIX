import paddle
""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import os
import math
from typing import Union
import numpy as np
from .loss import ClipLoss
from .eva_vit_model import EVAVisionTransformer
from .eva_text_model import EVATextTransformer

from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.utils.log import logger
from .eva_vit_model import EVAVisionTransformerConfig
from .eva_text_model import EVATextTransformerConfig


class EVACLIPConfig(PretrainedConfig):

    model_type = "evaclip"

    def __init__(
            self,
            vision_cfg={},
            text_cfg={},
            **kwargs, ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

        self.vision_config = vision_cfg
        self.text_config = text_cfg

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike]=None,
            visual_model_name_or_path: Union[str, os.PathLike]=None,
            text_model_name_or_path: Union[str, os.PathLike]=None,
            **kwargs) -> "PretrainedConfig":
        assert pretrained_model_name_or_path is not None or (visual_model_name_or_path is not None and text_model_name_or_path is not None), \
                f'Either `pretrained_model_name_or_path` or (`visual_model_name_or_path` and `text_model_name_or_path`) must be set, but' \
                    f'received `pretrained_model_name_or_path={pretrained_model_name_or_path}` and `visual_model_name_or_path={visual_model_name_or_path}`, ' \
                        f'`text_model_name_or_path={text_model_name_or_path}`'
        config_dict = {}
        if pretrained_model_name_or_path is not None:
            config_dict, kwargs = cls.get_config_dict(
                pretrained_model_name_or_path, **kwargs)

            if ("model_type" in config_dict and hasattr(cls, "model_type") and
                    config_dict["model_type"] != cls.model_type):
                logger.warning(
                    f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                    f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
                )
        if visual_model_name_or_path is not None:
            visual_config_dict, kwargs = cls.get_config_dict(
                visual_model_name_or_path, **kwargs)

            if ("model_type" in visual_config_dict and
                    visual_config_dict["model_type"] !=
                    "evavision_transformer"):
                logger.warning(
                    f"You are using a model of type {visual_config_dict['model_type']} to instantiate a model of type "
                    f"evavision_transformer. This is not supported for all configurations of models and can yield errors."
                )
            config_dict['vision_cfg'] = visual_config_dict
        if text_model_name_or_path is not None:
            text_config_dict, kwargs = cls.get_config_dict(
                text_model_name_or_path, **kwargs)
            config_dict['text_cfg'] = text_config_dict

            if ("model_type" in text_config_dict and
                    text_config_dict["model_type"] != "evatext_transformer"):
                logger.warning(
                    f"You are using a model of type {text_config_dict['model_type']} to instantiate a model of type "
                    f"evatext_transformer. This is not supported for all configurations of models and can yield errors."
                )

        return cls.from_dict(config_dict, **kwargs)


class EVACLIPPretrainedModel(PretrainedModel):
    """
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = "config.json"
    config_class = EVACLIPConfig
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "evaclip"


class EVACLIP(EVACLIPPretrainedModel):
    def __init__(self,
                 config,
                 disable_text=False,
                 local_loss=False,
                 gather_with_grad=False,
                 cache_labels=True,
                 data_world_rank=0,
                 data_world_size=1,
                 enable_recompute=False):
        super().__init__(config)
        vision_config = EVAVisionTransformerConfig(**config.vision_config)
        text_config = EVATextTransformerConfig(**config.text_config)
        self.visual = EVAVisionTransformer(vision_config)
        if not disable_text:
            self.text = EVATextTransformer(text_config)
        init_data = paddle.ones(shape=[1]) * np.log(1 / 0.07)
        self.logit_scale = self.create_parameter(
            shape=[1],
            default_initializer=paddle.nn.initializer.Assign(init_data))

        self.loss = ClipLoss(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=data_world_rank,
            world_size=data_world_size, )

        if enable_recompute:
            self.visual.set_grad_checkpointing(True)
            if not disable_text:
                self.text.set_grad_checkpointing(True)

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        self.visual.lock(
            unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self,
                        unlocked_layers: int=0,
                        freeze_layer_norm: bool=True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def no_weight_decay(self):
        return {'logit_scale'}

    @paddle.no_grad()
    def clip_scale(self):
        share_buffer = self.logit_scale.clip(0, math.log(100))
        self.logit_scale.copy_(share_buffer, True)

    def encode_image(self, image, normalize: bool=False):
        features = self.visual(image)
        out = paddle.nn.functional.normalize(
            x=features, axis=-1) if normalize else features
        return out

    def encode_text(self, text, text_features=None, normalize: bool=False):
        if text_features is not None:
            # directly use text_features if given
            return paddle.nn.functional.normalize(
                x=text_features, axis=-1) if normalize else text_features
        features = self.text(text)
        return paddle.nn.functional.normalize(
            x=features, axis=-1) if normalize else features

    def forward(self, image, input_ids, text_emb=None, skiploss=False):
        self.clip_scale()
        text = input_ids
        text_features = text_emb
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(
            text, text_features=text_features, normalize=True)
        if skiploss:
            return image_features, text_features, self.logit_scale.exp()

        loss_itc, logits_per_image, logits_per_text, labels = self.loss(
            (image_features, text_features, self.logit_scale.exp()))
        return loss_itc, image_features, text_features, self.logit_scale.exp()
