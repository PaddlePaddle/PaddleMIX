""" CoCa Model

Adapted from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/coca_model.py
"""
from typing import Optional, Union
import os
import paddle
from paddle import nn
from paddle.nn import functional as F
import numpy as np
from dataclasses import dataclass

from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.transformers.model_utils import PretrainedModel

from .loss import CoCaLoss
from .eva_vit_model import EVAVisionTransformerConfig, EVAVisionTransformer
from .transformer import (LayerNormFp32, LayerNorm, QuickGELU,
                          MultimodalTransformerConfig, MultimodalTransformer,
                          EVATextTransformerConfig, EVATextTransformer)
from paddlenlp.utils.log import logger


class CoCaConfig(PretrainedConfig):

    model_type = "coca"

    def __init__(
            self,
            vision_cfg={},
            text_cfg={},
            multimodal_cfg={},
            **kwargs, ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

        self.vision_config = vision_cfg
        self.text_config = text_cfg
        self.multimodal_config = multimodal_cfg

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: Union[str, os.PathLike],
                        **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **kwargs)

        if ("model_type" in config_dict and hasattr(cls, "model_type") and
                config_dict["model_type"] != cls.model_type):
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class CoCaPretrainedModel(PretrainedModel):
    """
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = "config.json"
    config_class = CoCaConfig
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "coca"


class CoCa(CoCaPretrainedModel):
    def __init__(
            self,
            config,
            coca_caption_loss_weight=2.0,
            coca_contrastive_loss_weight=1.0,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            data_world_rank=0,
            data_world_size=1, ):
        super().__init__(config)

        vision_config = EVAVisionTransformerConfig(**config.vision_config)
        text_config = EVATextTransformerConfig(**config.text_config)
        multimodal_config = MultimodalTransformerConfig(
            **config.multimodal_config)

        self.visual = EVAVisionTransformer(vision_config)
        self.text = EVATextTransformer(text_config)
        self.text_decoder = MultimodalTransformer(multimodal_config)

        init_data = paddle.ones(shape=[1]) * np.log(1 / 0.07)
        self.logit_scale = self.create_parameter(
            shape=[1],
            default_initializer=paddle.nn.initializer.Assign(init_data))

        self.loss = CoCaLoss(
            caption_loss_weight=coca_caption_loss_weight,
            clip_loss_weight=coca_contrastive_loss_weight,
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=data_world_rank,
            world_size=data_world_size, )

    # @paddle.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)
        self.text_decoder.set_grad_checkpointing(enable)

    def _encode_image(self, images, normalize=True):
        image_latent, tokens_embs = self.visual(images)
        image_latent = F.normalize(
            image_latent, axis=-1) if normalize else image_latent
        return image_latent, tokens_embs

    def _encode_text(self, text, normalize=True, embed_cls=True):
        text = text[:, :-1] if embed_cls else text  # make space for CLS token
        text_latent, token_emb = self.text(text)
        text_latent = F.normalize(
            text_latent, axis=-1) if normalize else text_latent
        return text_latent, token_emb

    def encode_image(self, images, normalize=True):
        image_latent, _ = self._encode_image(images, normalize=normalize)
        return image_latent

    def encode_text(self, text, normalize=True, embed_cls=True):
        text_latent, _ = self._encode_text(
            text, normalize=normalize, embed_cls=embed_cls)
        return text_latent

    def forward(self,
                image,
                input_ids,
                embed_cls=True,
                image_latent=None,
                image_embs=None,
                type_ids=None):
        text = input_ids
        text_latent, token_embs = self._encode_text(text, embed_cls=embed_cls)
        if image_latent is None or image_embs is None:
            image_latent, image_embs = self._encode_image(image)

        # TODO: add assertion to avoid bugs?
        labels = text[:, -token_embs.shape[1]:]

        logits = self.text_decoder(image_embs, token_embs)

        loss_itc, logits_per_image, logits_per_text, labels = self.loss((
            image_latent, text_latent, self.logit_scale.exp(), logits, labels))
        return loss_itc, image_latent, text_latent, self.logit_scale.exp()
