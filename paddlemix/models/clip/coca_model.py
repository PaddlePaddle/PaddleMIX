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

""" CoCa Model

Adapted from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/coca_model.py
"""
import copy
import os
from typing import Union

import numpy as np
import paddle
from paddle.nn import functional as F
from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.utils.log import logger

from paddlemix.models.model_utils import MixPretrainedModel

from .loss import CoCaLoss
from .multi_modal_model import MultimodalTransformer, MultimodalTransformerConfig
from .text_model import TextTransformer, TextTransformerConfig
from .vit_model import VisionTransformer, VisionTransformerConfig


class CoCaConfig(PretrainedConfig):

    model_type = "coca"

    def __init__(
        self,
        embed_dim=None,
        vision_cfg={},
        text_cfg={},
        multimodal_cfg={},
        fusedlinear=False,
        flash_attn=False,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

        self.vision_config = vision_cfg
        self.text_config = text_cfg
        self.multimodal_config = multimodal_cfg
        if embed_dim is not None:
            self.vision_config["embed_dim"] = embed_dim
            self.text_config["embed_dim"] = embed_dim

        if fusedlinear:
            self.vision_config["fusedlinear"] = True
            self.text_config["fusedlinear"] = True
            self.multimodal_config["fusedlinear"] = True

        if flash_attn:
            self.vision_config["flash_attn"] = True
            self.text_config["flash_attn"] = True
            self.multimodal_config["flash_attn"] = True

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
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
        output["multimodal_cfg"] = self.multimodal_config
        output["model_type"] = self.__class__.model_type
        return output


class CoCaPretrainedModel(MixPretrainedModel):
    """
    See :class:`~paddlemix.models.model_utils.MixPretrainedModel` for more details.
    """

    model_config_file = "config.json"
    config_class = CoCaConfig
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "coca"


class CoCa(CoCaPretrainedModel):
    def __init__(
        self,
        config,
        coca_caption_loss_weight=1.0,
        coca_contrastive_loss_weight=1.0,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=True,
        data_world_rank=0,
        data_world_size=1,
    ):
        super().__init__(config)

        vision_config = VisionTransformerConfig(**config.vision_config)
        text_config = TextTransformerConfig(**config.text_config)
        multimodal_config = MultimodalTransformerConfig(**config.multimodal_config)

        self.visual = VisionTransformer(vision_config)
        self.text = TextTransformer(text_config)
        self.text_decoder = MultimodalTransformer(multimodal_config)

        init_data = paddle.ones(shape=[1]) * np.log(1 / 0.07)
        self.logit_scale = self.create_parameter(
            shape=[1], default_initializer=paddle.nn.initializer.Assign(init_data)
        )

        self.loss = CoCaLoss(
            caption_loss_weight=coca_caption_loss_weight,
            clip_loss_weight=coca_contrastive_loss_weight,
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            text_loss=True,
            rank=data_world_rank,
            world_size=data_world_size,
        )

    # @paddle.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)
        self.text_decoder.set_grad_checkpointing(enable)

    def _encode_image(self, images, normalize=True):
        image_latent, tokens_embs = self.visual(images)
        image_latent = F.normalize(image_latent, axis=-1) if normalize else image_latent
        return image_latent, tokens_embs

    def _encode_text(self, text, normalize=True, embed_cls=True):
        text = text[:, :-1] if embed_cls else text  # make space for CLS token
        text_latent, token_emb = self.text(text)
        text_latent = F.normalize(text_latent, axis=-1) if normalize else text_latent
        return text_latent, token_emb

    def encode_image(self, images, normalize=True):
        image_latent, _ = self._encode_image(images, normalize=normalize)
        return image_latent

    def encode_text(self, text, normalize=True, embed_cls=True):
        text_latent, _ = self._encode_text(text, normalize=normalize, embed_cls=embed_cls)
        return text_latent

    def forward(
        self, image, input_ids=None, embed_cls=True, image_latent=None, image_embs=None, type_ids=None, **kwargs
    ):
        text = input_ids
        if image_latent is None or image_embs is None:
            image_latent, image_embs = self._encode_image(image)
        if text is not None:
            text_latent, token_embs = self._encode_text(text, embed_cls=embed_cls)
        else:
            return paddle.to_tensor(0), image_latent
        # TODO: add assertion to avoid bugs?
        labels = text[:, -token_embs.shape[1] :]

        logits = self.text_decoder(image_embs, token_embs)

        loss_itc, logits_per_image, logits_per_text, labels = self.loss(
            (image_latent, text_latent, self.logit_scale.exp(), logits, labels)
        )
        return loss_itc, image_latent, text_latent, self.logit_scale.exp()
