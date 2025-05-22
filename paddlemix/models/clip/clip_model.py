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
import paddle.nn.functional as F

""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import copy
import math
import os
from typing import Union

import numpy as np
from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.utils.log import logger

from paddlemix.models.model_utils import MixPretrainedModel

from .loss import ClipLoss
from .text_model import TextTransformer, TextTransformerConfig
from .vit_model import VisionTransformer, VisionTransformerConfig


class CLIPConfig(PretrainedConfig):

    model_type = "clip"

    def __init__(
        self,
        embed_dim=None,
        vision_cfg={},
        text_cfg={},
        custom_text=False,
        fusedlinear=False,
        flash_attn=False,
        **kwargs,
    ):
        kwargs["return_dict"] = kwargs.pop("return_dict", True)
        super().__init__(**kwargs)

        self.vision_config = vision_cfg
        self.text_config = text_cfg
        self.custom_text = custom_text
        if embed_dim is not None:
            self.vision_config["embed_dim"] = embed_dim
            self.text_config["embed_dim"] = embed_dim

        if fusedlinear:
            self.vision_config["fusedlinear"] = True
            self.text_config["fusedlinear"] = True

        if flash_attn:
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


class CLIPPretrainedModel(MixPretrainedModel):
    """
    See :class:`~paddlemix.models.model_utils.MixPretrainedModel` for more details.
    """

    model_config_file = "config.json"
    config_class = CLIPConfig
    resource_files_names = {"model_state": "model_state.pdparams"}
    base_model_prefix = "clip"

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
            config = CLIPConfig.from_dict(config_dict)
            return cls(config, *args, **kwargs)


class CLIP(CLIPPretrainedModel):
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
            self.visual = VisionTransformer.from_pretrained(config.vision_config)
            if not disable_text:
                self.text = TextTransformer.from_pretrained(config.text_config)
        else:
            vision_config = VisionTransformerConfig(**config.vision_config)
            text_config = TextTransformerConfig(**config.text_config)
            self.visual = VisionTransformer(vision_config)
            if not disable_text:
                if config.custom_text:
                    self.text = TextTransformer(text_config)
                else:
                    text = TextTransformer(text_config)
                    self.transformer = text.transformer
                    # self.context_length = text.context_length
                    # self.vocab_size = text.vocab_size
                    self.token_embedding = text.token_embedding
                    self.positional_embedding = text.positional_embedding
                    self.ln_final = text.ln_final
                    self.text_projection = text.text_projection
                    self.register_buffer("attn_mask", text.attn_mask, persistable=False)
        init_data = paddle.ones(shape=[1]) * np.log(1 / 0.07)
        self.logit_scale = self.create_parameter(
            shape=[1], default_initializer=paddle.nn.initializer.Assign(init_data)
        )
        self.custom_text = config.custom_text

        self.loss = ClipLoss(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            text_loss=True,
            rank=data_world_rank,
            world_size=data_world_size,
        )

        if enable_recompute:
            self.visual.set_grad_checkpointing(True)
            if not disable_text:
                self.transformer.enable_recompute = True

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.enable_recompute = enable

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
        if self.custom_text:
            features = self.text(text)
            return paddle.nn.functional.normalize(x=features, axis=-1) if normalize else features
        else:
            cast_dtype = self.transformer.get_cast_dtype()

            x = self.token_embedding(text).cast(cast_dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding.cast(cast_dtype)
            x = self.transformer(x, attn_mask=self.attn_mask)
            x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[paddle.arange(x.shape[0]), text.argmax(axis=-1)] @ self.text_projection
            return paddle.nn.functional.normalize(x, axis=-1) if normalize else x

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


    def clip_score(self, image, input_ids=None, text_emb=None, **kwargs):
        """
        Calculate the CLIP score (cosine similarity) between image and text embeddings.

        Args:
            image: The input image tensor.
            input_ids: Tokenized text input (optional).
            text_emb: Precomputed text embeddings (optional).
            **kwargs: Additional arguments.

        Returns:
            numpy.ndarray: The CLIP score (cosine similarity) between the image and text embeddings.
        """
        self.clip_scale()

        # Extract image features
        image_features = self.encode_image(image, normalize=True)
        text = input_ids

        # Extract text features
        if text is not None or text_emb is not None:
            text_features = self.encode_text(text, text_features=text_emb, normalize=True)
        else:
            print("Text input or embedding is empty, returning a score of 0.")
            return 0.0
        
        # Compute cosine similarity as the CLIP score
        clip_score = F.cosine_similarity(image_features, text_features)

        return clip_score.numpy()