import paddle
""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from functools import partial
import numpy as np
from .loss import ClipLoss
try:
    from .hf_model import HFTextEncoder
except:
    HFTextEncoder = None
from .timm_model import TimmModel
from .eva_vit_model import EVAVisionTransformer
from .transformer import LayerNorm, QuickGELU, Attention, VisionTransformer, EVATextTransformer
# from .fusedln import FusedLayerNorm
from paddle.nn import LayerNorm as FusedLayerNorm

from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.utils.log import logger
from .eva_vit_model import EVAVisionTransformerConfig
from .transformer import EVATextTransformerConfig


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
    def from_pretrained(cls,
                        pretrained_model_name_or_path: Union[str, os.PathLike],
                        **kwargs) -> "PretrainedConfig":
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path,
                                                  **kwargs)

        if ("model_type" in config_dict and hasattr(cls, "model_type") and
                config_dict["model_type"] != cls.model_type):
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
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
    def __init__(
            self,
            config,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            data_world_rank=0,
            data_world_size=1, ):
        super().__init__(config)
        vision_config = EVAVisionTransformerConfig(**config.vision_config)
        text_config = EVATextTransformerConfig(**config.text_config)
        self.visual = EVAVisionTransformer(vision_config)
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
        share_buffer = self.logit_scale.clip(0, 4.6052)
        self.logit_scale.copy_(share_buffer, True)

    def encode_image(self, image, normalize: bool=False):
        features = self.visual(image)
        out = paddle.nn.functional.normalize(
            x=features, axis=-1) if normalize else features
        return out

    def encode_text(self, text, text_features=None, normalize: bool=False):
        if text_features is not None:
            # directly use text_features if given
            return paddle.nn.functional.normalize(x=text_features, axis=-1) if normalize else text_features
        features = self.text(text)
        return paddle.nn.functional.normalize(
            x=features, axis=-1) if normalize else features

    def forward(self, image, input_ids, text_emb=None, skiploss=False):
        self.clip_scale()
        text = input_ids
        text_features = text_emb
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, text_features=text_features, normalize=True)
        if skiploss:
            return image_features, text_features, self.logit_scale.exp()

        loss_itc, logits_per_image, logits_per_text, labels = self.loss(
            (image_features, text_features, self.logit_scale.exp()))
        return loss_itc, image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: paddle.nn.Layer, dtype='float16'):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (paddle.nn.Conv1D, paddle.nn.Conv2D, paddle.nn.Linear,
                          fleet.meta_parallel.ColumnParallelLinear)):
            l.weight.data = l.weight.data.cast(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.cast(dtype)
        if isinstance(l, (paddle.nn.MultiHeadAttention, Attention)):
            for attr in [
                    * [f'{s}_proj_weight' for s in ['in', 'q', 'k', 'v']],
                    'in_proj_bias', 'bias_k', 'bias_v'
            ]:
                tensor = getattr(l, attr, None)
                if tensor is not None:
                    tensor.data = tensor.data.cast(dtype)
        else:
            l.data = l.data.cast(dtype)
        for name in ['text_projection', 'proj']:
            if hasattr(l, name) and isinstance(l, paddle.Tensor):
                attr = getattr(l, name, None)
                if attr is not None:
                    attr.data = attr.data.to(dtype)

    model.apply(fn=_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp


def convert_to_custom_text_state_dict(state_dict: dict):
    if 'text_projection' in state_dict:
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(
                    k.startswith(p)
                    for p in
                ('text_projection', 'positional_embedding', 'token_embedding',
                 'transformer', 'ln_final', 'logit_scale')):
                k = 'text.' + k
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict(state_dict: dict,
                                       quick_gelu=True,
                                       cast_dtype='float16'):
    vit = 'visual.proj' in state_dict
    if vit:
        vision_width = state_dict['visual.conv1.weight'].shape[0]
        vision_layers = len([
            k for k in state_dict.keys()
            if k.startswith('visual.') and k.endswith('.attn.in_proj_weight')
        ])
        vision_patch_size = state_dict['visual.conv1.weight'].shape[-1]
        grid_size = round((state_dict['visual.positional_embedding'].shape[0] -
                           1)**0.5)
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split('.')[2] for k in state_dict
                    if k.startswith(f'visual.layer{b}')))
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict['visual.layer1.0.conv1.weight'].shape[0]
        output_width = round((state_dict['visual.attnpool.positional_embedding']
                              .shape[0] - 1)**0.5)
        vision_patch_size = None
        assert output_width**2 + 1 == state_dict[
            'visual.attnpool.positional_embedding'].shape[0]
        image_size = output_width * 32
    embed_dim = state_dict['text_projection'].shape[1]
    context_length = state_dict['positional_embedding'].shape[0]
    vocab_size = state_dict['token_embedding.weight'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split('.')[2] for k in state_dict
            if k.startswith(f'transformer.resblocks')))
    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size)
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers)
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,
        cast_dtype=cast_dtype)
    for key in ['input_resolution', 'context_length', 'vocab_size']:
        state_dict.pop(key, None)
    convert_weights_to_fp16(model)
    model.set_state_dict(state_dict=state_dict)
    return model.eval()
