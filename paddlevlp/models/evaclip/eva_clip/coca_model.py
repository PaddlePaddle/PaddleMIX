from typing import Optional
""" CoCa Model

Adapted from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/coca_model.py
"""
import paddle
from paddle import nn
from paddle.nn import functional as F
import numpy as np
from dataclasses import dataclass
from .loss import ClipLoss, CoCaLoss
# Note: in models/loss.py not loss.py

from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    MultimodalTransformer, )
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower


@dataclass
class MultimodalCfg(CLIPTextCfg):
    mlp_ratio: int = 4
    dim_head: int = 64
    heads: int = 8
    n_queries: int = 256
    attn_pooler_heads: int = 8


def _build_text_decoder_tower(
        embed_dim,
        multimodal_cfg,
        quick_gelu: bool=False,
        cast_dtype: Optional[paddle.dtype]=None, ):
    multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(
        multimodal_cfg, dict) else multimodal_cfg
    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in ('float16', 'bfloat16'
                                                 ) else LayerNorm

    decoder = MultimodalTransformer(
        context_length=multimodal_cfg.context_length,
        width=multimodal_cfg.width,
        heads=multimodal_cfg.heads,
        layers=multimodal_cfg.layers,
        ls_init_value=multimodal_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer, )

    return decoder


class CoCa(nn.Layer):
    def __init__(
            self,
            args,  #
            embed_dim: int,
            multimodal_cfg: MultimodalCfg,
            text_cfg: CLIPTextCfg,
            vision_cfg: CLIPVisionCfg,
            quick_gelu: bool=False,
            cast_dtype: Optional[paddle.dtype]=None,
            pad_id: int=0, ):
        super().__init__()
        print("coca text cfg:", text_cfg)
        # multimodal_cfg = MultimodalCfg(**multimodal_cfg) if isinstance(multimodal_cfg, dict) else multimodal_cfg
        # text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        # vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg

        self.text = _build_text_tower(
            embed_dim=embed_dim,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype, )

        vocab_size = (
            text_cfg['vocab_size']  # for hf models
            if "hf_model_name" in text_cfg and
            text_cfg['hf_model_name'] is not None else text_cfg['vocab_size'])

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype, )

        self.text_decoder = _build_text_decoder_tower(
            vocab_size,
            multimodal_cfg=multimodal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype, )

        init_data = paddle.ones(shape=[1]) * np.log(1 / 0.07)
        self.logit_scale = self.create_parameter(
            shape=[1],
            default_initializer=paddle.nn.initializer.Assign(init_data))

        self.pad_id = pad_id
        self.loss = CoCaLoss(
            caption_loss_weight=args.coca_caption_loss_weight,
            clip_loss_weight=args.coca_contrastive_loss_weight,
            local_loss=False,  # args.local_loss,
            gather_with_grad=False,  # args.gather_with_grad,
            cache_labels=True,
            rank=args.data_world_rank,
            world_size=args.data_world_size, )

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
                text,
                embed_cls=True,
                image_latent=None,
                image_embs=None,
                type_ids=None):
        text_latent, token_embs = self._encode_text(text, embed_cls=embed_cls)
        if image_latent is None or image_embs is None:
            image_latent, image_embs = self._encode_image(image)

        # TODO: add assertion to avoid bugs?
        labels = text[:, -token_embs.shape[1]:]

        logits = self.text_decoder(image_embs, token_embs)
        return [
            image_latent, text_latent, self.logit_scale.exp(), logits, labels
        ]


def prepare_inputs_for_generation(input_ids, image_inputs, past=None,
                                  **kwargs):
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.astype(dtype='int64').cumsum(axis=-1) - 1
        #position_ids.masked_fill_(attention_mask == 0, 1)
        mask = attention_mask == 0
        position_ids = paddle.where(mask > 0, mask, paddle.full_like(mask, 1))
    else:
        position_ids = None
    return {
        "text": input_ids,
        "images": image_inputs,
        "past_key_values": past,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }
