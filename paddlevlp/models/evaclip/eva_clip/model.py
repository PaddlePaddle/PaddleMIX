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
from .modified_resnet import ModifiedResNet
from .timm_model import TimmModel
from .eva_vit_model import EVAVisionTransformer
from .transformer import LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer
from .fusedln import FusedLayerNorm
# from paddle.nn import LayerNorm as FusedLayerNorm


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None
    patch_dropout: float = 0.0
    global_average_pool: bool = False
    drop_path_rate: Optional[float] = None
    timm_model_name: str = None
    timm_model_pretrained: bool = False
    timm_pool: str = 'avg'
    timm_proj: str = 'linear'
    timm_proj_bias: bool = False
    eva_model_name: str = None
    qkv_bias: bool = True
    fusedLN: bool = False
    xattn: bool = False
    postnorm: bool = False
    rope: bool = False
    pt_hw_seq_len: int = 16
    intp_freq: bool = False
    naiveswiglu: bool = False
    subln: bool = False
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer
    n_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    output_tokens: bool = False


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'
    masked_language_modeling: bool = False
    fusedLN: bool = False
    xattn: bool = False
    attn_mask: bool = True
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = 'bfloat16'
    elif precision == 'fp16':
        cast_dtype = 'float16'
    return cast_dtype


def _build_vision_tower(embed_dim: int,
                        vision_cfg: CLIPVisionCfg,
                        quick_gelu: bool=False,
                        cast_dtype: Optional[paddle.dtype]=None):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)
    act_layer = QuickGELU if quick_gelu else paddle.nn.GELU
    if vision_cfg.eva_model_name:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNorm
        visual = EVAVisionTransformer(
            img_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            num_classes=embed_dim,
            use_mean_pooling=vision_cfg.global_average_pool,
            init_values=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            embed_dim=vision_cfg.width,
            depth=vision_cfg.layers,
            num_heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            qkv_bias=vision_cfg.qkv_bias,
            drop_path_rate=vision_cfg.drop_path_rate,
            norm_layer=partial(
                FusedLayerNorm, epsilon=1e-6)
            if vision_cfg.fusedLN else partial(
                norm_layer, epsilon=1e-6),
            xattn=vision_cfg.xattn,
            rope=vision_cfg.rope,
            postnorm=vision_cfg.postnorm,
            pt_hw_seq_len=vision_cfg.pt_hw_seq_len,
            intp_freq=vision_cfg.intp_freq,
            naiveswiglu=vision_cfg.naiveswiglu,
            subln=vision_cfg.subln,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens, )
    elif vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size)
        act_layer = paddle.nn.GELU
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width)
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in ('float16', 'bfloat16'
                                                     ) else LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer)
    return visual


def _build_text_tower(embed_dim: int,
                      text_cfg: CLIPTextCfg,
                      quick_gelu: bool=False,
                      cast_dtype: Optional[paddle.dtype]=None):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)
    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            tokenizer_name=text_cfg.hf_tokenizer_name,
            proj=text_cfg.proj,
            pooler_type=text_cfg.pooler_type,
            masked_language_modeling=text_cfg.masked_language_modeling)
    else:
        act_layer = QuickGELU if quick_gelu else paddle.nn.GELU
        norm_layer = LayerNorm
        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=FusedLayerNorm if text_cfg.fusedLN else norm_layer,
            xattn=text_cfg.xattn,
            attn_mask=text_cfg.attn_mask,
            embed_cls=text_cfg.embed_cls,
            pad_id=text_cfg.pad_id,
            output_tokens=text_cfg.output_tokens)
    return text


class CLIP(paddle.nn.Layer):
    def __init__(self,
                 args,
                 embed_dim: int,
                 vision_cfg: CLIPVisionCfg,
                 text_cfg: CLIPTextCfg,
                 quick_gelu: bool=False,
                 cast_dtype: Optional[paddle.dtype]=None):
        super().__init__()
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu,
                                          cast_dtype)
        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistable=False)
        init_data = paddle.ones(shape=[1]) * np.log(1 / 0.07)
        self.logit_scale = self.create_parameter(
            shape=[1],
            default_initializer=paddle.nn.initializer.Assign(init_data))

        self.loss = ClipLoss(
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=args.data_world_rank,
            world_size=args.data_world_size, )

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        self.visual.lock(
            unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def no_weight_decay(self):
        return {'logit_scale'}

    def encode_image(self, image, normalize: bool=False):
        features = self.visual(image)
        return paddle.nn.functional.normalize(
            x=features, axis=-1) if normalize else features

    def encode_text(self, text, normalize: bool=False):
        cast_dtype = self.transformer.get_cast_dtype()
        if isinstance(cast_dtype, paddle.dtype):
            dtype = cast_dtype
        elif isinstance(
                cast_dtype,
                str) and cast_dtype not in ['cpu', 'cuda', 'ipu', 'xpu']:
            dtype = cast_dtype
        elif isinstance(cast_dtype, paddle.Tensor):
            dtype = cast_dtype.dtype
        else:
            dtype = self.token_embedding(text).dtype
        x = self.token_embedding(text).cast(dtype)
        if isinstance(cast_dtype, paddle.dtype):
            dtype = cast_dtype
        elif isinstance(
                cast_dtype,
                str) and cast_dtype not in ['cpu', 'cuda', 'ipu', 'xpu']:
            dtype = cast_dtype
        elif isinstance(cast_dtype, paddle.Tensor):
            dtype = cast_dtype.dtype
        else:
            dtype = self.positional_embedding.dtype
        x = x + self.positional_embedding.cast(dtype)
        x = x.transpose(perm=[1, 0, 2])
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.transpose(perm=[1, 0, 2])
        x = self.ln_final(x)
        x = x[paddle.arange(end=x.shape[0]), text.argmax(
            axis=-1)] @self.text_projection
        return paddle.nn.functional.normalize(x=x, axis=-1) if normalize else x

    def forward(self, image, text, type_ids=None):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        # return [image_features, text_features, self.logit_scale.exp()]

        loss_itc, logits_per_image, logits_per_text, labels = self.loss(
            (image_features, text_features, self.logit_scale.exp()))
        return loss_itc, image_features, text_features, self.logit_scale.exp()


class CustomCLIP(paddle.nn.Layer):
    def __init__(self,
                 args,
                 embed_dim: int,
                 vision_cfg: CLIPVisionCfg,
                 text_cfg: CLIPTextCfg,
                 quick_gelu: bool=False,
                 cast_dtype: Optional[paddle.dtype]=None,
                 itm_task: bool=False):
        super().__init__()
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu,
                                          cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu,
                                      cast_dtype)
        init_data = paddle.ones(shape=[1]) * np.log(1 / 0.07)
        self.logit_scale = self.create_parameter(
            shape=[1],
            default_initializer=paddle.nn.initializer.Assign(init_data))

        self.loss = ClipLoss(
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=args.data_world_rank,
            world_size=args.data_world_size, )
        # print("args.data_world_rank:{}, args.data_world_size:{}".format(args.data_world_rank, args.data_world_size))

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

    def encode_text(self, text, normalize: bool=False):
        features = self.text(text)
        return paddle.nn.functional.normalize(
            x=features, axis=-1) if normalize else features
        # return features

    def forward(self, image, input_ids, skiploss=False):
        self.clip_scale()
        text = input_ids
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
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
        """Class Method: *.split, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
        counts: list = [
            len(
                set(
                    k.split('.')[2] for k in state_dict
                    if k.startswith(f'visual.layer{b}')))
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict['visual.layer1.0.conv1.weight'].shape[0]
        output_width = round((state_dict[
            'visual.attnpool.positional_embedding'].shape[0] - 1)**0.5)
        vision_patch_size = None
        assert output_width**2 + 1 == state_dict[
            'visual.attnpool.positional_embedding'].shape[0]
        image_size = output_width * 32
    embed_dim = state_dict['text_projection'].shape[1]
    context_length = state_dict['positional_embedding'].shape[0]
    vocab_size = state_dict['token_embedding.weight'].shape[0]
    transformer_width = state_dict['ln_final.weight'].shape[0]
    transformer_heads = transformer_width // 64
    """Class Method: *.split, not convert, please check whether it is torch.Tensor.*/Optimizer.*/nn.Module.*, and convert manually"""
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
