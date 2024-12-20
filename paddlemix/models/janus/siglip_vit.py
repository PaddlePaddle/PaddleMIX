# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import collections
import math
import os
import warnings
from dataclasses import dataclass
from enum import Enum
from functools import partial
from itertools import repeat
from typing import (
    Callable,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import paddle
from paddle.nn.initializer import TruncatedNormal

trunc_normal_ = TruncatedNormal(std=0.02)

__all__ = [
    "SigLIPVisionCfg",
    "SigLIPVisionTransformer",
]


def drop_path(x, drop_prob=0.0, training=False):

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob, dtype=x.dtype)
    shape = (paddle.shape(x)[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape).astype(x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Format(str, Enum):
    NCHW = "NCHW"
    NHWC = "NHWC"
    NCL = "NCL"
    NLC = "NLC"


class PatchEmbed(paddle.nn.Layer):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        output_fmt: Optional[str] = None,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)
        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad
        self.proj = paddle.nn.Conv2D(
            in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias_attr=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else paddle.nn.Identity()

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        if img_size is None:
            return None, None, None
        img_size = to_2tuple(img_size)
        grid_size = tuple([(s // p) for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def set_input_size(
        self,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        new_patch_size = None
        if patch_size is not None:
            new_patch_size = to_2tuple(patch_size)

        img_size = img_size or self.img_size
        if img_size != self.img_size or new_patch_size is not None:
            self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(img_size[1] / self.patch_size[1])
        else:
            return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]

    def forward(self, x):
        B, C, H, W = tuple(x.shape)
        if self.img_size is not None:
            pass

        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = paddle.nn.functional.pad(x=x, pad=(0, pad_w, 0, pad_h), pad_from_left_axis=False)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(start_axis=2).transpose([0, 2, 1])

        x = self.norm(x)
        return x


def resample_abs_pos_embed(
    posemb: paddle.Tensor,
    new_size: List[int],
    old_size: Optional[List[int]] = None,
    num_prefix_tokens: int = 1,
    interpolation: str = "bicubic",
    antialias: bool = True,
    verbose: bool = False,
):
    num_pos_tokens = tuple(posemb.shape)[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb
    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw
    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb
    embed_dim = tuple(posemb.shape)[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.astype(dtype="float32")
    posemb = posemb.reshape([1, old_size[0], old_size[1], -1]).transpose(perm=[0, 3, 1, 2])
    posemb = paddle.nn.functional.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.transpose(perm=[0, 2, 3, 1]).reshape([1, -1, embed_dim])
    posemb = posemb.to(orig_dtype)
    if posemb_prefix is not None:
        posemb = paddle.concat(x=[posemb_prefix, posemb], axis=1)
    # if not torch.jit.is_scripting() and verbose:
    #     _logger.info(f'Resized position embedding: {old_size} to {new_size}.')
    return posemb


def named_apply(
    fn: Callable, module: paddle.nn.Layer, name="", depth_first: bool = True, include_root: bool = False
) -> paddle.nn.Layer:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with paddle.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(min=2 * l - 1, max=2 * u - 1)
        tensor.erfinv_()
        tensor.multiply_(y=paddle.to_tensor(std * math.sqrt(2.0)))
        tensor.add_(y=paddle.to_tensor(mean))
        tensor.clip_(min=a, max=b)
        return tensor


class Mlp(paddle.nn.Layer):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=paddle.nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(paddle.nn.Conv2D, kernel_size=1) if use_conv else paddle.nn.Linear
        self.fc1 = linear_layer(in_features, hidden_features, bias_attr=bias[0])
        self.act = act_layer()
        self.drop1 = paddle.nn.Dropout(p=drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else paddle.nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias_attr=bias[1])
        self.drop2 = paddle.nn.Dropout(p=drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class AttentionPoolLatent(paddle.nn.Layer):
    """Attention pooling w/ latent query"""

    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        embed_dim: int = None,
        num_heads: int = 8,
        feat_size: Optional[int] = None,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        latent_len: int = 1,
        latent_dim: int = None,
        pos_embed: str = "",
        pool_type: str = "token",
        norm_layer: Optional[paddle.nn.Layer] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.feat_size = feat_size
        self.scale = self.head_dim**-0.5
        self.pool = pool_type
        self.fused_attn = False
        if pos_embed == "abs":
            assert feat_size is not None
            self.pos_embed = paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.zeros(shape=[feat_size, in_features])
            )
        else:
            self.pos_embed = None
        self.latent_dim = latent_dim or embed_dim
        self.latent_len = latent_len
        self.latent = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.zeros(shape=[1, self.latent_len, embed_dim])
        )
        self.q = paddle.nn.Linear(in_features=embed_dim, out_features=embed_dim, bias_attr=qkv_bias)
        self.kv = paddle.nn.Linear(in_features=embed_dim, out_features=embed_dim * 2, bias_attr=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else paddle.nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else paddle.nn.Identity()
        self.proj = paddle.nn.Linear(in_features=embed_dim, out_features=embed_dim)
        # self.attn_drop = paddle.nn.Dropout(p=drop)
        self.proj_drop = paddle.nn.Dropout(p=drop)
        self.norm = norm_layer(out_features) if norm_layer is not None else paddle.nn.Identity()
        self.mlp = Mlp(embed_dim, int(embed_dim * mlp_ratio))
        # self.init_weights()

    # def init_weights(self):
    #     if self.pos_embed is not None:
    #         trunc_normal_(self.pos_embed, std=tuple(self.pos_embed.shape)[1] ** -0.5)
    #     trunc_normal_(self.latent, std=self.latent_dim**-0.5)

    def forward(self, x):
        B, N, C = tuple(x.shape)
        if self.pos_embed is not None:
            x = x + self.pos_embed.unsqueeze(axis=0).to(x.dtype)
        q_latent = self.latent.expand(shape=[B, -1, -1])
        q = self.q(q_latent).reshape([B, self.latent_len, self.num_heads, self.head_dim]).transepose(perm=[0, 2, 1, 3])
        kv = self.kv(x).reshape([B, N, 2, self.num_heads, self.head_dim]).transpose(perm=[2, 0, 3, 1, 4])
        k, v = kv.unbind(axis=0)
        q, k = self.q_norm(q), self.k_norm(k)

        # k_v_seq_len = k.shape[-2]
        attn_weights = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * self.scale
        attn_weights = paddle.nn.functional.softmax(attn_weights, axis=-1)
        # attn_weights = self.attn_drop(attn_weights)
        x = paddle.matmul(attn_weights, v)

        x = x.transpose([0, 2, 1]).contiguous()
        x = x.reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x + self.mlp(self.norm(x))
        if self.pool == "token":
            x = x[:, 0]
        elif self.pool == "avg":
            x = x.mean(axis=1)
        return x


class DropPath(paddle.nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class PatchDropout(paddle.nn.Layer):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.0
        self.prob = prob
        self.exclude_first_token = exclude_first_token
        print(f"os.getenv('RoPE')={os.getenv('RoPE')}")

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x
        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        batch = x.shape[0]
        num_tokens = x.shape[1]
        batch_indices = paddle.arange(end=batch)
        batch_indices = batch_indices[..., None]
        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))
        rand = paddle.randn(shape=[batch, num_tokens])
        patch_indices_keep = rand.topk(k=num_patches_keep, axis=-1).indices
        x = x[batch_indices, patch_indices_keep]
        if self.exclude_first_token:
            x = paddle.concat(x=(cls_tokens, x), axis=1)
        if self.training and os.getenv("RoPE") == "1":
            return x, patch_indices_keep
        return x


def init_weights(self):
    if self.pos_embed is not None:
        trunc_normal_(self.pos_embed, std=tuple(self.pos_embed.shape)[1] ** -0.5)
    trunc_normal_(self.latent, std=self.latent_dim**-0.5)


def init_weights_vit_timm(module: paddle.nn.Layer, name: str = "") -> None:
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, paddle.nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            init_Constant = paddle.nn.initializer.Constant(value=0.0)
            init_Constant(module.bias)
    elif hasattr(module, "init_weights"):
        module.init_weights()


class Attention(paddle.nn.Layer):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: paddle.nn.Layer = paddle.nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = False
        self.qkv = paddle.nn.Linear(in_features=dim, out_features=dim * 3, bias_attr=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else paddle.nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else paddle.nn.Identity()
        self.attn_drop = paddle.nn.Dropout(p=attn_drop)
        self.proj = paddle.nn.Linear(in_features=dim, out_features=dim)
        self.proj_drop = paddle.nn.Dropout(p=proj_drop) if proj_drop > 0.0 else paddle.nn.Identity()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        B, N, C = tuple(x.shape)
        qkv = (
            self.qkv(x).reshape([B, N, 3, self.num_heads, self.head_dim]).transpose(perm=[2, 0, 3, 1, 4])
        )  # 3 b H N D
        q, k, v = qkv.unbind(axis=0)
        q, k = self.q_norm(q), self.k_norm(k)
        # k_v_seq_len = k.shape[-2]
        attn_weights = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * self.scale
        attn_weights = paddle.nn.functional.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_drop(attn_weights)
        x = paddle.matmul(attn_weights, v)
        x = x.transpose([0, 2, 1]).contiguous()
        x = x.reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(paddle.nn.Layer):
    def __init__(self, dim: int, init_values: float = 1e-05, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = paddle.base.framework.EagerParamBase.from_tensor(tensor=init_values * paddle.ones(shape=dim))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return x.multiply_(y=paddle.to_tensor(self.gamma)) if self.inplace else x * self.gamma


class Block(paddle.nn.Layer):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: paddle.nn.Layer = paddle.nn.GELU,
        norm_layer: paddle.nn.Layer = paddle.nn.LayerNorm,
        mlp_layer: paddle.nn.Layer = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else paddle.nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else paddle.nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        # import pdb;pdb.set_trace()
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class SigLIPVisionTransformer(paddle.nn.Layer):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    dynamic_img_size: Final[bool]

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[paddle.nn.Layer] = None,
        act_layer: Optional[paddle.nn.Layer] = None,
        block_fn: Type[paddle.nn.Layer] = Block,
        mlp_layer: Type[paddle.nn.Layer] = Mlp,
        ignore_head: bool = False,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ("", "avg", "token", "map")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm
        norm_layer = partial(paddle.nn.LayerNorm, epsilon=1e-06)
        act_layer = paddle.nn.GELU
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False
        self.ignore_head = ignore_head
        embed_args = {}
        if dynamic_img_size:
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = (
            paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.zeros(shape=[1, 1, embed_dim]))
            if class_token
            else None
        )
        self.reg_token = (
            paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.zeros(shape=[1, reg_tokens, embed_dim]))
            if reg_tokens
            else None
        )
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.randn(shape=[1, embed_len, embed_dim]) * 0.02
        )
        self.pos_drop = paddle.nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(patch_drop_rate, num_prefix_tokens=self.num_prefix_tokens)
        else:
            self.patch_drop = paddle.nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else paddle.nn.Identity()
        dpr = [x.item() for x in paddle.linspace(start=0, stop=drop_path_rate, num=depth)]
        self.blocks = paddle.nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim) if not use_fc_norm else paddle.nn.Identity()
        if global_pool == "map":
            AttentionPoolLatent.init_weights = init_weights
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, norm_layer=norm_layer
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else paddle.nn.Identity()
        self.head_drop = paddle.nn.Dropout(p=drop_rate)
        self.head = (
            paddle.nn.Linear(in_features=self.embed_dim, out_features=num_classes)
            if num_classes > 0
            else paddle.nn.Identity()
        )
        if weight_init != "skip":
            self.init_weights(weight_init)

    def init_weights(self, mode: Literal["jax", "jax_nlhb", "moco", ""] = "") -> None:
        assert mode in ("jax", "jax_nlhb", "moco", "")
        trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            init_Normal = paddle.nn.initializer.Normal(std=1e-06)
            init_Normal(self.cls_token)
        named_apply(init_weights_vit_timm, self)

    def no_weight_decay(self) -> Set:
        return {"pos_embed", "cls_token", "dist_token"}

    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(stem="^cls_token|pos_embed|patch_embed", blocks=[("^blocks\\.(\\d+)", None), ("^norm", (99999,))])

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    def get_classifier(self) -> paddle.nn.Layer:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "token", "map")
            if global_pool == "map" and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != "map " and self.attn_pool is not None:
                self.attn_pool = None
            self.global_pool = global_pool
        self.head = (
            paddle.nn.Linear(in_features=self.embed_dim, out_features=num_classes)
            if num_classes > 0
            else paddle.nn.Identity()
        )

    def _pos_embed(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.dynamic_img_size:
            B, H, W, C = tuple(x.shape)
            pos_embed = resample_abs_pos_embed(
                self.pos_embed, (H, W), num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed
        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(shape=[tuple(x.shape)[0], -1, -1]))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(shape=[tuple(x.shape)[0], -1, -1]))
        if self.no_embed_class:
            x = x + pos_embed
            if to_cat:
                x = paddle.concat(x=to_cat + [x], axis=1)
        else:
            if to_cat:
                x = paddle.concat(x=to_cat + [x], axis=1)
            x = x + pos_embed
        return self.pos_drop(x)

    def _intermediate_layers(self, x: paddle.Tensor, n: Union[int, Sequence] = 1) -> List[paddle.Tensor]:
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)
        return outputs

    def get_intermediate_layers(
        self,
        x: paddle.Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ) -> Tuple[Union[paddle.Tensor, Tuple[paddle.Tensor]]]:
        """Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens :] for out in outputs]
        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(tuple(x.shape)[0], grid_size[0], grid_size[1], -1)
                .transpose(perm=[0, 3, 1, 2])
                .contiguous()
                for out in outputs
            ]
        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def forward_features(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing:
            x = paddle.distributed.fleet.utils.recompute(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: paddle.Tensor, pre_logits: bool = False) -> paddle.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens :].mean(axis=1)
        elif self.global_pool:
            x = x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.forward_features(x)
        if not self.ignore_head:
            x = self.forward_head(x)
        return x


@dataclass
class SigLIPVisionCfg:
    width: int = 1152
    layers: Union[Tuple[int, int, int, int], int] = 27
    heads: int = 16
    patch_size: int = 14
    image_size: Union[Tuple[int, int], int] = 336
    global_pool: str = "map"
    mlp_ratio: float = 3.7362
    class_token: bool = False
    num_classes: int = 0
    use_checkpoint: bool = False
