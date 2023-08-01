import sys
import paddle
""" Attention Pool 2D

Implementations of 2D spatial feature pooling using multi-head attention instead of average pool.

Based on idea in CLIP by OpenAI, licensed Apache 2.0
https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

Hacked together by / Copyright 2021 Ross Wightman
"""
from typing import Union, Tuple
from .timm_ext import to_2tuple
from .pos_embed_sincos import apply_rot_embed, RotaryEmbedding
from .timm_ext import trunc_normal_


class RotAttentionPool2d(paddle.nn.Layer):
    """ Attention based 2D feature pooling w/ rotary (relative) pos embedding.
    This is a multi-head attention based replacement for (spatial) average pooling in NN architectures.

    Adapted from the AttentionPool2d in CLIP w/ rotary embedding instead of learned embed.
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

    NOTE: While this impl does not require a fixed feature size, performance at differeing resolutions from
    train varies widely and falls off dramatically. I'm not sure if there is a way around this... -RW
    """

    def __init__(self,
                 in_features: int,
                 out_features: int=None,
                 embed_dim: int=None,
                 num_heads: int=4,
                 qkv_bias: bool=True):
        super().__init__()
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        self.qkv = paddle.nn.Linear(
            in_features=in_features,
            out_features=embed_dim * 3,
            bias_attr=qkv_bias)
        self.proj = paddle.nn.Linear(
            in_features=embed_dim, out_features=out_features)
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.pos_embed = RotaryEmbedding(self.head_dim)
        trunc_normal_(self.qkv.weight, std=in_features**-0.5)
        #         torch.nn.init.zeros_(self.qkv.bias)
        init_data = paddle.zeros(shape=self.qkv.bias.shape)
        self.qkv.bias = self.create_parameter(
            shape=self.qkv.bias.shape,
            default_initializer=paddle.nn.initializer.Assign(init_data))

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        x = x.reshape((B, -1, N)).transpose(perm=[0, 2, 1])
        x = paddle.concat(x=[x.mean(axis=1, keepdim=True), x], axis=1)
        x = self.qkv(x).reshape(
            (B, N + 1, 3, self.num_heads,
             self.head_dim)).transpose(perm=[2, 0, 3, 1, 4])
        q, k, v = x[0], x[1], x[2]
        qc, q = q[:, :, :1], q[:, :, 1:]
        sin_emb, cos_emb = self.pos_embed.get_embed((H, W))
        q = apply_rot_embed(q, sin_emb, cos_emb)
        q = paddle.concat(x=[qc, q], axis=2)
        kc, k = k[:, :, :1], k[:, :, 1:]
        k = apply_rot_embed(k, sin_emb, cos_emb)
        k = paddle.concat(x=[kc, k], axis=2)
        x = k
        perm_0 = list(range(x.ndim))
        perm_0[-2] = -1
        perm_0[-1] = -2
        attn = q @x.transpose(perm=perm_0) * self.scale
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        x = attn @v
        perm_1 = list(range(x.ndim))
        perm_1[1] = 2
        perm_1[2] = 1
        x = x.transpose(perm=perm_1).reshape((B, N + 1, -1))
        x = self.proj(x)
        return x[:, (0)]


class AttentionPool2d(paddle.nn.Layer):
    """ Attention based 2D feature pooling w/ learned (absolute) pos embedding.
    This is a multi-head attention based replacement for (spatial) average pooling in NN architectures.

    It was based on impl in CLIP by OpenAI
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py

    NOTE: This requires feature size upon construction and well prevent adaptive sizing of the network.
    """

    def __init__(self,
                 in_features: int,
                 feat_size: Union[int, Tuple[int, int]],
                 out_features: int=None,
                 embed_dim: int=None,
                 num_heads: int=4,
                 qkv_bias: bool=True):
        super().__init__()
        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.feat_size = to_2tuple(feat_size)
        self.qkv = paddle.nn.Linear(
            in_features=in_features,
            out_features=embed_dim * 3,
            bias_attr=qkv_bias)
        self.proj = paddle.nn.Linear(
            in_features=embed_dim, out_features=out_features)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        spatial_dim = self.feat_size[0] * self.feat_size[1]
        #         self.pos_embed = torch.nn.Parameter(paddle.zeros(shape=[spatial_dim + 1, in_features]))
        init_data = paddle.zeros(shape=[spatial_dim + 1, in_features])
        self.pos_embed = self.create_parameter(
            shape=[spatial_dim + 1, in_features],
            default_initializer=paddle.nn.initializer.Assign(init_data))
        trunc_normal_(self.pos_embed, std=in_features**-0.5)
        trunc_normal_(self.qkv.weight, std=in_features**-0.5)
        # torch.nn.init.zeros_(self.qkv.bias)
        init_data = paddle.zeros(shape=self.qkv.bias.shape)
        self.qkv.bias = self.create_parameter(
            shape=self.qkv.bias.shape,
            default_initializer=paddle.nn.initializer.Assign(init_data))

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        assert self.feat_size[0] == H
        assert self.feat_size[1] == W
        x = x.reshape((B, -1, N)).transpose(perm=[0, 2, 1])
        x = paddle.concat(x=[x.mean(axis=1, keepdim=True), x], axis=1)
        if isinstance(x.dtype, paddle.dtype):
            dtype = x.dtype
        elif isinstance(x.dtype,
                        str) and x.dtype not in ['cpu', 'cuda', 'ipu', 'xpu']:
            dtype = x.dtype
        elif isinstance(x.dtype, paddle.Tensor):
            dtype = x.dtype.dtype
        else:
            dtype = self.pos_embed.unsqueeze(axis=0).dtype
        x = x + self.pos_embed.unsqueeze(axis=0).cast(dtype)
        x = self.qkv(x).reshape(
            (B, N + 1, 3, self.num_heads,
             self.head_dim)).transpose(perm=[2, 0, 3, 1, 4])
        q, k, v = x[0], x[1], x[2]
        x = k
        perm_2 = list(range(x.ndim))
        perm_2[-2] = -1
        perm_2[-1] = -2
        attn = q @x.transpose(perm=perm_2) * self.scale
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        x = attn @v
        perm_3 = list(range(x.ndim))
        perm_3[1] = 2
        perm_3[2] = 1
        x = x.transpose(perm=perm_3).reshape((B, N + 1, -1))
        x = self.proj(x)
        return x[:, (0)]
