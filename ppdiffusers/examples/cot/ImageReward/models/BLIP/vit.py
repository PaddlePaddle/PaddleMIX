# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from functools import partial

import paddle


def drop_path(x, drop_prob=0.0, training=False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob, dtype=x.dtype)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(paddle.nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(paddle.nn.Layer):
    """Image to Patch Embedding"""

    def __init__(self, img_size=[224, 224], patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        if isinstance(img_size, int):
            img_size = [img_size, img_size]
        self.num_patches_w = img_size[0] // patch_size
        self.num_patches_h = img_size[1] // patch_size

        num_patches = self.num_patches_w * self.num_patches_h
        self.patch_shape = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = paddle.nn.Conv2D(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    @property
    def num_patches_in_h(self):
        return self.img_size[1] // self.patch_size

    @property
    def num_patches_in_w(self):
        return self.img_size[0] // self.patch_size

    def forward(self, x, mask=None):
        B, C, H, W = x.shape
        return self.proj(x)


class Mlp(paddle.nn.Layer):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=paddle.nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = paddle.nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = act_layer()
        self.fc2 = paddle.nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = paddle.nn.Dropout(p=drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = paddle.nn.Linear(in_features=dim, out_features=dim * 3, bias_attr=qkv_bias)
        self.attn_drop = paddle.nn.Dropout(p=attn_drop)
        self.proj = paddle.nn.Linear(in_features=dim, out_features=dim)
        self.proj_drop = paddle.nn.Dropout(p=proj_drop)
        self.attn_gradients = None
        self.attention_map = None

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, register_hook=False):
        B, N, C = tuple(x.shape)
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads]).transpose(perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = q @ k.transpose((0, 1, 3, 2)) * self.scale
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(hook=self.save_attn_gradients)
        x = (attn @ v).transpose((0, 2, 1, 3)).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=paddle.nn.GELU,
        norm_layer=paddle.nn.LayerNorm,
        use_grad_checkpointing=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, register_hook=False):
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(paddle.nn.Layer):
    """Vision Transformer
    A Paddle impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        representation_size=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        use_grad_checkpointing=False,
        ckpt_layer=0,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(paddle.nn.LayerNorm, epsilon=1e-06)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches_in_h * self.patch_embed.num_patches_in_w

        self.cls_token = self.create_parameter(
            shape=(1, 1, embed_dim), default_initializer=paddle.nn.initializer.TruncatedNormal(std=0.02)
        )

        self.pos_embed = self.create_parameter(
            shape=(1, num_patches + 1, embed_dim), default_initializer=paddle.nn.initializer.TruncatedNormal(std=0.02)
        )

        self.pos_drop = paddle.nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in paddle.linspace(start=0, stop=drop_path_rate, num=depth)]
        self.blocks = paddle.nn.LayerList(
            sublayers=[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    use_grad_checkpointing=use_grad_checkpointing and i >= depth - ckpt_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward(self, x, register_blk=-1):
        B = tuple(x.shape)[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand((B, self.cls_token.shape[-2], self.cls_token.shape[-1]))
        x = x.flatten(2).transpose([0, 2, 1])  # b * hw * c
        x = paddle.concat(x=(cls_tokens, x), axis=1)
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.pos_drop(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, register_blk == i)
        x = self.norm(x)
        return x


def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):
    embedding_size = tuple(pos_embed_checkpoint.shape)[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    orig_size = int((tuple(pos_embed_checkpoint.shape)[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches**0.5)
    if orig_size != new_size:
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape([-1, orig_size, orig_size, embedding_size]).transpose(perm=[0, 3, 1, 2])
        pos_tokens = paddle.nn.functional.interpolate(
            x=pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.transpose(perm=[0, 2, 3, 1]).flatten(start_axis=1, stop_axis=2)
        new_pos_embed = paddle.concat(x=(extra_tokens, pos_tokens), axis=1)
        print("reshape position embedding from %d to %d" % (orig_size**2, new_size**2))
        return new_pos_embed
    else:
        return pos_embed_checkpoint
