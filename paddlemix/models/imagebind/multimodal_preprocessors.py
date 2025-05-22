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

import html
import math
from functools import lru_cache
from typing import Callable, List, Optional, Tuple

import ftfy
import numpy as np
import paddle
import regex as re

from .helpers import VerboseNNModule, cast_if_src_dtype


def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table"""

    def get_position_angle_vec(position):
        return [(position / np.power(10000, 2 * (hid_j // 2) / d_hid)) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return paddle.to_tensor(data=sinusoid_table, dtype="float32").unsqueeze(axis=0)


def interpolate_pos_encoding_2d(target_spatial_size, pos_embed):
    N = pos_embed.shape[1]
    if N == target_spatial_size:
        return pos_embed
    dim = pos_embed.shape[-1]
    pos_embed, updated = cast_if_src_dtype(pos_embed, "bfloat16", "float32")
    pos_embed = paddle.nn.functional.interpolate(
        x=pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).transpose(perm=[0, 3, 1, 2]),
        scale_factor=math.sqrt(target_spatial_size / N),
        mode="bicubic",
    )
    if updated:
        pos_embed, _ = cast_if_src_dtype(pos_embed, "float32", "bfloat16")

    # pos_embed = pos_embed.transpose(perm=[0, 2, 3, 1]).view(1, -1, dim)
    pos_embed = pos_embed.transpose(perm=[0, 2, 3, 1]).reshape((1, -1, dim))
    return pos_embed


def interpolate_pos_encoding(npatch_per_img, pos_embed, patches_layout, input_shape=None, first_patch_idx=1):
    assert first_patch_idx == 0 or first_patch_idx == 1, "there is 1 CLS token or none"
    N = pos_embed.shape[1] - first_patch_idx
    if npatch_per_img == N:
        return pos_embed
    assert patches_layout[-1] == patches_layout[-2], "Interpolation of pos embed not supported for non-square layouts"
    class_emb = pos_embed[:, :first_patch_idx]
    pos_embed = pos_embed[:, first_patch_idx:]
    if input_shape is None or patches_layout[0] == 1:
        pos_embed = interpolate_pos_encoding_2d(npatch_per_img, pos_embed)
    elif patches_layout[0] > 1:
        assert len(input_shape) == 4, "temporal interpolation not supported"
        num_frames = patches_layout[0]
        num_spatial_tokens = patches_layout[1] * patches_layout[2]

        # pos_embed = pos_embed.view(1, num_frames, num_spatial_tokens, -1)
        pos_embed = pos_embed.reshape((1, num_frames, num_spatial_tokens, -1))
        pos_embed = interpolate_pos_encoding_2d(npatch_per_img, pos_embed[0, 0, ...].unsqueeze(axis=0))
    else:
        raise ValueError("This type of interpolation isn't implemented")
    return paddle.concat(x=(class_emb, pos_embed), axis=1)


def _get_pos_embedding(npatch_per_img, pos_embed, patches_layout, input_shape, first_patch_idx=1):
    pos_embed = interpolate_pos_encoding(
        npatch_per_img,
        pos_embed,
        patches_layout,
        input_shape=input_shape,
        first_patch_idx=first_patch_idx,
    )
    return pos_embed


class PatchEmbedGeneric(paddle.nn.Layer):
    """
    PatchEmbed from Hydra
    """

    def __init__(self, proj_stem, norm_layer: Optional[paddle.nn.Layer] = None):
        super().__init__()
        if len(proj_stem) > 1:
            self.proj = paddle.nn.Sequential(*proj_stem)
        else:
            self.proj = proj_stem[0]
        self.norm_layer = norm_layer

    def get_patch_layout(self, img_size):
        with paddle.no_grad():
            dummy_img = paddle.zeros(shape=[1] + img_size)
            dummy_out = self.proj(dummy_img)
        embed_dim = dummy_out.shape[1]
        patches_layout = tuple(dummy_out.shape[2:])
        num_patches = np.prod(patches_layout)
        return patches_layout, num_patches, embed_dim

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(start_axis=2)
        perm_1 = list(range(x.ndim))
        perm_1[1] = 2
        perm_1[2] = 1
        x = x.transpose(perm=perm_1)
        if self.norm_layer is not None:
            x = self.norm_layer(x)
        return x


class SpatioTemporalPosEmbeddingHelper(VerboseNNModule):
    def __init__(
        self,
        patches_layout: List,
        num_patches: int,
        num_cls_tokens: int,
        embed_dim: int,
        learnable: bool,
    ) -> None:
        super().__init__()
        self.num_cls_tokens = num_cls_tokens
        self.patches_layout = patches_layout
        self.num_patches = num_patches
        self.num_tokens = num_cls_tokens + num_patches
        self.learnable = learnable
        if self.learnable:

            self.pos_embed = paddle.create_parameter(
                shape=[1, self.num_tokens, embed_dim],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(value=0.0),
            )
            paddle.nn.initializer.TruncatedNormal(std=0.02)(self.pos_embed)

        else:
            self.register_buffer("pos_embed", get_sinusoid_encoding_table(self.num_tokens, embed_dim))

    def get_pos_embedding(self, vision_input, all_vision_tokens):
        input_shape = vision_input.shape
        pos_embed = _get_pos_embedding(
            all_vision_tokens.shape[1] - self.num_cls_tokens,
            pos_embed=self.pos_embed,
            patches_layout=self.patches_layout,
            input_shape=input_shape,
            first_patch_idx=self.num_cls_tokens,
        )
        return pos_embed


class RGBDTPreprocessor(VerboseNNModule):
    def __init__(
        self,
        rgbt_stem: PatchEmbedGeneric,
        depth_stem: Optional[PatchEmbedGeneric],
        img_size: Tuple = (3, 224, 224),
        num_cls_tokens: int = 1,
        pos_embed_fn: Optional[Callable] = None,
        use_type_embed: bool = False,
        init_param_style: str = "openclip",
    ) -> None:
        super().__init__()
        stem = rgbt_stem if rgbt_stem is not None else depth_stem
        self.patches_layout, self.num_patches, self.embed_dim = stem.get_patch_layout(img_size)
        self.rgbt_stem = rgbt_stem
        self.depth_stem = depth_stem
        self.use_pos_embed = pos_embed_fn is not None
        self.use_type_embed = use_type_embed
        self.num_cls_tokens = num_cls_tokens
        if self.use_pos_embed:
            self.pos_embedding_helper = pos_embed_fn(
                patches_layout=self.patches_layout,
                num_cls_tokens=num_cls_tokens,
                num_patches=self.num_patches,
                embed_dim=self.embed_dim,
            )
        if self.num_cls_tokens > 0:

            self.cls_token = paddle.create_parameter(
                shape=[1, self.num_cls_tokens, self.embed_dim],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(value=0.0),
            )
        if self.use_type_embed:

            self.type_embed = paddle.create_parameter(
                shape=[1, 1, self.embed_dim],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(value=0.0),
            )
        self.init_parameters(init_param_style)

    @paddle.no_grad()
    def init_parameters(self, init_param_style):
        if init_param_style == "openclip":
            scale = self.embed_dim**-0.5
            if self.use_pos_embed:
                paddle.nn.initializer.Normal()(self.pos_embedding_helper.pos_embed)

                self.pos_embedding_helper.pos_embed.set_value(self.pos_embedding_helper.pos_embed * scale)
            if self.num_cls_tokens > 0:
                paddle.nn.initializer.Normal()(self.cls_token)

                self.cls_token.set_value(self.cls_token * scale)
        elif init_param_style == "vit":
            self.cls_token.data.fill_(value=0)
        else:
            raise ValueError(f"Unknown init {init_param_style}")
        if self.use_type_embed:
            paddle.nn.initializer.Normal()(self.type_embed)

    def tokenize_input_and_cls_pos(self, input, stem, mask):
        tokens = stem(input)
        assert tokens.ndim == 3
        assert tokens.shape[2] == self.embed_dim
        B = tokens.shape[0]
        if self.num_cls_tokens > 0:
            class_tokens = self.cls_token.expand(shape=[B, -1, -1])
            tokens = paddle.concat(x=(class_tokens, tokens), axis=1)
        if self.use_pos_embed:
            pos_embed = self.pos_embedding_helper.get_pos_embedding(input, tokens)
            tokens = tokens + pos_embed
        if self.use_type_embed:
            tokens = tokens + self.type_embed.expand(shape=[B, -1, -1])
        return tokens

    def forward(self, vision=None, depth=None, patch_mask=None):
        if patch_mask is not None:
            raise NotImplementedError()
        if vision is not None:
            vision_tokens = self.tokenize_input_and_cls_pos(vision, self.rgbt_stem, patch_mask)
        if depth is not None:
            depth_tokens = self.tokenize_input_and_cls_pos(depth, self.depth_stem, patch_mask)
        if vision is not None and depth is not None:
            final_tokens = vision_tokens + depth_tokens
        else:
            final_tokens = vision_tokens if vision is not None else depth_tokens
        return_dict = {"trunk": {"tokens": final_tokens}, "head": {}}
        return return_dict


class AudioPreprocessor(RGBDTPreprocessor):
    def __init__(self, audio_stem: PatchEmbedGeneric, **kwargs) -> None:
        super().__init__(rgbt_stem=audio_stem, depth_stem=None, **kwargs)

    def forward(self, audio=None):
        return super().forward(vision=audio)


class ThermalPreprocessor(RGBDTPreprocessor):
    def __init__(self, thermal_stem: PatchEmbedGeneric, **kwargs) -> None:
        super().__init__(rgbt_stem=thermal_stem, depth_stem=None, **kwargs)

    def forward(self, thermal=None):
        return super().forward(vision=thermal)


def build_causal_attention_mask(context_length):
    out_0 = paddle.empty(shape=[context_length, context_length])
    out_0.stop_gradient = not False
    mask = out_0
    mask.fill_(value=float("-inf"))
    mask = paddle.triu(mask, 1)
    return mask


class TextPreprocessor(VerboseNNModule):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int,
        causal_masking: bool,
        supply_seq_len_to_head: bool = True,
        num_cls_tokens: int = 0,
        init_param_style: str = "openclip",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embedding = paddle.nn.Embedding(vocab_size, embed_dim)

        self.pos_embed = paddle.create_parameter(
            shape=[1, self.context_length + num_cls_tokens, embed_dim],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Assign(
                value=paddle.empty(shape=[1, self.context_length + num_cls_tokens, embed_dim])
            ),
        )
        self.causal_masking = causal_masking
        if self.causal_masking:
            mask = build_causal_attention_mask(self.context_length)
            self.register_buffer("mask", mask)
        self.supply_seq_len_to_head = supply_seq_len_to_head
        self.num_cls_tokens = num_cls_tokens
        self.embed_dim = embed_dim
        if num_cls_tokens > 0:
            assert self.causal_masking is False, "Masking + CLS token isn't implemented"

            self.cls_token = paddle.create_parameter(
                shape=[1, self.num_cls_tokens, embed_dim],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(value=0.0),
            )
        self.init_parameters(init_param_style)

    @paddle.no_grad()
    def init_parameters(self, init_param_style="openclip"):
        paddle.nn.initializer.Normal(std=0.02)(self.token_embedding.weight)
        paddle.nn.initializer.Normal(std=0.01)(self.pos_embed)

        if init_param_style == "openclip":
            scale = self.embed_dim**-0.5
            if self.num_cls_tokens > 0:
                paddle.nn.initializer.Normal()(self.cls_token)

                self.cls_token.set_value(self.cls_token * scale)
        elif init_param_style == "vit":
            self.cls_token.data.fill_(value=0)
        else:
            raise ValueError(f"Unknown init {init_param_style}")

    def forward(self, text):
        text_tokens = self.token_embedding(text)
        if self.num_cls_tokens > 0:
            B = text_tokens.shape[0]
            class_tokens = self.cls_token.expand(shape=[B, -1, -1])
            text_tokens = paddle.concat(x=(class_tokens, text_tokens), axis=1)
        text_tokens = text_tokens + self.pos_embed
        return_dict = {"trunk": {"tokens": text_tokens}, "head": {}}
        if self.supply_seq_len_to_head:
            text_lengths = text.argmax(axis=-1)
            return_dict["head"] = {"seq_len": text_lengths}
        if self.causal_masking:
            return_dict["trunk"].update({"attn_mask": self.mask})
        return return_dict


class Im2Video(paddle.nn.Layer):
    """Convert an image into a trivial video."""

    def __init__(self, time_dim=2):
        super().__init__()
        self.time_dim = time_dim

    def forward(self, x):
        if x.ndim == 4:
            return x.unsqueeze(axis=self.time_dim)
        elif x.ndim == 5:
            return x
        else:
            raise ValueError(f"Dimension incorrect {x.shape}")


class PadIm2Video(Im2Video):
    def __init__(self, ntimes, pad_type, time_dim=2):
        super().__init__(time_dim=time_dim)
        assert ntimes > 0
        assert pad_type in ["zero", "repeat"]
        self.ntimes = ntimes
        self.pad_type = pad_type

    def forward(self, x):
        x = super().forward(x)
        if x.shape[self.time_dim] == 1:
            if self.pad_type == "repeat":
                new_shape = [1] * len(x.shape)
                new_shape[self.time_dim] = self.ntimes
                x = x.tile(repeat_times=new_shape)
            elif self.pad_type == "zero":
                padarg = [0, 0] * len(x.shape)
                padarg[2 * self.time_dim + 1] = self.ntimes - x.shape[self.time_dim]
                x = paddle.nn.functional.pad(x=x, pad=padarg)
        return x


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub("\\s+", " ", text)
    text = text.strip()
    return text


class IMUPreprocessor(VerboseNNModule):
    def __init__(
        self,
        kernel_size: int,
        imu_stem: PatchEmbedGeneric,
        embed_dim: int,
        img_size: Tuple = (6, 2000),
        num_cls_tokens: int = 1,
        pos_embed_fn: Optional[Callable] = None,
        init_param_style: str = "openclip",
    ) -> None:
        super().__init__()
        self.imu_stem = imu_stem
        self.embed_dim = embed_dim
        self.use_pos_embed = pos_embed_fn is not None
        self.num_cls_tokens = num_cls_tokens
        self.kernel_size = kernel_size

        self.pos_embed = paddle.create_parameter(
            shape=[1, img_size[1] // kernel_size + num_cls_tokens, embed_dim],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Assign(
                value=paddle.empty(shape=[1, img_size[1] // kernel_size + num_cls_tokens, embed_dim])
            ),
        )
        if self.num_cls_tokens > 0:

            self.cls_token = paddle.create_parameter(
                shape=[1, self.num_cls_tokens, self.embed_dim],
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(value=0.0),
            )
        self.init_parameters(init_param_style)

    @paddle.no_grad()
    def init_parameters(self, init_param_style):
        paddle.nn.initializer.TruncatedNormal(std=0.01)(self.pos_embed)

        if init_param_style == "openclip":
            scale = self.embed_dim**-0.5
            if self.num_cls_tokens > 0:
                paddle.nn.initializer.TruncatedNormal()(self.cls_token)

                self.cls_token.set_value(self.cls_token * scale)
        elif init_param_style == "vit":
            self.cls_token.data.fill_(value=0)
        else:
            raise ValueError(f"Unknown init {init_param_style}")

    def tokenize_input_and_cls_pos(self, input, stem):
        tokens = stem.norm_layer(stem.proj(input))
        assert tokens.ndim == 3
        assert tokens.shape[2] == self.embed_dim
        B = tokens.shape[0]
        if self.num_cls_tokens > 0:
            class_tokens = self.cls_token.expand(shape=[B, -1, -1])
            tokens = paddle.concat(x=(class_tokens, tokens), axis=1)
        if self.use_pos_embed:
            tokens = tokens + self.pos_embed
        return tokens

    def forward(self, imu):

        imu = imu.unfold(-1, self.kernel_size, self.kernel_size).transpose(perm=[0, 2, 1, 3])  # 需要对齐
        imu = imu.reshape((imu.shape[0], imu.shape[1], -1))
        imu_tokens = self.tokenize_input_and_cls_pos(imu, self.imu_stem)
        return_dict = {"trunk": {"tokens": imu_tokens}, "head": {}}
        return return_dict
