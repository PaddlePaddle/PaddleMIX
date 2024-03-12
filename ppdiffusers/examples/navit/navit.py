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

from functools import partial
from typing import List, Union

import numpy as np
import paddle
import paddle.nn.functional as F
from einops import rearrange, repeat
from paddle import Tensor, nn


# helpers
def pad_sequence_paddle(sequences, padding_value=0):
    """
    Implement a function similar to PyTorch's pad_sequence in PaddlePaddle.

    Args:
    - sequences (list of Tensor): The list of sequences to be padded.
    - padding_value (float, optional): The value used for padding, default is 0.

    Returns:
    - Tensor: The result of padding all sequences to the same length.
    """
    # Calculate the maximum length
    max_len = max([seq.shape[0] for seq in sequences])

    # Pad sequences
    padded_sequences = []
    for seq in sequences:
        # Calculate the length to pad
        padding_len = max_len - seq.shape[0]

        # Create a padding tensor
        if padding_len > 0:
            padding_tensor = paddle.full([padding_len] + list(seq.shape[1:]), padding_value, dtype=seq.dtype)
            # Concatenate the original sequence and the padding tensor
            padded_seq = paddle.concat([seq, padding_tensor], axis=0)
        else:
            padded_seq = seq

        padded_sequences.append(padded_seq)

    # Stack the padded sequences to form a batch
    padded_batch = paddle.stack(padded_sequences, axis=0)
    return padded_batch


def orig_pad_sequence(
    sequences: Union[Tensor, List[Tensor]],
    batch_first: bool = False,
    padding_value: float = 0.0,
) -> Tensor:
    if batch_first:
        return pad_sequence_paddle(sequences, padding_value)
    else:
        assert False, "Not implemented"


def finfo(dtype):
    if dtype == paddle.float32:
        return np.finfo(np.float32)
    if dtype == paddle.float16:
        return np.finfo(np.float16)
    if dtype == paddle.float64:
        return np.finfo(np.float64)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def always(val):
    return lambda *args: val


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def divisible_by(numer, denom):
    return (numer % denom) == 0


def group_images_by_max_seq_len(
    images: List[Tensor], patch_size: int, calc_token_dropout=None, max_seq_len=2048
) -> List[List[Tensor]]:

    calc_token_dropout = default(calc_token_dropout, always(0.0))

    groups = []
    group = []
    seq_len = 0

    if isinstance(calc_token_dropout, (float, int)):
        calc_token_dropout = always(calc_token_dropout)

    for image in images:
        assert isinstance(image, Tensor)

        image_dims = image.shape[-2:]
        ph, pw = map(lambda t: t // patch_size, image_dims)

        image_seq_len = ph * pw
        image_seq_len = int(image_seq_len * (1 - calc_token_dropout(*image_dims)))

        assert image_seq_len <= max_seq_len, f"image with dimensions {image_dims} exceeds maximum sequence length"

        if (seq_len + image_seq_len) > max_seq_len:
            groups.append(group)
            group = []
            seq_len = 0

        group.append(image)
        seq_len += image_seq_len

    if len(group) > 0:
        groups.append(group)

    return groups


# normalization
# they use layernorm without bias, something that pytorch does not offer
class LayerNorm(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        x = paddle.ones(dim)
        self.gamma = paddle.create_parameter(
            shape=x.shape, dtype=x.dtype, default_initializer=paddle.nn.initializer.Assign(x)
        )
        self.gamma.stop_gradient = False

        self.register_buffer("beta", paddle.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# they use a query-key normalization that is equivalent to rms norm (no mean-centering, learned gamma), from vit 22B paper
class RMSNorm(nn.Layer):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim**0.5
        x = paddle.ones([heads, 1, dim])
        self.gamma = paddle.create_parameter(
            shape=x.shape, dtype=x.dtype, default_initializer=paddle.nn.initializer.Assign(x)
        )
        self.gamma.stop_gradient = False

    def forward(self, x):
        normed = F.normalize(x, axis=-1)
        return normed * self.scale * self.gamma


# feedforward
def FeedForward(dim, hidden_dim, dropout=0.0):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )


# attention
class Attention(nn.Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.norm = LayerNorm(dim)

        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)

        self.attend = nn.Softmax(axis=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias_attr=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias_attr=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias_attr=False), nn.Dropout(dropout))

    def forward(self, x, context=None, mask=None, attn_mask=None):
        x = self.norm(x)
        kv_input = default(context, x)

        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, axis=-1))

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # dots = torch.matmul(q, k.transpose(-1, -2))
        x = k
        perm_0 = list(range(x.ndim))
        perm_0[-1] = -2
        perm_0[-2] = -1
        dots = paddle.matmul(x=q, y=x.transpose(perm=perm_0))

        if exists(mask):
            mask = rearrange(mask, "b j -> b 1 1 j")
            dots = dots.masked_fill(~mask, -finfo(dots.dtype).max)

        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -finfo(dots.dtype).max)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = paddle.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# transformer block
class Transformer(nn.Layer):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.LayerList([])
        for _ in range(depth):
            self.layers.append(
                nn.LayerList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

        self.norm = LayerNorm(dim)

    def forward(self, x, mask=None, attn_mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask, attn_mask=attn_mask) + x
            x = ff(x) + x
        return self.norm(x)


class NaViT(nn.Layer):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        token_dropout_prob=None
    ):
        super().__init__()
        image_height, image_width = pair(image_size)

        # what percent of tokens to dropout
        # if int or float given, then assume constant dropout prob
        # otherwise accept a callback that in turn calculates dropout prob from height and width

        self.calc_token_dropout = None

        if callable(token_dropout_prob):
            self.calc_token_dropout = token_dropout_prob

        elif isinstance(token_dropout_prob, (float, int)):
            assert 0.0 < token_dropout_prob < 1.0
            token_dropout_prob = float(token_dropout_prob)
            self.calc_token_dropout = lambda height, width: token_dropout_prob

        # calculate patching related stuff

        assert divisible_by(image_height, patch_size) and divisible_by(
            image_width, patch_size
        ), "Image dimensions must be divisible by the patch size."

        patch_height_dim, patch_width_dim = (image_height // patch_size), (image_width // patch_size)
        patch_dim = channels * (patch_size**2)

        self.channels = channels
        self.patch_size = patch_size

        self.to_patch_embedding = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim),
        )

        x = paddle.randn([patch_height_dim, dim])
        self.pos_embed_height = paddle.create_parameter(
            shape=x.shape, dtype=x.dtype, default_initializer=paddle.nn.initializer.Assign(x)
        )
        x = paddle.randn([patch_width_dim, dim])
        self.pos_embed_width = paddle.create_parameter(
            shape=x.shape, dtype=x.dtype, default_initializer=paddle.nn.initializer.Assign(x)
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # final attention pooling queries

        x = paddle.randn([dim])
        self.attn_pool_queries = paddle.create_parameter(
            shape=x.shape, dtype=x.dtype, default_initializer=paddle.nn.initializer.Assign(x)
        )
        self.attn_pool = Attention(dim=dim, dim_head=dim_head, heads=heads)

        # output to logits

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(LayerNorm(dim), nn.Linear(dim, num_classes, bias_attr=False))

    @property
    def device(self):
        return next(self.parameters()).place

    def forward(
        self,
        batched_images: Union[
            List[Tensor], List[List[Tensor]]
        ],  # assume different resolution images already grouped correctly
        group_images=False,
        group_max_seq_len=2048,
    ):
        p, c, has_token_dropout = self.patch_size, self.channels, exists(self.calc_token_dropout)

        arange = paddle.arange
        pad_sequence = partial(orig_pad_sequence, batch_first=True)

        # auto pack if specified

        if group_images:
            batched_images = group_images_by_max_seq_len(
                batched_images,
                patch_size=self.patch_size,
                calc_token_dropout=self.calc_token_dropout,
                max_seq_len=group_max_seq_len,
            )

        # process images into variable lengthed sequences with attention mask

        num_images = []
        batched_sequences = []
        batched_positions = []
        batched_image_ids = []

        for images in batched_images:
            num_images.append(len(images))

            sequences = []
            positions = []
            image_ids = paddle.empty((0,), dtype="int64")

            for image_id, image in enumerate(images):
                assert image.ndim == 3 and image.shape[0] == c
                image_dims = image.shape[-2:]
                assert all(
                    [divisible_by(dim, p) for dim in image_dims]
                ), f"height and width {image_dims} of images must be divisible by patch size {p}"

                ph, pw = map(lambda dim: dim // p, image_dims)

                pos = paddle.stack(paddle.meshgrid((arange(ph), arange(pw))), axis=-1)

                pos = rearrange(pos, "h w c -> (h w) c")
                seq = rearrange(image, "c (h p1) (w p2) -> (h w) (c p1 p2)", p1=p, p2=p)

                seq_len = seq.shape[-2]

                if has_token_dropout:
                    token_dropout = self.calc_token_dropout(*image_dims)
                    num_keep = max(1, int(seq_len * (1 - token_dropout)))
                    topk_values, keep_indices = paddle.randn((seq_len,)).topk(num_keep, axis=-1)

                    seq = seq[keep_indices]
                    pos = pos[keep_indices]

                image_ids = F.pad(image_ids, (0, seq.shape[-2]), value=image_id)
                sequences.append(seq)
                positions.append(pos)

            batched_image_ids.append(image_ids)
            batched_sequences.append(paddle.concat(sequences, axis=0))
            batched_positions.append(paddle.concat(positions, axis=0))

        # derive key padding mask

        lengths = paddle.to_tensor([seq.shape[-2] for seq in batched_sequences], dtype="int64")
        max_length = arange(lengths.amax().item())
        key_pad_mask = rearrange(lengths, "b -> b 1") <= rearrange(max_length, "n -> 1 n")

        # derive attention mask, and combine with key padding mask from above

        batched_image_ids = pad_sequence(batched_image_ids)
        attn_mask = rearrange(batched_image_ids, "b i -> b 1 i 1") == rearrange(batched_image_ids, "b j -> b 1 1 j")
        attn_mask = attn_mask & rearrange(key_pad_mask, "b j -> b 1 1 j")

        # combine patched images as well as the patched width / height positions for 2d positional embedding

        patches = pad_sequence(batched_sequences)
        patch_positions = pad_sequence(batched_positions)

        # need to know how many images for final attention pooling

        num_images = paddle.to_tensor(data=num_images, dtype="int64")
        # to patches

        x = self.to_patch_embedding(patches)

        # factorized 2d absolute positional embedding

        h_indices, w_indices = patch_positions.unbind(axis=-1)

        h_pos = self.pos_embed_height[h_indices]
        w_pos = self.pos_embed_width[w_indices]

        x = x + h_pos + w_pos

        # embed dropout
        x = self.dropout(x)

        # attention

        x = self.transformer(x, attn_mask=attn_mask)

        # do attention pooling at the end

        max_queries = num_images.amax().item()

        queries = repeat(self.attn_pool_queries, "d -> b n d", n=max_queries, b=x.shape[0])

        # attention pool mask

        image_id_arange = arange(max_queries)

        attn_pool_mask = rearrange(image_id_arange, "i -> i 1") == rearrange(batched_image_ids, "b j -> b 1 j")

        attn_pool_mask = attn_pool_mask & rearrange(key_pad_mask, "b j -> b 1 j")

        attn_pool_mask = rearrange(attn_pool_mask, "b i j -> b 1 i j")

        # attention pool

        x = self.attn_pool(queries, context=x, attn_mask=attn_pool_mask) + queries

        x = rearrange(x, "b n d -> (b n) d")

        # each batch element may not have same amount of images

        is_images = image_id_arange < rearrange(num_images, "b -> b 1")
        is_images = rearrange(is_images, "b n -> (b n)")

        x = x[is_images]

        # project out to logits

        x = self.to_latent(x)

        return self.mlp_head(x)
