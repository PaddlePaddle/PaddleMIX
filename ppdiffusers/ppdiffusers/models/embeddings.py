# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import math
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import paddle
from paddle import nn

from ..utils import USE_PEFT_BACKEND
from .activations import FP32SiLU, get_activation
from .lora import LoRACompatibleLinear


def get_timestep_embedding(
    timesteps: paddle.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * paddle.arange(start=0, end=half_dim, dtype="float32")

    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = paddle.exp(exponent)
    emb = timesteps[:, None].cast("float32") * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = paddle.concat([paddle.cos(emb), paddle.sin(emb)], axis=-1)
    else:
        emb = paddle.concat([paddle.sin(emb), paddle.cos(emb)], axis=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = paddle.concat(emb, paddle.zeros([emb.shape[0], 1]), axis=-1)
    return emb


def get_2d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, interpolation_scale=1.0, base_size=16
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PatchEmbed(nn.Layer):
    """2D Image to Patch Embedding with support for SD3 cropping."""

    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
        add_pos_embed=True,
        data_format="NCHW",
        pos_embed_max_size=None,  # For SD3 cropping
    ):
        super().__init__()

        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size
        self.data_format = data_format

        self.proj = nn.Conv2D(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias_attr=bias,
            data_format=data_format,
        )
        if layer_norm:
            norm_elementwise_affine_kwargs = dict(weight_attr=False, bias_attr=False)
            self.norm = nn.LayerNorm(embed_dim, eps=1e-6, **norm_elementwise_affine_kwargs)
        else:
            self.norm = None

        self.patch_size = patch_size
        # See:
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L161
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale

        # Calculate positional embeddings based on max size or default
        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(num_patches**0.5)

        # NOTE, new add for unidiffusers!
        self.add_pos_embed = add_pos_embed
        if add_pos_embed:
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim, grid_size, base_size=self.base_size, interpolation_scale=self.interpolation_scale
            )
            persistent = True if pos_embed_max_size else False
            self.register_buffer(
                "pos_embed", paddle.to_tensor(pos_embed).cast("float32").unsqueeze(0), persistable=persistent
            )

    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed.reshape([1, self.pos_embed_max_size, self.pos_embed_max_size, -1])
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        spatial_pos_embed = spatial_pos_embed.reshape([1, -1, spatial_pos_embed.shape[-1]])
        return spatial_pos_embed

    def forward(self, latent):
        if self.data_format == "NCHW":
            # height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
            if self.pos_embed_max_size is not None:
                height, width = latent.shape[-2:]
            else:
                height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size
        else:
            height, width = latent.shape[-3] // self.patch_size, latent.shape[-2] // self.patch_size
        latent = self.proj(latent)
        if self.flatten:
            if self.data_format == "NCHW":
                latent = latent.flatten(2).transpose([0, 2, 1])  # BCHW -> BNC
            else:
                latent = latent.flatten(1, 2)  # BHWC -> BNC
        if self.layer_norm:
            latent = self.norm(latent)

        # Interpolate or crop positional embeddings as needed
        if self.add_pos_embed:
            if self.pos_embed_max_size:
                pos_embed = self.cropped_pos_embed(height, width)
            else:
                if self.height != height or self.width != width:
                    pos_embed = get_2d_sincos_pos_embed(
                        embed_dim=self.pos_embed.shape[-1],
                        grid_size=(height, width),
                        base_size=self.base_size,
                        interpolation_scale=self.interpolation_scale,
                    )
                    pos_embed = paddle.to_tensor(pos_embed).astype(paddle.float32).unsqueeze(0)
                else:
                    pos_embed = self.pos_embed

        # NOTE, new add for unidiffusers!
        if self.add_pos_embed:
            return (latent + pos_embed).cast(latent.dtype)
        else:
            return latent.cast(latent.dtype)


class TimestepEmbedding(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
    ):
        super().__init__()
        linear_cls = nn.Linear if USE_PEFT_BACKEND else LoRACompatibleLinear

        self.linear_1 = linear_cls(in_channels, time_embed_dim)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias_attr=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = linear_cls(time_embed_dim, time_embed_dim_out)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition.cast(sample.dtype))  # NEW ADD cast dtype

        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class Timesteps(nn.Layer):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


class GaussianFourierProjection(nn.Layer):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(
        self, embedding_size: int = 256, scale: float = 1.0, set_W_to_weight=True, log=True, flip_sin_to_cos=False
    ):
        super().__init__()
        self.register_buffer("weight", paddle.randn((embedding_size,)) * scale)
        self.log = log
        self.flip_sin_to_cos = flip_sin_to_cos

        if set_W_to_weight:
            # to delete later
            self.register_buffer("W", paddle.randn((embedding_size,)) * scale)

            self.weight = self.W

    def forward(self, x):
        # TODO must cast x to float32
        x = x.cast(self.weight.dtype)
        if self.log:
            x = paddle.log(x)

        x_proj = x[:, None] * self.weight[None, :] * 2 * np.pi

        if self.flip_sin_to_cos:
            out = paddle.concat([paddle.cos(x_proj), paddle.sin(x_proj)], axis=-1)
        else:
            out = paddle.concat([paddle.sin(x_proj), paddle.cos(x_proj)], axis=-1)
        return out


class SinusoidalPositionalEmbedding(nn.Layer):
    """Apply positional information to a sequence of embeddings.

    Takes in a sequence of embeddings with shape (batch_size, seq_length, embed_dim) and adds positional embeddings to
    them

    Args:
        embed_dim: (int): Dimension of the positional embedding.
        max_seq_length: Maximum sequence length to apply positional embeddings

    """

    def __init__(self, embed_dim: int, max_seq_length: int = 32):
        super().__init__()
        position = paddle.arange(max_seq_length, dtype="float32").unsqueeze(1)
        div_term = paddle.exp(paddle.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = paddle.zeros([1, max_seq_length, embed_dim])
        pe[0, :, 0::2] = paddle.sin(position * div_term)
        pe[0, :, 1::2] = paddle.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        _, seq_length, _ = x.shape
        x = x + self.pe[:, :seq_length]
        return x


class ImagePositionalEmbeddings(nn.Layer):
    """
    Converts latent image classes into vector embeddings. Sums the vector embeddings with positional embeddings for the
    height and width of the latent space.

    For more details, see figure 10 of the dall-e paper: https://arxiv.org/abs/2102.12092

    For VQ-diffusion:

    Output vector embeddings are used as input for the transformer.

    Note that the vector embeddings for the transformer are different than the vector embeddings from the VQVAE.

    Args:
        num_embed (`int`):
            Number of embeddings for the latent pixels embeddings.
        height (`int`):
            Height of the latent image i.e. the number of height embeddings.
        width (`int`):
            Width of the latent image i.e. the number of width embeddings.
        embed_dim (`int`):
            Dimension of the produced vector embeddings. Used for the latent pixel, height, and width embeddings.
    """

    def __init__(
        self,
        num_embed: int,
        height: int,
        width: int,
        embed_dim: int,
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.num_embed = num_embed
        self.embed_dim = embed_dim

        self.emb = nn.Embedding(self.num_embed, embed_dim)
        self.height_emb = nn.Embedding(self.height, embed_dim)
        self.width_emb = nn.Embedding(self.width, embed_dim)

    def forward(self, index):
        emb = self.emb(index)

        height_emb = self.height_emb(paddle.arange(self.height).reshape([1, self.height]))

        # 1 x H x D -> 1 x H x 1 x D
        height_emb = height_emb.unsqueeze(2)

        width_emb = self.width_emb(paddle.arange(self.width).reshape([1, self.width]))

        # 1 x W x D -> 1 x 1 x W x D
        width_emb = width_emb.unsqueeze(1)

        pos_emb = height_emb + width_emb

        # 1 x H x W x D -> 1 x L xD
        pos_emb = pos_emb.reshape([1, self.height * self.width, -1])

        emb = emb + pos_emb[:, : emb.shape[1], :]

        return emb


class LabelEmbedding(nn.Layer):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.

    Args:
        num_classes (`int`): The number of classes.
        hidden_size (`int`): The size of the vector embeddings.
        dropout_prob (`float`): The probability of dropping a label.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                paddle.rand(
                    (labels.shape[0],),
                )
                < self.dropout_prob
            )
        else:
            drop_ids = paddle.to_tensor(force_drop_ids == 1)
        labels = paddle.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels: paddle.Tensor, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (self.training and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class TextImageProjection(nn.Layer):
    def __init__(
        self,
        text_embed_dim: int = 1024,
        image_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_image_text_embeds: int = 10,
    ):
        super().__init__()

        self.num_image_text_embeds = num_image_text_embeds
        self.image_embeds = nn.Linear(image_embed_dim, self.num_image_text_embeds * cross_attention_dim)
        self.text_proj = nn.Linear(text_embed_dim, cross_attention_dim)

    def forward(self, text_embeds: paddle.Tensor, image_embeds: paddle.Tensor):
        batch_size = text_embeds.shape[0]

        # image
        image_text_embeds = self.image_embeds(image_embeds)
        image_text_embeds = image_text_embeds.reshape([batch_size, self.num_image_text_embeds, -1])

        # text
        text_embeds = self.text_proj(text_embeds)

        return paddle.concat([image_text_embeds, text_embeds], axis=1)


class ImageProjection(nn.Layer):
    def __init__(
        self,
        image_embed_dim: int = 768,
        cross_attention_dim: int = 768,
        num_image_text_embeds: int = 32,
    ):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.num_image_text_embeds = num_image_text_embeds
        self.image_embeds = nn.Linear(image_embed_dim, self.num_image_text_embeds * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: paddle.Tensor):
        batch_size = paddle.shape(image_embeds)[0]

        # image
        image_embeds = self.image_embeds(image_embeds)
        image_embeds = image_embeds.reshape([batch_size, self.num_image_text_embeds, self.cross_attention_dim])
        image_embeds = self.norm(image_embeds)
        return image_embeds


class CombinedTimestepTextProjEmbeddings(nn.Layer):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.cast(dtype=pooled_projection.dtype))  # (N, D)

        pooled_projections = self.text_embedder(pooled_projection)

        conditioning = timesteps_emb + pooled_projections

        return conditioning


class CombinedTimestepLabelEmbeddings(nn.Layer):
    def __init__(self, num_classes, embedding_dim, class_dropout_prob=0.1):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.class_embedder = LabelEmbedding(num_classes, embedding_dim, class_dropout_prob)

    def forward(self, timestep, class_labels, hidden_dtype=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.cast(hidden_dtype))  # (N, D)

        class_labels = self.class_embedder(class_labels)  # (N, D)

        conditioning = timesteps_emb + class_labels  # (N, D)

        return conditioning


class TextTimeEmbedding(nn.Layer):
    def __init__(self, encoder_dim: int, time_embed_dim: int, num_heads: int = 64):
        super().__init__()
        self.norm1 = nn.LayerNorm(encoder_dim)
        self.pool = AttentionPooling(num_heads, encoder_dim)
        self.proj = nn.Linear(encoder_dim, time_embed_dim)
        self.norm2 = nn.LayerNorm(time_embed_dim)

    def forward(self, hidden_states):
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.pool(hidden_states)
        hidden_states = self.proj(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class TextImageTimeEmbedding(nn.Layer):
    def __init__(self, text_embed_dim: int = 768, image_embed_dim: int = 768, time_embed_dim: int = 1536):
        super().__init__()
        self.text_proj = nn.Linear(text_embed_dim, time_embed_dim)
        self.text_norm = nn.LayerNorm(time_embed_dim)
        self.image_proj = nn.Linear(image_embed_dim, time_embed_dim)

    def forward(self, text_embeds: paddle.Tensor, image_embeds: paddle.Tensor):
        # text
        time_text_embeds = self.text_proj(text_embeds)
        time_text_embeds = self.text_norm(time_text_embeds)

        # image
        time_image_embeds = self.image_proj(image_embeds)

        return time_image_embeds + time_text_embeds


class ImageTimeEmbedding(nn.Layer):
    def __init__(self, image_embed_dim: int = 768, time_embed_dim: int = 1536):
        super().__init__()
        self.image_proj = nn.Linear(image_embed_dim, time_embed_dim)
        self.image_norm = nn.LayerNorm(time_embed_dim)

    def forward(self, image_embeds: paddle.Tensor):
        # image
        time_image_embeds = self.image_proj(image_embeds)
        time_image_embeds = self.image_norm(time_image_embeds)
        return time_image_embeds


class ImageHintTimeEmbedding(nn.Layer):
    def __init__(self, image_embed_dim: int = 768, time_embed_dim: int = 1536):
        super().__init__()
        self.image_proj = nn.Linear(image_embed_dim, time_embed_dim)
        self.image_norm = nn.LayerNorm(time_embed_dim)
        self.input_hint_block = nn.Sequential(
            nn.Conv2D(3, 16, 3, padding=1),
            nn.Silu(),
            nn.Conv2D(16, 16, 3, padding=1),
            nn.Silu(),
            nn.Conv2D(16, 32, 3, padding=1, stride=2),
            nn.Silu(),
            nn.Conv2D(32, 32, 3, padding=1),
            nn.Silu(),
            nn.Conv2D(32, 96, 3, padding=1, stride=2),
            nn.Silu(),
            nn.Conv2D(96, 96, 3, padding=1),
            nn.Silu(),
            nn.Conv2D(96, 256, 3, padding=1, stride=2),
            nn.Silu(),
            nn.Conv2D(256, 4, 3, padding=1),
        )

    def forward(self, image_embeds: paddle.Tensor, hint: paddle.Tensor):
        # image
        time_image_embeds = self.image_proj(image_embeds)
        time_image_embeds = self.image_norm(time_image_embeds)
        hint = self.input_hint_block(hint)
        return time_image_embeds, hint


class AttentionPooling(nn.Layer):
    # Copied from https://github.com/deep-floyd/IF/blob/2f91391f27dd3c468bf174be5805b4cc92980c0b/deepfloyd_if/model/nn.py#L54

    def __init__(self, num_heads, embed_dim, dtype=None):
        super().__init__()
        self.dtype = dtype
        self.positional_embedding = nn.Parameter(paddle.randn([1, embed_dim]) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // self.num_heads
        self.scale = 1 / math.sqrt(math.sqrt(self.dim_per_head))

    def forward(self, x):
        bs, length, width = x.shape

        def shape(x):
            # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
            x = x.reshape([bs, -1, self.num_heads, self.dim_per_head])
            # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
            x = x.transpose([0, 2, 1, 3])
            # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
            x = x.reshape([bs * self.num_heads, -1, self.dim_per_head])
            # (bs*n_heads, length, dim_per_head) --> (bs*n_heads, dim_per_head, length)
            x = x.transpose([0, 2, 1])
            return x

        class_token = x.mean(axis=1, keepdim=True) + self.positional_embedding.cast(x.dtype)
        x = paddle.concat([class_token, x], axis=1)  # (bs, length+1, width)

        # (bs*n_heads, class_token_length, dim_per_head)
        q = shape(self.q_proj(class_token))
        # (bs*n_heads, length+class_token_length, dim_per_head)
        k = shape(self.k_proj(x))
        v = shape(self.v_proj(x))

        # (bs*n_heads, class_token_length, length+class_token_length):
        weight = paddle.einsum(
            "bct,bcs->bts", q * self.scale, k * self.scale
        )  # More stable with f16 than dividing afterwards
        weight = nn.functional.softmax(weight.cast("float32"), axis=-1).cast(weight.dtype)

        # (bs*n_heads, dim_per_head, class_token_length)
        a = paddle.einsum("bts,bcs->bct", weight, v)

        # (bs, length+1, width)
        a = a.reshape([bs, -1, 1]).transpose([0, 2, 1])

        return a[:, 0, :]  # cls_token


class FourierEmbedder(nn.Layer):
    def __init__(self, num_freqs=64, temperature=100):
        super().__init__()

        self.num_freqs = num_freqs
        self.temperature = temperature

        freq_bands = temperature ** (paddle.arange(num_freqs) / num_freqs)
        freq_bands = freq_bands[None, None, None]
        self.register_buffer("freq_bands", freq_bands, persistable=False)

    def __call__(self, x):
        x = self.freq_bands * x.unsqueeze(-1)
        return paddle.stack((x.sin(), x.cos()), axis=-1).transpose([0, 1, 3, 4, 2]).reshape([*x.shape[:2], -1])


class PositionNet(nn.Layer):
    def __init__(self, positive_len, out_dim, feature_type="text-only", fourier_freqs=8):
        super().__init__()
        self.positive_len = positive_len
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4  # 2: sin/cos, 4: xyxy

        if isinstance(out_dim, tuple):
            out_dim = out_dim[0]

        if feature_type == "text-only":
            self.linears = nn.Sequential(
                nn.Linear(self.positive_len + self.position_dim, 512),
                nn.Silu(),
                nn.Linear(512, 512),
                nn.Silu(),
                nn.Linear(512, out_dim),
            )
            self.null_positive_feature = nn.Parameter(
                paddle.zeros(
                    [
                        self.positive_len,
                    ]
                )
            )

        elif feature_type == "text-image":
            self.linears_text = nn.Sequential(
                nn.Linear(self.positive_len + self.position_dim, 512),
                nn.Silu(),
                nn.Linear(512, 512),
                nn.Silu(),
                nn.Linear(512, out_dim),
            )
            self.linears_image = nn.Sequential(
                nn.Linear(self.positive_len + self.position_dim, 512),
                nn.Silu(),
                nn.Linear(512, 512),
                nn.Silu(),
                nn.Linear(512, out_dim),
            )
            self.null_text_feature = nn.Parameter(paddle.zeros([self.positive_len]))
            self.null_image_feature = nn.Parameter(paddle.zeros([self.positive_len]))

        self.null_position_feature = nn.Parameter(paddle.zeros([self.position_dim]))

    def forward(
        self,
        boxes,
        masks,
        positive_embeddings=None,
        phrases_masks=None,
        image_masks=None,
        phrases_embeddings=None,
        image_embeddings=None,
    ):
        masks = masks.unsqueeze(-1)

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes)  # B*N*4 -> B*N*C

        # learnable null embedding
        xyxy_null = self.null_position_feature.reshape([1, 1, -1])

        # replace padding with learnable null embedding
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null

        # positionet with text only information
        if positive_embeddings is not None:
            # learnable null embedding
            positive_null = self.null_positive_feature.reshape([1, 1, -1])

            # replace padding with learnable null embedding
            positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null

            objs = self.linears(paddle.concat([positive_embeddings, xyxy_embedding], axis=-1))

        # positionet with text and image information
        else:
            phrases_masks = phrases_masks.unsqueeze(-1)
            image_masks = image_masks.unsqueeze(-1)

            # learnable null embedding
            text_null = self.null_text_feature.reshape([1, 1, -1])
            image_null = self.null_image_feature.reshape([1, 1, -1])

            # replace padding with learnable null embedding
            phrases_embeddings = phrases_embeddings * phrases_masks + (1 - phrases_masks) * text_null
            image_embeddings = image_embeddings * image_masks + (1 - image_masks) * image_null

            objs_text = self.linears_text(paddle.concat([phrases_embeddings, xyxy_embedding], axis=-1))
            objs_image = self.linears_image(paddle.concat([image_embeddings, xyxy_embedding], axis=-1))
            objs = paddle.concat([objs_text, objs_image], axis=1)

        return objs


class CombinedTimestepSizeEmbeddings(nn.Layer):
    """
    For PixArt-Alpha.

    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(self, embedding_dim, size_emb_dim, use_additional_conditions: bool = False):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions:
            self.use_additional_conditions = True
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
            self.aspect_ratio_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)

    def apply_condition(self, size: paddle.Tensor, batch_size: int, embedder: nn.Layer):
        if size.ndim == 1:
            size = size[:, None]

        if size.shape[0] != batch_size:
            size = size.tile([batch_size // size.shape[0], 1])
            if size.shape[0] != batch_size:
                raise ValueError(f"`batch_size` should be {size.shape[0]} but found {batch_size}.")

        current_batch_size, dims = size.shape[0], size.shape[1]
        size = size.reshape((-1,))
        size_freq = self.additional_condition_proj(size).cast(size.dtype)

        size_emb = embedder(size_freq)
        size_emb = size_emb.reshape([current_batch_size, dims * self.outdim])
        return size_emb

    def forward(self, timestep, resolution, aspect_ratio, batch_size, hidden_dtype):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.cast(dtype=hidden_dtype))  # (N, D)

        if self.use_additional_conditions:
            resolution = self.apply_condition(resolution, batch_size=batch_size, embedder=self.resolution_embedder)
            aspect_ratio = self.apply_condition(
                aspect_ratio, batch_size=batch_size, embedder=self.aspect_ratio_embedder
            )
            conditioning = timesteps_emb + paddle.concat([resolution, aspect_ratio], axis=1)
        else:
            conditioning = timesteps_emb

        return conditioning


class CaptionProjection(nn.Layer):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, num_tokens=120):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size)
        self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.register_buffer("y_embedding", paddle.randn(num_tokens, in_features) / in_features**0.5)

    def forward(self, caption, force_drop_ids=None):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class PixArtAlphaTextProjection(nn.Layer):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias_attr=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.Silu()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias_attr=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


def get_3d_sincos_pos_embed(
    embed_dim: int,
    spatial_size: Union[int, Tuple[int, int]],
    temporal_size: int,
    spatial_interpolation_scale: float = 1.0,
    temporal_interpolation_scale: float = 1.0,
) -> np.ndarray:
    """
    Args:
        embed_dim (`int`):
        spatial_size (`int` or `Tuple[int, int]`):
        temporal_size (`int`):
        spatial_interpolation_scale (`float`, defaults to 1.0):
        temporal_interpolation_scale (`float`, defaults to 1.0):
    """
    if embed_dim % 4 != 0:
        raise ValueError("`embed_dim` must be divisible by 4")
    if isinstance(spatial_size, int):
        spatial_size = spatial_size, spatial_size
    embed_dim_spatial = 3 * embed_dim // 4
    embed_dim_temporal = embed_dim // 4
    grid_h = np.arange(spatial_size[1], dtype=np.float32) / spatial_interpolation_scale
    grid_w = np.arange(spatial_size[0], dtype=np.float32) / spatial_interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, spatial_size[1], spatial_size[0]])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)
    grid_t = np.arange(temporal_size, dtype=np.float32) / temporal_interpolation_scale
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(pos_embed_spatial, temporal_size, axis=0)
    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(pos_embed_temporal, spatial_size[0] * spatial_size[1], axis=1)
    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    return pos_embed


class CogVideoXPatchEmbed(paddle.nn.Layer):
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        embed_dim: int = 1920,
        text_embed_dim: int = 4096,
        bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_positional_embeddings: bool = True,
        use_learned_positional_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_frames = sample_frames
        self.temporal_compression_ratio = temporal_compression_ratio
        self.max_text_seq_length = max_text_seq_length
        self.spatial_interpolation_scale = spatial_interpolation_scale
        self.temporal_interpolation_scale = temporal_interpolation_scale
        self.use_positional_embeddings = use_positional_embeddings
        self.use_learned_positional_embeddings = use_learned_positional_embeddings

        self.proj = paddle.nn.Conv2D(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=patch_size,
            bias_attr=bias,
        )

        self.text_proj = paddle.nn.Linear(in_features=text_embed_dim, out_features=embed_dim)

        if use_positional_embeddings or use_learned_positional_embeddings:
            persistent = use_learned_positional_embeddings
            pos_embedding = self._get_positional_embeddings(sample_height, sample_width, sample_frames)
            self.register_buffer(name="pos_embedding", tensor=pos_embedding, persistable=persistent)

    def _get_positional_embeddings(self, sample_height: int, sample_width: int, sample_frames: int) -> paddle.Tensor:
        post_patch_height = sample_height // self.patch_size
        post_patch_width = sample_width // self.patch_size
        post_time_compression_frames = (sample_frames - 1) // self.temporal_compression_ratio + 1
        num_patches = post_patch_height * post_patch_width * post_time_compression_frames

        pos_embedding = get_3d_sincos_pos_embed(
            self.embed_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            self.spatial_interpolation_scale,
            self.temporal_interpolation_scale,
        )
        pos_embedding = paddle.to_tensor(data=pos_embedding).flatten(start_axis=0, stop_axis=1)
        joint_pos_embedding = paddle.zeros([1, self.max_text_seq_length + num_patches, self.embed_dim])
        joint_pos_embedding[0, self.max_text_seq_length :] = pos_embedding
        return joint_pos_embedding

    def forward(self, text_embeds: paddle.Tensor, image_embeds: paddle.Tensor):
        """
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        text_embeds = self.text_proj(text_embeds)

        batch, num_frames, channels, height, width = image_embeds.shape
        image_embeds = image_embeds.reshape([-1, channels, height, width])
        image_embeds = self.proj(image_embeds)
        image_embeds = image_embeds.reshape([batch, num_frames] + image_embeds.shape[1:])
        image_embeds = image_embeds.flatten(3).transpose([0, 1, 3, 2])  # [batch, num_frames, height x width, channels]
        image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]

        embeds = paddle.concat(x=[text_embeds, image_embeds], axis=1).contiguous()
        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            if self.use_learned_positional_embeddings and (self.sample_width != width or self.sample_height != height):
                warnings.warn(
                    "It is currently not possible to generate videos at a different resolution that the defaults. This should only be the case with 'THUDM/CogVideoX-5b-I2V'.If you think this is incorrect, please open an issue at https://github.com/huggingface/diffusers/issues."
                )
            pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1
            if (
                self.sample_height != height
                or self.sample_width != width
                or self.sample_frames != pre_time_compression_frames
            ):
                pos_embedding = self._get_positional_embeddings(height, width, pre_time_compression_frames)
                pos_embedding = pos_embedding.cast(embeds.dtype)
            else:
                pos_embedding = self.pos_embedding
            embeds = embeds + pos_embedding
        return embeds


def get_3d_rotary_pos_embed(
    embed_dim, crops_coords, grid_size, temporal_size, theta: int = 10000, use_real: bool = True
) -> Union[paddle.Tensor, Tuple[paddle.Tensor, paddle.Tensor]]:
    """
    RoPE for video tokens with 3D structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    crops_coords (`Tuple[int]`):
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the spatial positional embedding (height, width).
    temporal_size (`int`):
        The size of the temporal dimension.
    theta (`float`):
        Scaling factor for frequency computation.
    use_real (`bool`):
        If True, return real part and imaginary part separately. Otherwise, return complex numbers.

    Returns:
        `torch.Tensor`: positional embedding with shape `(temporal_size * grid_size[0] * grid_size[1], embed_dim/2)`.
    """
    start, stop = crops_coords
    grid_h = np.linspace(start[0], stop[0], grid_size[0], endpoint=False, dtype=np.float32)
    grid_w = np.linspace(start[1], stop[1], grid_size[1], endpoint=False, dtype=np.float32)
    grid_t = np.linspace(0, temporal_size, temporal_size, endpoint=False, dtype=np.float32)
    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3
    freqs_t = 1.0 / theta ** (paddle.arange(start=0, end=dim_t, step=2).astype(dtype="float32") / dim_t)
    grid_t = paddle.to_tensor(data=grid_t).astype(dtype="float32")
    freqs_t = paddle.einsum("n , f -> n f", grid_t, freqs_t)
    freqs_t = freqs_t.repeat_interleave(repeats=2, axis=-1)
    freqs_h = 1.0 / theta ** (paddle.arange(start=0, end=dim_h, step=2).astype(dtype="float32") / dim_h)
    freqs_w = 1.0 / theta ** (paddle.arange(start=0, end=dim_w, step=2).astype(dtype="float32") / dim_w)
    grid_h = paddle.to_tensor(data=grid_h).astype(dtype="float32")
    grid_w = paddle.to_tensor(data=grid_w).astype(dtype="float32")
    freqs_h = paddle.einsum("n , f -> n f", grid_h, freqs_h)
    freqs_w = paddle.einsum("n , f -> n f", grid_w, freqs_w)
    freqs_h = freqs_h.repeat_interleave(repeats=2, axis=-1)
    freqs_w = freqs_w.repeat_interleave(repeats=2, axis=-1)

    def broadcast(tensors, dim=-1):
        num_tensors = len(tensors)
        shape_lens = {len(tuple(t.shape)) for t in tensors}
        assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
        shape_len = list(shape_lens)[0]
        dim = dim + shape_len if dim < 0 else dim
        dims = list(zip(*(list(tuple(t.shape)) for t in tensors)))
        expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
        assert all(
            [*(len(set(t[1])) <= 2 for t in expandable_dims)]
        ), "invalid dimensions for broadcastable concatenation"
        max_dims = [(t[0], max(t[1])) for t in expandable_dims]
        expanded_dims = [(t[0], (t[1],) * num_tensors) for t in max_dims]
        expanded_dims.insert(dim, (dim, dims[dim]))
        expandable_shapes = list(zip(*(t[1] for t in expanded_dims)))
        tensors = [t[0].expand(shape=t[1]) for t in zip(tensors, expandable_shapes)]
        return paddle.concat(x=tensors, axis=dim)

    freqs = broadcast((freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)
    t, h, w, d = tuple(freqs.shape)
    freqs = freqs.reshape([t * h * w, d])
    sin = freqs.sin()
    cos = freqs.cos()
    if use_real:
        return cos, sin
    else:
        freqs_cis = paddle.complex(
            paddle.ones_like(x=freqs) * paddle.cos(freqs), paddle.ones_like(x=freqs) * paddle.sin(freqs)
        )
        return freqs_cis


def apply_rotary_emb(
    x: paddle.Tensor,
    freqs_cis: Union[paddle.Tensor, Tuple[paddle.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis
        cos = cos[None, None]
        sin = sin[None, None]
        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape([*tuple(x.shape)[:-1], -1, 2]).unbind(axis=-1)
            x_rotated = paddle.stack(x=[-x_imag, x_real], axis=-1).flatten(start_axis=3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape([*tuple(x.shape)[:-1], 2, -1]).unbind(axis=-2)
            x_rotated = paddle.concat(x=[-x_imag, x_real], axis=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")
        out = (x.astype(dtype="float32") * cos + x_rotated.astype(dtype="float32") * sin).to(x.dtype)
        return out
    else:
        x_rotated = paddle.as_complex(x=x.astype(dtype="float32").reshape(*tuple(x.shape)[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(axis=2)
        x_out = paddle.as_real(x=x_rotated * freqs_cis).flatten(start_axis=3)
        return x_out.astype(dtype=x.dtype)

class CombinedTimestepGuidanceTextProjEmbeddings(paddle.nn.Layer):
    def __init__(self, embedding_dim, pooled_projection_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

    def forward(self, timestep, guidance, pooled_projection):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))

        guidance_proj = self.time_proj(guidance)
        guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))

        time_guidance_emb = timesteps_emb + guidance_emb

        pooled_projections = self.text_embedder(pooled_projection)
        conditioning = time_guidance_emb + pooled_projections

        return conditioning

def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype="float32",
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0
    if isinstance(pos, int):
        pos = paddle.arange(end=pos)
    if isinstance(pos, np.ndarray):
        pos = paddle.to_tensor(data=pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0
        / theta ** (paddle.arange(start=0, end=dim, step=2, dtype=freqs_dtype)[: dim // 2] / dim
        )
        / linear_factor
    )  # [D/2]
    freqs = paddle.outer(x=pos, y=freqs)  # type: ignore   # [S, D/2]
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = (freqs.cos().repeat_interleave(repeats=2, axis=1).astype(dtype="float32"))  # [S, D]
        freqs_sin = (freqs.sin().repeat_interleave(repeats=2, axis=1).astype(dtype="float32"))  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = paddle.concat(x=[freqs.cos(), freqs.cos()], axis=-1).astype(dtype="float32")  # [S, D]
        freqs_sin = paddle.concat(x=[freqs.sin(), freqs.sin()], axis=-1).astype(dtype="float32")  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = paddle.complex(
            paddle.ones_like(x=freqs) * paddle.cos(freqs),
            paddle.ones_like(x=freqs) * paddle.sin(freqs),
        )  # complex64     # [S, D/2]
        return freqs_cis