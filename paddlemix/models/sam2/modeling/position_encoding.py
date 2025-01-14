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

import math
from typing import Optional, Tuple

import numpy as np
import paddle


class PositionEmbeddingSine(paddle.nn.Layer):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention Is All You Need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats, temperature: int = 10000, normalize: bool = True, scale: Optional[float] = None):
        super().__init__()
        assert num_pos_feats % 2 == 0, "Expecting even model width"
        self.num_pos_feats = num_pos_feats // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.cache = {}

    def _encode_xy(self, x, y):
        assert len(x) == len(y) and x.ndim == y.ndim == 1
        x_embed = x * self.scale
        y_embed = y * self.scale
        dim_t = paddle.arange(dtype="float32", end=self.num_pos_feats)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, None] / dim_t
        pos_y = y_embed[:, None] / dim_t
        pos_x = paddle.stack(x=(pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), axis=2).flatten(start_axis=1)
        pos_y = paddle.stack(x=(pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), axis=2).flatten(start_axis=1)
        return pos_x, pos_y

    @paddle.no_grad()
    def encode_boxes(self, x, y, w, h):
        pos_x, pos_y = self._encode_xy(x, y)
        pos = paddle.concat(x=(pos_y, pos_x, h[:, None], w[:, None]), axis=1)
        return pos

    encode = encode_boxes

    @paddle.no_grad()
    def encode_points(self, x, y, labels):
        (bx, nx), (by, ny), (bl, nl) = tuple(x.shape), tuple(y.shape), tuple(labels.shape)
        assert bx == by and nx == ny and bx == bl and nx == nl
        pos_x, pos_y = self._encode_xy(x.flatten(), y.flatten())
        pos_x, pos_y = pos_x.reshape([bx, nx, -1]), pos_y.reshape([by, ny, -1])
        pos = paddle.concat(x=(pos_y, pos_x, labels[:, :, None]), axis=2)
        return pos

    @paddle.no_grad()
    def forward(self, x: paddle.Tensor):
        cache_key = tuple(x.shape)[-2], tuple(x.shape)[-1]
        if cache_key in self.cache:
            return self.cache[cache_key][None].tile([tuple(x.shape)[0], 1, 1, 1])
        y_embed = (
            paddle.arange(start=1, end=tuple(x.shape)[-2] + 1, dtype="float32")
            .reshape([1, -1, 1])
            .tile((tuple(x.shape)[0], 1, tuple(x.shape)[-1]))
        )
        x_embed = (
            paddle.arange(start=1, end=tuple(x.shape)[-1] + 1, dtype="float32")
            .reshape([1, 1, -1])
            .tile((tuple(x.shape)[0], tuple(x.shape)[-2], 1))
        )
        if self.normalize:
            eps = 1e-06
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = paddle.arange(dtype="float32", end=self.num_pos_feats)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = paddle.stack(x=(pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), axis=4).flatten(start_axis=3)
        pos_y = paddle.stack(x=(pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), axis=4).flatten(start_axis=3)
        pos = paddle.concat(x=(pos_y, pos_x), axis=3).transpose(perm=[0, 3, 1, 2])
        self.cache[cache_key] = pos[0]
        return pos


class PositionEmbeddingRandom(paddle.nn.Layer):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            name="positional_encoding_gaussian_matrix", tensor=scale * paddle.randn(shape=(2, num_pos_feats))
        )

    def _pe_encoding(self, coords: paddle.Tensor) -> paddle.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return paddle.concat(x=[paddle.sin(x=coords), paddle.cos(x=coords)], axis=-1)

    def forward(self, size: Tuple[int, int]) -> paddle.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        grid = paddle.ones(shape=(h, w), dtype="float32")
        y_embed = grid.cumsum(axis=0) - 0.5
        x_embed = grid.cumsum(axis=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(paddle.stack(x=[x_embed, y_embed], axis=-1))
        return pe.transpose(perm=[2, 0, 1])

    def forward_with_coords(self, coords_input: paddle.Tensor, image_size: Tuple[int, int]) -> paddle.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to("float32"))


def init_t_xy(end_x: int, end_y: int):
    t = paddle.arange(end=end_x * end_y)
    t_x = (t % end_x).astype(dtype="float32")
    t_y = paddle.floor(paddle.divide(x=t.astype(dtype="float32"), y=paddle.to_tensor(end_x, dtype="float32"))).astype(
        dtype="float32"
    )
    return t_x, t_y


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 10000.0):
    freqs_x = 1.0 / theta ** (paddle.arange(start=0, end=dim, step=4)[: dim // 4].astype(dtype="float32") / dim)
    freqs_y = 1.0 / theta ** (paddle.arange(start=0, end=dim, step=4)[: dim // 4].astype(dtype="float32") / dim)
    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = paddle.outer(t_x, freqs_x)
    freqs_y = paddle.outer(t_y, freqs_y)
    freqs_cis_x = paddle.complex(
        paddle.ones_like(x=freqs_x) * paddle.cos(freqs_x), paddle.ones_like(x=freqs_x) * paddle.sin(freqs_x)
    )
    freqs_cis_y = paddle.complex(
        paddle.ones_like(x=freqs_y) * paddle.cos(freqs_y), paddle.ones_like(x=freqs_y) * paddle.sin(freqs_y)
    )
    return paddle.concat(x=[freqs_cis_x, freqs_cis_y], axis=-1)


def reshape_for_broadcast(freqs_cis: paddle.Tensor, x: paddle.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert tuple(freqs_cis.shape) == (tuple(x.shape)[-2], tuple(x.shape)[-1])
    shape = [(d if i >= ndim - 2 else 1) for i, d in enumerate(tuple(x.shape))]
    return freqs_cis.reshape(shape)


def apply_rotary_enc(xq: paddle.Tensor, xk: paddle.Tensor, freqs_cis: paddle.Tensor, repeat_freqs_k: bool = False):
    xq_ = paddle.as_complex(x=xq.astype(dtype="float32").reshape([*tuple(xq.shape)[:-1], -1, 2]))
    xk_ = (
        paddle.as_complex(x=xk.astype(dtype="float32").reshape([*tuple(xk.shape)[:-1], -1, 2]))
        if tuple(xk.shape)[-2] != 0
        else None
    )
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = paddle.as_real(x=xq_ * freqs_cis).flatten(start_axis=3)
    if xk_ is None:
        return xq_out.astype(dtype=xq.dtype), xk
    if repeat_freqs_k:
        r = tuple(xk_.shape)[-2] // tuple(xq_.shape)[-2]
        if "gpu" in str(freqs_cis.place):
            freqs_cis = freqs_cis.tile((*([1] * (freqs_cis.ndim - 2)), r, 1))
        else:
            freqs_cis = (
                freqs_cis.unsqueeze(axis=2).expand(shape=[-1, -1, r, -1, -1]).flatten(start_axis=2, stop_axis=3)
            )
    xk_out = paddle.as_real(x=xk_ * freqs_cis).flatten(start_axis=3)
    return xq_out.astype(dtype=xq.dtype), xk_out.astype(dtype=xk.dtype)
