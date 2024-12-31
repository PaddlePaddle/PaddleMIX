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

from typing import List, Optional

import paddle
from einops import pack, rearrange, unpack


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def round_ste(z: paddle.Tensor) -> paddle.Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


class FSQ(paddle.nn.Layer):
    def __init__(
        self,
        levels: List[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
    ):
        super().__init__()
        _levels = paddle.to_tensor(data=levels, dtype="int32")
        self.register_buffer(name="_levels", tensor=_levels, persistable=False)
        _basis = paddle.cumprod(x=paddle.to_tensor(data=[1] + levels[:-1]), dim=0, dtype="int32")
        self.register_buffer(name="_basis", tensor=_basis, persistable=False)
        self.scale = scale
        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim
        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim
        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim
        self.dim = default(dim, len(_levels) * num_codebooks)
        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            paddle.nn.Linear(in_features=self.dim, out_features=effective_codebook_dim)
            if has_projections
            else paddle.nn.Identity()
        )
        self.project_out = (
            paddle.nn.Linear(in_features=effective_codebook_dim, out_features=self.dim)
            if has_projections
            else paddle.nn.Identity()
        )
        self.has_projections = has_projections
        self.codebook_size = self._levels.prod().item()
        implicit_codebook = self.indices_to_codes(paddle.arange(end=self.codebook_size), project_out=False)
        self.register_buffer(name="implicit_codebook", tensor=implicit_codebook, persistable=False)

    def bound(self, z: paddle.Tensor, eps: float = 0.001) -> paddle.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = paddle.where(condition=self._levels % 2 == 0, x=0.5, y=0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: paddle.Tensor) -> paddle.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: paddle.Tensor) -> paddle.Tensor:
        half_width = self._levels // 2
        return zhat_normalized * half_width + half_width

    def _scale_and_shift_inverse(self, zhat: paddle.Tensor) -> paddle.Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: paddle.Tensor) -> paddle.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert tuple(zhat.shape)[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat)
        return (zhat * self._basis).sum(axis=-1).astype("int32")

    def indices_to_codes(self, indices: paddle.Tensor, project_out=True) -> paddle.Tensor:
        """Inverse of `codes_to_indices`."""
        is_img_or_video = indices.ndim >= 3 + int(self.keep_num_codebooks_dim)
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = indices // self._basis % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)
        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")
        if project_out:
            codes = self.project_out(codes)
        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")
        return codes

    @paddle.amp.auto_cast(enable=False)
    def forward(self, z: paddle.Tensor) -> paddle.Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension
        c - number of codebook dim
        """
        is_img_or_video = z.ndim >= 4
        if is_img_or_video:
            z = rearrange(z, "b d ... -> b ... d")
            z, ps = pack_one(z, "b * d")
        assert (
            tuple(z.shape)[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {tuple(z.shape)[-1]}"
        z = self.project_in(z)
        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)
        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        codes = rearrange(codes, "b n c d -> b n (c d)")
        out = self.project_out(codes)
        if is_img_or_video:
            out = unpack_one(out, ps, "b * d")
            out = rearrange(out, "b ... d -> b d ...")
            indices = unpack_one(indices, ps, "b * c")
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")
        return out, indices
