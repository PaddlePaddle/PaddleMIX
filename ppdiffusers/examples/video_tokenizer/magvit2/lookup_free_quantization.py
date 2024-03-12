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

from collections import namedtuple
from math import ceil, log2

import paddle
from einops import pack, rearrange, reduce, unpack

Return = namedtuple("Return", ["quantized", "indices", "entropy_aux_loss"])
LossBreakdown = namedtuple("LossBreakdown", ["per_sample_entropy", "batch_entropy", "commitment"])


def exists(v):
    return v is not None


def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def log(t, eps=1e-05):
    return t.clip(min=eps).log()


def entropy(prob):
    return (-prob * log(prob)).sum(axis=-1)


class LFQ(paddle.nn.Layer):
    def __init__(
        self,
        *,
        dim=None,
        codebook_size=None,
        entropy_loss_weight=0.1,
        commitment_loss_weight=0.25,
        diversity_gamma=1.0,
        straight_through_activation=paddle.nn.Identity(),
        num_codebooks=1,
        keep_num_codebooks_dim=None,
        codebook_scale=1.0,
        frac_per_sample_entropy=1.0
    ):
        super().__init__()
        assert exists(dim) or exists(codebook_size), "either dim or codebook_size must be specified for LFQ"
        assert (
            not exists(codebook_size) or log2(codebook_size).is_integer()
        ), f"your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})"
        codebook_size = default(codebook_size, lambda: 2**dim)
        codebook_dim = int(log2(codebook_size))
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)
        has_projections = dim != codebook_dims
        self.project_in = (
            paddle.nn.Linear(in_features=dim, out_features=codebook_dims) if has_projections else paddle.nn.Identity()
        )
        self.project_out = (
            paddle.nn.Linear(in_features=codebook_dims, out_features=dim) if has_projections else paddle.nn.Identity()
        )
        self.has_projections = has_projections
        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks
        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim
        self.activation = straight_through_activation
        assert 0 < frac_per_sample_entropy <= 1.0
        self.frac_per_sample_entropy = frac_per_sample_entropy
        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight
        self.codebook_scale = codebook_scale
        self.commitment_loss_weight = commitment_loss_weight
        self.register_buffer(name="mask", tensor=2 ** paddle.arange(start=codebook_dim - 1, end=-1, step=-1))
        self.register_buffer(name="zero", tensor=paddle.to_tensor(data=0.0), persistable=False)
        all_codes = paddle.arange(end=codebook_size)
        bits = (all_codes[..., None] & self.mask != 0).astype(dtype="float32")
        codebook = self.bits_to_codes(bits)
        self.register_buffer(name="codebook", tensor=codebook, persistable=False)

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    def dtype(self):
        return self.codebook.dtype

    def indices_to_codes(self, indices, project_out=True):
        is_img_or_video = indices.ndim >= 3 + int(self.keep_num_codebooks_dim)
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... -> ... 1")

        bits = (indices[..., None] & self.mask != 0).astype(self.dtype)
        codes = self.bits_to_codes(bits)
        codes = rearrange(codes, "... c d -> ... (c d)")
        if project_out:
            codes = self.project_out(codes)
        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")
        return codes

    @paddle.amp.auto_cast(enable=False)
    def forward(self, x, inv_temperature=100.0, return_loss_breakdown=False, mask=None):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """
        x = x.astype(dtype="float32")
        is_img_or_video = x.ndim >= 4
        if is_img_or_video:
            x = rearrange(x, "b d ... -> b ... d")
            x, ps = pack_one(x, "b * d")
        assert tuple(x.shape)[-1] == self.dim, f"expected dimension of {self.dim} but received {tuple(x.shape)[-1]}"
        x = self.project_in(x)
        x = rearrange(x, "b n (c d) -> b n c d", c=self.num_codebooks)
        original_input = x
        codebook_value = paddle.ones_like(x=x) * self.codebook_scale
        quantized = paddle.where(condition=x > 0, x=codebook_value, y=-codebook_value)

        if self.training:
            x = self.activation(x)
            x = x + (quantized - x).detach()
        else:
            x = quantized

        indices = reduce((x > 0).astype(dtype="int32") * self.mask.astype(dtype="int32"), "b n c d -> b n c", "sum")
        if self.training:

            distance = -2 * paddle.matmul(original_input, self.codebook, transpose_y=True)

            prob = paddle.nn.functional.softmax(-distance * inv_temperature, axis=-1)

            if exists(mask):
                prob = prob[mask]
            else:
                prob = rearrange(prob, "b n ... -> (b n) ...")
            if self.frac_per_sample_entropy < 1.0:
                num_tokens = tuple(prob.shape)[0]
                num_sampled_tokens = int(num_tokens * self.frac_per_sample_entropy)
                rand_mask = paddle.randn(shape=num_tokens).argsort(axis=-1) < num_sampled_tokens
                per_sample_probs = prob[rand_mask]
            else:
                per_sample_probs = prob
            per_sample_entropy = entropy(per_sample_probs).mean()
            avg_prob = reduce(per_sample_probs, "... c d -> c d", "mean")
            codebook_entropy = entropy(avg_prob).mean()

            entropy_aux_loss = per_sample_entropy - self.diversity_gamma * codebook_entropy
        else:
            entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero
        if self.training:
            commit_loss = paddle.nn.functional.mse_loss(
                input=original_input, label=quantized.detach(), reduction="none"
            )
            if exists(mask):
                commit_loss = commit_loss[mask]
            commit_loss = commit_loss.mean()
        else:
            commit_loss = self.zero
        x = rearrange(x, "b n c d -> b n (c d)")
        x = self.project_out(x)
        if is_img_or_video:
            x = unpack_one(x, ps, "b * d")
            x = rearrange(x, "b ... d -> b d ...")
            indices = unpack_one(indices, ps, "b * c")
        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")
        aux_loss = entropy_aux_loss * self.entropy_loss_weight + commit_loss * self.commitment_loss_weight

        ret = Return(x, indices, aux_loss)

        if not return_loss_breakdown:
            return ret
        return ret, LossBreakdown(per_sample_entropy, codebook_entropy, commit_loss)
