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

from math import pi

import paddle
from beartype import beartype
from beartype.typing import Literal, Optional, Union
from einops import rearrange, repeat


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def broadcast(tensors, dim=-1):
    broadcasted_tensors = paddle.broadcast_tensors(tensors)
    return paddle.concat(x=broadcasted_tensors, axis=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(axis=-1)
    x = paddle.stack(x=(-x2, x1), axis=-1)
    return rearrange(x, "... d r -> ... (d r)")


@paddle.amp.auto_cast(enable=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    if t.ndim == 3:
        seq_len = tuple(t.shape)[seq_dim]
        freqs = freqs[-seq_len:].to(t)
    rot_dim = tuple(freqs.shape)[-1]
    end_index = start_index + rot_dim
    assert (
        rot_dim <= tuple(t.shape)[-1]
    ), f"feature dimension {tuple(t.shape)[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
    t_left, t, t_right = t[(...), :start_index], t[(...), start_index:end_index], t[(...), end_index:]
    t = t * freqs.cos() * scale + rotate_half(t) * freqs.sin() * scale
    return paddle.concat(x=(t_left, t, t_right), axis=-1)


def apply_learned_rotations(rotations, t, start_index=0, freq_ranges=None):
    if exists(freq_ranges):
        rotations = paddle.einsum("..., f -> ... f", rotations, freq_ranges)
        rotations = rearrange(rotations, "... r f -> ... (r f)")
    rotations = repeat(rotations, "... n -> ... (n r)", r=2)
    return apply_rotary_emb(rotations, t, start_index=start_index)


class RotaryEmbedding(paddle.nn.Layer):
    @beartype
    def __init__(
        self,
        dim,
        custom_freqs: Optional[paddle.Tensor] = None,
        freqs_for: Union[Literal["lang"], Literal["pixel"], Literal["constant"]] = "lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        use_xpos=False,
        xpos_scale_base=512,
        interpolate_factor=1.0,
        theta_rescale_factor=1.0,
        seq_before_head_dim=False,
        cache_if_possible=True,
    ):
        super().__init__()
        theta *= theta_rescale_factor ** (dim / (dim - 2))
        self.freqs_for = freqs_for
        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / theta ** (paddle.arange(start=0, end=dim, step=2)[: dim // 2].astype(dtype="float32") / dim)
        elif freqs_for == "pixel":
            freqs = paddle.linspace(start=1.0, stop=max_freq / 2, num=dim // 2) * pi
        elif freqs_for == "constant":
            freqs = paddle.ones(shape=num_freqs).astype(dtype="float32")
        self.cache_if_possible = cache_if_possible
        self.tmp_store("cached_freqs", None)
        self.tmp_store("cached_scales", None)
        out_6 = paddle.create_parameter(
            shape=freqs.shape, dtype=freqs.numpy().dtype, default_initializer=paddle.nn.initializer.Assign(freqs)
        )
        out_6.stop_gradient = not learned_freq
        self.freqs = out_6
        self.learned_freq = learned_freq
        self.tmp_store("dummy", paddle.to_tensor(data=0))
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2
        assert interpolate_factor >= 1.0
        self.interpolate_factor = interpolate_factor
        self.use_xpos = use_xpos
        if not use_xpos:
            self.tmp_store("scale", None)
            return
        scale = (paddle.arange(start=0, end=dim, step=2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.tmp_store("scale", scale)

    @property
    def device(self):
        return self.dummy.place

    def tmp_store(self, key, value):
        self.register_buffer(name=key, tensor=value, persistable=False)

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (paddle.arange(dtype=dtype, end=seq_len) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0, freq_seq_len=None):
        seq_dim = default(seq_dim, self.default_seq_dim)
        assert (
            not self.use_xpos
        ), "you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings"
        device, dtype, seq_len = t.place, t.dtype, tuple(t.shape)[seq_dim]
        if exists(freq_seq_len):
            assert freq_seq_len >= seq_len
            seq_len = freq_seq_len
        freqs = self.forward(
            self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset), seq_len=seq_len, offset=offset
        )
        if seq_dim == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")
        return apply_rotary_emb(freqs, t, seq_dim=seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim=None, offset=0):
        seq_dim = default(seq_dim, self.default_seq_dim)
        q_len, k_len = tuple(q.shape)[seq_dim], tuple(k.shape)[seq_dim]
        assert q_len <= k_len
        rotated_q = self.rotate_queries_or_keys(q, seq_dim=seq_dim, freq_seq_len=k_len)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim=seq_dim)
        rotated_q = rotated_q.astype(q.dtype)
        rotated_k = rotated_k.astype(k.dtype)
        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim=None):
        seq_dim = default(seq_dim, self.default_seq_dim)
        assert self.use_xpos
        device, dtype, seq_len = q.place, q.dtype, tuple(q.shape)[seq_dim]
        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)
        freqs = self.forward(seq, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)
        if seq_dim == -3:
            freqs = rearrange(freqs, "n d -> n 1 d")
            scale = rearrange(scale, "n d -> n 1 d")
        rotated_q = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale**-1, seq_dim=seq_dim)
        rotated_q = rotated_q.astype(q.dtype)
        rotated_k = rotated_k.astype(k.dtype)
        return rotated_q, rotated_k

    @beartype
    def get_scale(self, t: paddle.Tensor, seq_len: Optional[int] = None, offset=0):
        assert self.use_xpos
        should_cache = self.cache_if_possible and exists(seq_len)
        if should_cache and exists(self.cached_scales) and seq_len + offset <= tuple(self.cached_scales.shape)[0]:
            return self.cached_scales[offset : offset + seq_len]
        scale = 1.0
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, "n -> n 1")
            scale = paddle.concat(x=(scale, scale), axis=-1)
        if should_cache:
            self.tmp_store("cached_scales", scale)
        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []
        for ind, dim in enumerate(dims):
            if self.freqs_for == "pixel":
                pos = paddle.linspace(start=-1, stop=1, num=dim)
            else:
                pos = paddle.arange(end=dim)
            freqs = self.forward(pos, seq_len=dim)
            all_axis = [None] * len(dims)
            all_axis[ind] = Colon
            new_axis_slice = Ellipsis, *all_axis, Colon
            all_freqs.append(freqs[new_axis_slice])
        all_freqs = paddle.broadcast_tensors(all_freqs)
        return paddle.concat(x=all_freqs, axis=-1)

    @paddle.amp.auto_cast(enable=False)
    def forward(self, t: paddle.Tensor, seq_len=None, offset=0):
        should_cache = (
            self.cache_if_possible and not self.learned_freq and exists(seq_len) and self.freqs_for != "pixel"
        )
        if should_cache and exists(self.cached_freqs) and offset + seq_len <= tuple(self.cached_freqs.shape)[0]:
            return self.cached_freqs[offset : offset + seq_len].detach()
        freqs = self.freqs
        freqs = paddle.einsum("..., f -> ... f", t.astype(freqs.dtype), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        if should_cache:
            self.tmp_store("cached_freqs", freqs.detach())
        return freqs
