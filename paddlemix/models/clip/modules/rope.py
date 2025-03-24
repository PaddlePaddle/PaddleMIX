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

import logging
from math import pi

import paddle


def broadcast(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, "tensors must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = dim + shape_len if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatenation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(shape=t[1]), zip(tensors, expandable_shapes)))
    return paddle.concat(x=tensors, axis=dim)


def rotate_half(x):
    x = x.reshape(list(x.shape)[:-1] + [-1, 2])
    x1, x2 = x.unbind(axis=-1)
    x = paddle.stack(x=(-x2, x1), axis=-1)
    return x.reshape(list(x.shape)[:-2] + [-1])


class VisionRotaryEmbedding(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / theta ** (paddle.arange(start=0, end=dim, step=2)[: dim // 2].astype(dtype="float32") / dim)
        elif freqs_for == "pixel":
            freqs = paddle.linspace(start=1.0, stop=max_freq / 2, num=dim // 2) * pi
        elif freqs_for == "constant":
            freqs = paddle.ones(shape=num_freqs).astype(dtype="float32")
        else:
            raise ValueError(f"unknown modality {freqs_for}")
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = paddle.arange(end=ft_seq_len) / ft_seq_len * pt_seq_len
        freqs_h = paddle.einsum("..., f -> ... f", t, freqs)
        freqs_h = freqs_h.repeat_interleave(2, axis=-1)
        freqs_w = paddle.einsum("..., f -> ... f", t, freqs)
        freqs_w = freqs_w.repeat_interleave(2, axis=-1)
        freqs = broadcast((freqs_h[:, (None), :], freqs_w[(None), :, :]), dim=-1)
        self.register_buffer("freqs_cos", freqs.cos(), persistable=False)
        self.register_buffer("freqs_sin", freqs.sin(), persistable=False)
        logging.info(f"Shape of rope freq: {self.freqs_cos.shape}")

    def forward(self, t, start_index=0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert (
            rot_dim <= t.shape[-1]
        ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
        t_left, t, t_right = (
            t[(...), :start_index],
            t[(...), start_index:end_index],
            t[(...), end_index:],
        )
        t = t * self.freqs_cos + rotate_half(t) * self.freqs_sin
        return paddle.concat(x=(t_left, t, t_right), axis=-1)


class VisionRotaryEmbeddingFast(paddle.nn.Layer):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
        patch_dropout=0.0,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / theta ** (paddle.arange(start=0, end=dim, step=2)[: dim // 2].astype(dtype="float32") / dim)
        elif freqs_for == "pixel":
            freqs = paddle.linspace(start=1.0, stop=max_freq / 2, num=dim // 2) * pi
        elif freqs_for == "constant":
            freqs = paddle.ones(shape=num_freqs).astype(dtype="float32")
        else:
            raise ValueError(f"unknown modality {freqs_for}")
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = paddle.arange(end=ft_seq_len) / ft_seq_len * pt_seq_len
        freqs = paddle.einsum("..., f -> ... f", t, freqs)
        freqs = freqs.repeat_interleave(2, axis=freqs.rank() - 1)
        freqs = broadcast((freqs[:, (None), :], freqs[(None), :, :]), dim=-1)
        freqs_cos = freqs.cos().reshape((-1, freqs.shape[-1]))
        freqs_sin = freqs.sin().reshape((-1, freqs.shape[-1]))
        self.patch_dropout = patch_dropout
        self.register_buffer("freqs_cos", freqs_cos, persistable=False)
        self.register_buffer("freqs_sin", freqs_sin, persistable=False)
        logging.info(f"Shape of rope freq: {self.freqs_cos.shape}")

    def forward(self, t, patch_indices_keep=None):
        if patch_indices_keep is not None:
            batch = t.shape[0]
            batch_indices = paddle.arange(end=batch)
            batch_indices = batch_indices[..., None]
            freqs_cos = self.freqs_cos.unsqueeze(0)
            freqs_cos = freqs_cos.unsqueeze(2)
            freqs_cos = freqs_cos.repeat_interleave(t.shape[0], axis=0)
            freqs_cos = freqs_cos.repeat_interleave(t.shape[1], axis=2)

            freqs_sin = self.freqs_sin.unsqueeze(0)
            freqs_sin = freqs_sin.unsqueeze(2)
            freqs_sin = freqs_sin.repeat_interleave(t.shape[0], axis=0)
            freqs_sin = freqs_sin.repeat_interleave(t.shape[1], axis=2)

            freqs_cos = freqs_cos[batch_indices, patch_indices_keep]
            freqs_cos = freqs_cos.transpose((0, 2, 1, 3))
            freqs_sin = freqs_sin[batch_indices, patch_indices_keep]
            freqs_sin = freqs_sin.transpose((0, 2, 1, 3))
            return t * freqs_cos + rotate_half(t) * freqs_sin
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin
