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

import copy
import pickle
from collections import namedtuple
from functools import partial, wraps
from math import log2
from pathlib import Path

import paddle
from attend import Attend
from beartype import beartype
from beartype.typing import List, Optional, Tuple, Union
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.paddle import Rearrange
from filter3d import filter3d
from finite_scalar_quantization import FSQ
from lookup_free_quantization import LFQ
from simplified_gate_loop import SimpleGateLoopLayer
from taylor_series_linear_attention import TaylorSeriesLinearAttn
from utils import _FUNCTIONAL_PAD


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def safe_get_index(it, ind, default=None):
    if ind < len(it):
        return it[ind]
    return default


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def identity(t, *args, **kwargs):
    return t


def divisible_by(num, den):
    return num % den == 0


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def append_dims(t, ndims: int):
    return t.reshape(*tuple(t.shape), *((1,) * ndims))


def is_odd(n):
    return not divisible_by(n, 2)


def maybe_del_attr_(o, attr):
    if hasattr(o, attr):
        delattr(o, attr)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else (t,) * length


def l2norm(t):
    return paddle.nn.functional.normalize(x=t, axis=-1, p=2)


def pad_at_dim(t, pad, dim=-1, value=0.0):
    dims_from_right = -dim - 1 if dim < 0 else t.ndim - dim - 1
    zeros = (0, 0) * dims_from_right
    return _FUNCTIONAL_PAD(pad=(*zeros, *pad), value=value, x=t, data_format="NCDHW")


def pick_video_frame(video, frame_indices):
    batch = tuple(video.shape)[0]
    video = rearrange(video, "b c f ... -> b f c ...")
    batch_indices = paddle.arange(end=batch)
    batch_indices = rearrange(batch_indices, "b -> b 1")
    images = video[batch_indices, frame_indices]
    images = rearrange(images, "b 1 c ... -> b c ...")
    return images


def gradient_penalty(images, output):
    gradients = paddle.grad(
        outputs=output,
        inputs=images,
        grad_outputs=paddle.ones(shape=tuple(output.shape)),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = rearrange(gradients, "b ... -> b (...)")
    return ((gradients.norm(p=2, axis=1) - 1) ** 2).mean()


def leaky_relu(p=0.1):
    return paddle.nn.LeakyReLU(negative_slope=p)


def hinge_discr_loss(fake, real):
    return (paddle.nn.functional.relu(x=1 + fake) + paddle.nn.functional.relu(x=1 - real)).mean()


def hinge_gen_loss(fake):
    return -fake.mean()


@paddle.amp.auto_cast(enable=False)
@beartype
def grad_layer_wrt_loss(loss: paddle.Tensor, layer: paddle.Tensor):
    return paddle.grad(outputs=loss, inputs=layer, grad_outputs=paddle.ones_like(x=loss), retain_graph=True)[
        0
    ].detach()


def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, "vgg")
        if has_vgg:
            vgg = self.vgg
            delattr(self, "vgg")
        out = fn(self, *args, **kwargs)
        if has_vgg:
            self.vgg = vgg
        return out

    return inner


def Sequential(*modules):
    modules = [*filter(exists, modules)]
    if len(modules) == 0:
        return paddle.nn.Identity()
    return paddle.nn.Sequential(*modules)


class Residual(paddle.nn.Layer):
    @beartype
    def __init__(self, fn: paddle.nn.Layer):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class ToTimeSequence(paddle.nn.Layer):
    @beartype
    def __init__(self, fn: paddle.nn.Layer):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = rearrange(x, "b c f ... -> b ... f c")
        x, ps = pack_one(x, "* n c")
        o = self.fn(x, **kwargs)
        o = unpack_one(o, ps, "* n c")
        return rearrange(o, "b ... f c -> b c f ...")


class SqueezeExcite(paddle.nn.Layer):
    def __init__(self, dim, *, dim_out=None, dim_hidden_min=16, init_bias=-10):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.to_k = paddle.nn.Conv2D(in_channels=dim, out_channels=1, kernel_size=1)
        dim_hidden = max(dim_hidden_min, dim_out // 2)
        self.net = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=dim, out_channels=dim_hidden, kernel_size=1),
            paddle.nn.LeakyReLU(negative_slope=0.1),
            paddle.nn.Conv2D(in_channels=dim_hidden, out_channels=dim_out, kernel_size=1),
            paddle.nn.Sigmoid(),
        )
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(self.net[-2].weight)
        init_Constant = paddle.nn.initializer.Constant(value=init_bias)
        init_Constant(self.net[-2].bias)

    def forward(self, x):
        orig_input, batch = x, tuple(x.shape)[0]
        is_video = x.ndim == 5
        if is_video:
            x = rearrange(x, "b c f h w -> (b f) c h w")
        context = self.to_k(x)
        context = paddle.nn.functional.softmax(rearrange(context, "b c h w -> b c (h w)"), axis=-1)
        spatial_flattened_input = rearrange(x, "b c h w -> b c (h w)")
        out = paddle.einsum("b i n, b c n -> b c i", context, spatial_flattened_input)
        out = rearrange(out, "... -> ... 1")
        gates = self.net(out)
        if is_video:
            gates = rearrange(gates, "(b f) c h w -> b c f h w", b=batch)
        return gates * orig_input


class TokenShift(paddle.nn.Layer):
    @beartype
    def __init__(self, fn: paddle.nn.Layer):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        x, x_shift = paddle.chunk(x, chunks=2, axis=1)
        x_shift = pad_at_dim(x_shift, (1, -1), dim=2)

        x = paddle.concat(x=(x, x_shift), axis=1)

        return self.fn(x, **kwargs)


class RMSNorm(paddle.nn.Layer):
    def __init__(self, dim, channel_first=False, images=False, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = paddle.create_parameter(
            shape=paddle.ones(shape=shape).shape,
            dtype=paddle.ones(shape=shape).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.ones(shape=shape)),
        )
        self.gamma.stop_gradient = not True
        out_8 = paddle.create_parameter(
            shape=paddle.zeros(shape=shape).shape,
            dtype=paddle.zeros(shape=shape).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=shape)),
        )
        out_8.stop_gradient = not True
        self.bias = out_8 if bias else 0.0

    def forward(self, x):
        return (
            paddle.nn.functional.normalize(x=x, axis=1 if self.channel_first else -1) * self.scale * self.gamma
            + self.bias
        )


class AdaptiveRMSNorm(paddle.nn.Layer):
    def __init__(self, dim, *, dim_cond, channel_first=False, images=False, bias=False):
        super().__init__()
        self.dim_cond = dim_cond
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.to_gamma = paddle.nn.Linear(in_features=dim_cond, out_features=dim)
        self.to_bias = paddle.nn.Linear(in_features=dim_cond, out_features=dim) if bias else None
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(self.to_gamma.weight)
        init_Constant = paddle.nn.initializer.Constant(value=1.0)
        init_Constant(self.to_gamma.bias)
        if bias:
            init_Constant = paddle.nn.initializer.Constant(value=0.0)
            init_Constant(self.to_bias.weight)
            init_Constant = paddle.nn.initializer.Constant(value=0.0)
            init_Constant(self.to_bias.bias)

    @beartype
    def forward(self, x: paddle.Tensor, *, cond: paddle.Tensor):
        batch = tuple(x.shape)[0]
        assert tuple(cond.shape) == (batch, self.dim_cond)
        gamma = self.to_gamma(cond)
        bias = 0.0
        if exists(self.to_bias):
            bias = self.to_bias(cond)
        if self.channel_first:
            gamma = append_dims(gamma, x.ndim - 2)
            if exists(self.to_bias):
                bias = append_dims(bias, x.ndim - 2)
        return paddle.nn.functional.normalize(x=x, axis=1 if self.channel_first else -1) * self.scale * gamma + bias


class Attention(paddle.nn.Layer):
    @beartype
    def __init__(
        self,
        *,
        dim,
        dim_cond: Optional[int] = None,
        causal=False,
        dim_head=32,
        heads=8,
        flash=False,
        dropout=0.0,
        num_memory_kv=4
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.need_cond = exists(dim_cond)
        if self.need_cond:
            self.norm = AdaptiveRMSNorm(dim, dim_cond=dim_cond)
        else:
            self.norm = RMSNorm(dim)
        self.to_qkv = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=dim, out_features=dim_inner * 3, bias_attr=False),
            Rearrange("b n (qkv h d) -> qkv b h n d", qkv=3, h=heads),
        )
        assert num_memory_kv > 0
        self.mem_kv = paddle.create_parameter(
            shape=paddle.randn(shape=[2, heads, num_memory_kv, dim_head]).shape,
            dtype=paddle.randn(shape=[2, heads, num_memory_kv, dim_head]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.randn(shape=[2, heads, num_memory_kv, dim_head])),
        )
        self.mem_kv.stop_gradient = not True

        self.attend = Attend(causal=causal, dropout=dropout, flash=flash)
        self.to_out = paddle.nn.Sequential(
            Rearrange("b h n d -> b n (h d)"),
            paddle.nn.Linear(in_features=dim_inner, out_features=dim, bias_attr=False),
        )

    @beartype
    def forward(self, x, mask: Optional[paddle.Tensor] = None, cond: Optional[paddle.Tensor] = None):
        maybe_cond_kwargs = dict(cond=cond) if self.need_cond else dict()
        x = self.norm(x, **maybe_cond_kwargs)
        q, k, v = self.to_qkv(x)
        mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=tuple(q.shape)[0]), self.mem_kv)
        k = paddle.concat(x=(mk, k), axis=-2)
        v = paddle.concat(x=(mv, v), axis=-2)
        out = self.attend(q, k, v, mask=mask)
        return self.to_out(out)


class LinearAttention(paddle.nn.Layer):
    @beartype
    def __init__(self, *, dim, dim_cond: Optional[int] = None, dim_head=8, heads=8, dropout=0.0):
        super().__init__()

        self.need_cond = exists(dim_cond)
        if self.need_cond:
            self.norm = AdaptiveRMSNorm(dim, dim_cond=dim_cond)
        else:
            self.norm = RMSNorm(dim)
        self.attn = TaylorSeriesLinearAttn(dim=dim, dim_head=dim_head, heads=heads)

    def forward(self, x, cond: Optional[paddle.Tensor] = None):
        maybe_cond_kwargs = dict(cond=cond) if self.need_cond else dict()
        x = self.norm(x, **maybe_cond_kwargs)
        return self.attn(x)


class LinearSpaceAttention(LinearAttention):
    def forward(self, x, *args, **kwargs):
        x = rearrange(x, "b c ... h w -> b ... h w c")
        x, batch_ps = pack_one(x, "* h w c")
        x, seq_ps = pack_one(x, "b * c")
        x = super().forward(x, *args, **kwargs)
        x = unpack_one(x, seq_ps, "b * c")
        x = unpack_one(x, batch_ps, "* h w c")
        return rearrange(x, "b ... h w c -> b c ... h w")


class SpaceAttention(Attention):
    def forward(self, x, *args, **kwargs):
        x = rearrange(x, "b c t h w -> b t h w c")
        x, batch_ps = pack_one(x, "* h w c")
        x, seq_ps = pack_one(x, "b * c")
        x = super().forward(x, *args, **kwargs)
        x = unpack_one(x, seq_ps, "b * c")
        x = unpack_one(x, batch_ps, "* h w c")
        return rearrange(x, "b t h w c -> b c t h w")


class TimeAttention(Attention):
    def forward(self, x, *args, **kwargs):
        x = rearrange(x, "b c t h w -> b h w t c")
        x, batch_ps = pack_one(x, "* t c")
        x = super().forward(x, *args, **kwargs)
        x = unpack_one(x, batch_ps, "* t c")
        return rearrange(x, "b h w t c -> b c t h w")


class GEGLU(paddle.nn.Layer):
    def forward(self, x):
        x, gate = x.chunk(chunks=2, axis=1)
        return paddle.nn.functional.gelu(x=gate) * x


class FeedForward(paddle.nn.Layer):
    @beartype
    def __init__(self, dim, *, dim_cond: Optional[int] = None, mult=4, images=False):
        super().__init__()
        conv_klass = paddle.nn.Conv2D if images else paddle.nn.Conv3D
        rmsnorm_klass = RMSNorm if not exists(dim_cond) else partial(AdaptiveRMSNorm, dim_cond=dim_cond)
        maybe_adaptive_norm_klass = partial(rmsnorm_klass, channel_first=True, images=images)
        dim_inner = int(dim * mult * 2 / 3)
        self.norm = maybe_adaptive_norm_klass(dim)
        self.net = Sequential(conv_klass(dim, dim_inner * 2, 1), GEGLU(), conv_klass(dim_inner, dim, 1))

    @beartype
    def forward(self, x: paddle.Tensor, *, cond: Optional[paddle.Tensor] = None):
        maybe_cond_kwargs = dict(cond=cond) if exists(cond) else dict()
        x = self.norm(x, **maybe_cond_kwargs)
        return self.net(x)


class Blur(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        f = paddle.to_tensor(data=[1, 2, 1], dtype="float32")
        self.register_buffer(name="f", tensor=f)

    def forward(self, x, space_only=False, time_only=False):
        assert not (space_only and time_only)
        f = self.f
        if space_only:
            f = paddle.einsum("i, j -> i j", f, f)
            f = rearrange(f, "... -> 1 1 ...")
        elif time_only:
            f = rearrange(f, "f -> 1 f 1 1")
        else:
            f = paddle.einsum("i, j, k -> i j k", f, f, f)
            f = rearrange(f, "... -> 1 ...")
        is_images = x.ndim == 4
        if is_images:
            x = rearrange(x, "b c h w -> b c 1 h w")
        out = filter3d(x, f, normalized=True)
        if is_images:
            out = rearrange(out, "b c 1 h w -> b c h w")
        return out


class DiscriminatorBlock(paddle.nn.Layer):
    def __init__(self, input_channels, filters, downsample=True, antialiased_downsample=True):
        super().__init__()
        self.conv_res = paddle.nn.Conv2D(
            in_channels=input_channels, out_channels=filters, kernel_size=1, stride=2 if downsample else 1
        )
        self.net = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=input_channels, out_channels=filters, kernel_size=3, padding=1),
            leaky_relu(),
            paddle.nn.Conv2D(in_channels=filters, out_channels=filters, kernel_size=3, padding=1),
            leaky_relu(),
        )
        self.maybe_blur = Blur() if antialiased_downsample else None
        self.downsample = (
            paddle.nn.Sequential(
                Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
                paddle.nn.Conv2D(in_channels=filters * 4, out_channels=filters, kernel_size=1),
            )
            if downsample
            else None
        )

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if exists(self.downsample):
            if exists(self.maybe_blur):
                x = self.maybe_blur(x, space_only=True)
            x = self.downsample(x)
        x = (x + res) * 2**-0.5
        return x


class Discriminator(paddle.nn.Layer):
    @beartype
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels=3,
        max_dim=512,
        attn_heads=8,
        attn_dim_head=32,
        linear_attn_dim_head=8,
        linear_attn_heads=16,
        ff_mult=4,
        antialiased_downsample=False
    ):
        super().__init__()
        image_size = pair(image_size)
        min_image_resolution = min(image_size)
        num_layers = int(log2(min_image_resolution) - 2)
        blocks = []
        layer_dims = [channels] + [(dim * 4 * 2**i) for i in range(num_layers + 1)]
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))
        blocks = []

        image_resolution = min_image_resolution
        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):

            is_not_last = ind != len(layer_dims_in_out) - 1
            block = DiscriminatorBlock(
                in_chan, out_chan, downsample=is_not_last, antialiased_downsample=antialiased_downsample
            )
            attn_block = Sequential(
                Residual(LinearSpaceAttention(dim=out_chan, heads=linear_attn_heads, dim_head=linear_attn_dim_head)),
                Residual(FeedForward(dim=out_chan, mult=ff_mult, images=True)),
            )
            blocks.append(paddle.nn.LayerList(sublayers=[block, attn_block]))
            image_resolution //= 2
        self.blocks = paddle.nn.LayerList(sublayers=blocks)
        dim_last = layer_dims[-1]
        downsample_factor = 2**num_layers
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))
        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last
        self.to_logits = Sequential(
            paddle.nn.Conv2D(in_channels=dim_last, out_channels=dim_last, kernel_size=3, padding=1),
            leaky_relu(),
            Rearrange("b ... -> b (...)"),
            paddle.nn.Linear(in_features=latent_dim, out_features=1),
            Rearrange("b 1 -> b"),
        )

    def forward(self, x):
        for block, attn_block in self.blocks:
            x = block(x)
            x = attn_block(x)
        return self.to_logits(x)


class Conv3DMod(paddle.nn.Layer):
    @beartype
    def __init__(
        self, dim, *, spatial_kernel, time_kernel, causal=True, dim_out=None, demod=True, eps=1e-08, pad_mode="zeros"
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.eps = eps
        assert is_odd(spatial_kernel) and is_odd(time_kernel)
        self.spatial_kernel = spatial_kernel
        self.time_kernel = time_kernel
        time_padding = (time_kernel - 1, 0) if causal else (time_kernel // 2,) * 2
        self.pad_mode = pad_mode
        self.padding = *((spatial_kernel // 2,) * 4), *time_padding
        out_10 = paddle.create_parameter(
            shape=paddle.randn(shape=(dim_out, dim, time_kernel, spatial_kernel, spatial_kernel)).shape,
            dtype=paddle.randn(shape=(dim_out, dim, time_kernel, spatial_kernel, spatial_kernel)).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.randn(shape=(dim_out, dim, time_kernel, spatial_kernel, spatial_kernel))
            ),
        )
        out_10.stop_gradient = not True
        self.weights = out_10
        self.demod = demod
        init_KaimingNormal = paddle.nn.initializer.KaimingNormal(nonlinearity="selu", negative_slope=0)
        init_KaimingNormal(self.weights)

    @beartype
    def forward(self, fmap, cond: paddle.Tensor):
        """
        notation

        b - batch
        n - convs
        o - output
        i - input
        k - kernel
        """
        b = tuple(fmap.shape)[0]
        weights = self.weights
        cond = rearrange(cond, "b i -> b 1 i 1 1 1")
        weights = weights * (cond + 1)
        if self.demod:
            inv_norm = reduce(weights**2, "b o i k0 k1 k2 -> b o 1 1 1 1", "sum").clamp(min=self.eps).rsqrt()
            weights = weights * inv_norm
        fmap = rearrange(fmap, "b c t h w -> 1 (b c) t h w")
        weights = rearrange(weights, "b o ... -> (b o) ...")
        fmap = _FUNCTIONAL_PAD(pad=self.padding, mode=self.pad_mode, x=fmap)
        fmap = paddle.nn.functional.conv3d(x=fmap, weight=weights, groups=b)
        return rearrange(fmap, "1 (b o) ... -> b o ...", b=b)


class SpatialDownsample2x(paddle.nn.Layer):
    def __init__(self, dim, dim_out=None, kernel_size=3, antialias=False):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.maybe_blur = Blur() if antialias else identity
        self.conv = paddle.nn.Conv2D(
            in_channels=dim, out_channels=dim_out, kernel_size=kernel_size, stride=2, padding=kernel_size // 2
        )

    def forward(self, x):
        x = self.maybe_blur(x, space_only=True)

        x = rearrange(x, "b c t h w -> b t c h w")
        x, ps = pack_one(x, "* c h w")

        out = self.conv(x)

        out = unpack_one(out, ps, "* c h w")
        out = rearrange(out, "b t c h w -> b c t h w")
        return out


class TimeDownsample2x(paddle.nn.Layer):
    def __init__(self, dim, dim_out=None, kernel_size=3, antialias=False):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.maybe_blur = Blur() if antialias else identity
        self.time_causal_padding = kernel_size - 1, 0
        self.conv = paddle.nn.Conv1D(in_channels=dim, out_channels=dim_out, kernel_size=kernel_size, stride=2)

    def forward(self, x):
        x = self.maybe_blur(x, time_only=True)
        x = rearrange(x, "b c t h w -> b h w c t")
        x, ps = pack_one(x, "* c t")

        x = _FUNCTIONAL_PAD(pad=self.time_causal_padding, x=x, data_format="NCL")
        out = self.conv(x)
        out = unpack_one(out, ps, "* c t")
        out = rearrange(out, "b h w c t -> b c t h w")
        return out


class SpatialUpsample2x(paddle.nn.Layer):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = paddle.nn.Conv2D(in_channels=dim, out_channels=dim_out * 4, kernel_size=1)

        self.net = paddle.nn.Sequential(
            conv, paddle.nn.Silu(), Rearrange("b (c p1 p2) h w -> b c (h p1) (w p2)", p1=2, p2=2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = paddle.empty(shape=[o // 4, i, h, w])
        init_KaimingUniform = paddle.nn.initializer.KaimingUniform(nonlinearity="leaky_relu")
        init_KaimingUniform(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 4) ...")
        with paddle.no_grad():
            conv.weight.set_value(conv_weight)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(conv.bias)

    def forward(self, x):
        x = rearrange(x, "b c t h w -> b t c h w")
        x, ps = pack_one(x, "* c h w")
        out = self.net(x)
        out = unpack_one(out, ps, "* c h w")
        out = rearrange(out, "b t c h w -> b c t h w")
        return out


class TimeUpsample2x(paddle.nn.Layer):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = paddle.nn.Conv1D(in_channels=dim, out_channels=dim_out * 2, kernel_size=1)
        self.net = paddle.nn.Sequential(conv, paddle.nn.Silu(), Rearrange("b (c p) t -> b c (t p)", p=2))
        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, t = conv.weight.shape
        conv_weight = paddle.empty(shape=[o // 2, i, t])
        init_KaimingUniform = paddle.nn.initializer.KaimingUniform(nonlinearity="leaky_relu")
        init_KaimingUniform(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 2) ...")
        with paddle.no_grad():
            conv.weight.set_value(conv_weight)
        init_Constant = paddle.nn.initializer.Constant(value=0.0)
        init_Constant(conv.bias.data)

    def forward(self, x):
        x = rearrange(x, "b c t h w -> b h w c t")
        x, ps = pack_one(x, "* c t")
        out = self.net(x)
        out = unpack_one(out, ps, "* c t")
        out = rearrange(out, "b h w c t -> b c t h w")
        return out


def SameConv2d(dim_in, dim_out, kernel_size):
    kernel_size = cast_tuple(kernel_size, 2)
    padding = [(k // 2) for k in kernel_size]
    return paddle.nn.Conv2D(in_channels=dim_in, out_channels=dim_out, kernel_size=kernel_size, padding=padding)


class CausalConv3d(paddle.nn.Layer):
    @beartype
    def __init__(
        self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], pad_mode="constant", **kwargs
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)
        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)
        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)
        stride = stride, 1, 1
        dilation = dilation, 1, 1
        self.conv = paddle.nn.Conv3D(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation)

    def forward(self, x):
        pad_mode = self.pad_mode if self.time_pad < tuple(x.shape)[2] else "constant"
        x = _FUNCTIONAL_PAD(pad=self.time_causal_padding, mode=pad_mode, x=x, data_format="NCDHW")
        return self.conv(x)


@beartype
def ResidualUnit(dim, kernel_size: Union[int, Tuple[int, int, int]], pad_mode: str = "constant"):
    net = Sequential(
        CausalConv3d(dim, dim, kernel_size, pad_mode=pad_mode),
        paddle.nn.ELU(),
        paddle.nn.Conv3D(in_channels=dim, out_channels=dim, kernel_size=1),
        paddle.nn.ELU(),
        SqueezeExcite(dim),
    )
    return Residual(net)


@beartype
class ResidualUnitMod(paddle.nn.Layer):
    def __init__(
        self, dim, kernel_size: Union[int, Tuple[int, int, int]], *, dim_cond, pad_mode: str = "constant", demod=True
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        assert height_kernel_size == width_kernel_size
        self.to_cond = paddle.nn.Linear(in_features=dim_cond, out_features=dim)
        self.conv = Conv3DMod(
            dim=dim,
            spatial_kernel=height_kernel_size,
            time_kernel=time_kernel_size,
            causal=True,
            demod=demod,
            pad_mode=pad_mode,
        )
        self.conv_out = paddle.nn.Conv3D(in_channels=dim, out_channels=dim, kernel_size=1)

    @beartype
    def forward(self, x, cond: paddle.Tensor):
        res = x
        cond = self.to_cond(cond)
        x = self.conv(x, cond=cond)
        x = paddle.nn.functional.elu(x=x)
        x = self.conv_out(x)
        x = paddle.nn.functional.elu(x=x)
        return x + res


class CausalConvTranspose3d(paddle.nn.Layer):
    def __init__(self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], *, time_stride, **kwargs):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)
        self.upsample_factor = time_stride
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2
        stride = time_stride, 1, 1
        padding = 0, height_pad, width_pad
        self.conv = paddle.nn.ConvTranspose3d(chan_in, chan_out, kernel_size, stride, padding=padding)

    def forward(self, x):
        assert x.ndim == 5
        t = tuple(x.shape)[2]
        out = self.conv(x)
        out = out[(...), : t * self.upsample_factor, :, :]
        return out


LossBreakdown = namedtuple(
    "LossBreakdown",
    [
        "recon_loss",
        "lfq_aux_loss",
        "quantizer_loss_breakdown",
        "perceptual_loss",
        "adversarial_gen_loss",
        "adaptive_adversarial_weight",
        "multiscale_gen_losses",
        "multiscale_gen_adaptive_weights",
    ],
)
DiscrLossBreakdown = namedtuple("DiscrLossBreakdown", ["discr_loss", "multiscale_discr_losses", "gradient_penalty"])


class VideoTokenizer(paddle.nn.Layer):
    @beartype
    def __init__(
        self,
        *,
        image_size,
        layers: Tuple[Union[str, Tuple[str, int]], ...] = ("residual", "residual", "residual"),
        residual_conv_kernel_size=3,
        num_codebooks=1,
        codebook_size: Optional[int] = None,
        channels=3,
        init_dim=64,
        max_dim=float("inf"),
        dim_cond=None,
        dim_cond_expansion_factor=4.0,
        input_conv_kernel_size: Tuple[int, int, int] = (7, 7, 7),
        output_conv_kernel_size: Tuple[int, int, int] = (3, 3, 3),
        pad_mode: str = "constant",
        lfq_entropy_loss_weight=0.1,
        lfq_commitment_loss_weight=1.0,
        lfq_diversity_gamma=2.5,
        quantizer_aux_loss_weight=1.0,
        lfq_activation=paddle.nn.Identity(),
        use_fsq=False,
        fsq_levels: Optional[List[int]] = None,
        attn_dim_head=32,
        attn_heads=8,
        attn_dropout=0.0,
        linear_attn_dim_head=8,
        linear_attn_heads=16,
        vgg: Optional[paddle.nn.Layer] = None,
        vgg_weights: Optional[str] = None,
        perceptual_loss_weight=0.1,
        discr_kwargs: Optional[dict] = None,
        multiscale_discrs: Tuple[paddle.nn.Layer, ...] = tuple(),
        use_gan=True,
        adversarial_loss_weight=1.0,
        grad_penalty_loss_weight=10.0,
        multiscale_adversarial_loss_weight=1.0,
        flash_attn=True,
        separate_first_frame_encoding=False
    ):
        super().__init__()
        _locals = locals()
        _locals.pop("self", None)
        _locals.pop("__class__", None)

        self._configs = pickle.dumps(_locals)
        self.channels = channels
        self.image_size = image_size
        self.conv_in = CausalConv3d(channels, init_dim, input_conv_kernel_size, pad_mode=pad_mode)
        self.conv_in_first_frame = paddle.nn.Identity()
        self.conv_out_first_frame = paddle.nn.Identity()
        if separate_first_frame_encoding:
            self.conv_in_first_frame = SameConv2d(channels, init_dim, input_conv_kernel_size[-2:])
            self.conv_out_first_frame = SameConv2d(init_dim, channels, output_conv_kernel_size[-2:])
        self.separate_first_frame_encoding = separate_first_frame_encoding
        self.encoder_layers = paddle.nn.LayerList(sublayers=[])
        self.decoder_layers = paddle.nn.LayerList(sublayers=[])
        self.conv_out = CausalConv3d(init_dim, channels, output_conv_kernel_size, pad_mode=pad_mode)
        dim = init_dim
        dim_out = dim
        layer_fmap_size = image_size
        time_downsample_factor = 1
        has_cond_across_layers = []
        for layer_def in layers:
            layer_type, *layer_params = cast_tuple(layer_def)
            has_cond = False
            if layer_type == "residual":
                encoder_layer = ResidualUnit(dim, residual_conv_kernel_size)
                decoder_layer = ResidualUnit(dim, residual_conv_kernel_size)
            elif layer_type == "consecutive_residual":
                (num_consecutive,) = layer_params
                encoder_layer = Sequential(
                    *[ResidualUnit(dim, residual_conv_kernel_size) for _ in range(num_consecutive)]
                )
                decoder_layer = Sequential(
                    *[ResidualUnit(dim, residual_conv_kernel_size) for _ in range(num_consecutive)]
                )
            elif layer_type == "cond_residual":
                assert exists(
                    dim_cond
                ), "dim_cond must be passed into VideoTokenizer, if tokenizer is to be conditioned"
                has_cond = True
                encoder_layer = ResidualUnitMod(
                    dim, residual_conv_kernel_size, dim_cond=int(dim_cond * dim_cond_expansion_factor)
                )
                decoder_layer = ResidualUnitMod(
                    dim, residual_conv_kernel_size, dim_cond=int(dim_cond * dim_cond_expansion_factor)
                )
                dim_out = dim
            elif layer_type == "compress_space":
                dim_out = safe_get_index(layer_params, 0)
                dim_out = default(dim_out, dim * 2)
                dim_out = min(dim_out, max_dim)
                encoder_layer = SpatialDownsample2x(dim, dim_out)
                decoder_layer = SpatialUpsample2x(dim_out, dim)
                assert layer_fmap_size > 1
                layer_fmap_size //= 2
            elif layer_type == "compress_time":
                dim_out = safe_get_index(layer_params, 0)
                dim_out = default(dim_out, dim * 2)
                dim_out = min(dim_out, max_dim)
                encoder_layer = TimeDownsample2x(dim, dim_out)
                decoder_layer = TimeUpsample2x(dim_out, dim)
                time_downsample_factor *= 2
            elif layer_type == "attend_space":
                attn_kwargs = dict(
                    dim=dim, dim_head=attn_dim_head, heads=attn_heads, dropout=attn_dropout, flash=flash_attn
                )
                encoder_layer = Sequential(Residual(SpaceAttention(**attn_kwargs)), Residual(FeedForward(dim)))
                decoder_layer = Sequential(Residual(SpaceAttention(**attn_kwargs)), Residual(FeedForward(dim)))
            elif layer_type == "linear_attend_space":
                linear_attn_kwargs = dict(dim=dim, dim_head=linear_attn_dim_head, heads=linear_attn_heads)
                encoder_layer = Sequential(
                    Residual(LinearSpaceAttention(**linear_attn_kwargs)), Residual(FeedForward(dim))
                )
                decoder_layer = Sequential(
                    Residual(LinearSpaceAttention(**linear_attn_kwargs)), Residual(FeedForward(dim))
                )
            elif layer_type == "gateloop_time":
                gateloop_kwargs = dict(use_heinsen=False)
                encoder_layer = ToTimeSequence(Residual(SimpleGateLoopLayer(dim=dim)))
                decoder_layer = ToTimeSequence(Residual(SimpleGateLoopLayer(dim=dim)))
            elif layer_type == "attend_time":
                attn_kwargs = dict(
                    dim=dim,
                    dim_head=attn_dim_head,
                    heads=attn_heads,
                    dropout=attn_dropout,
                    causal=True,
                    flash=flash_attn,
                )
                encoder_layer = Sequential(
                    Residual(TokenShift(TimeAttention(**attn_kwargs))),
                    Residual(TokenShift(FeedForward(dim, dim_cond=dim_cond))),
                )
                decoder_layer = Sequential(
                    Residual(TokenShift(TimeAttention(**attn_kwargs))),
                    Residual(TokenShift(FeedForward(dim, dim_cond=dim_cond))),
                )
            elif layer_type == "cond_attend_space":
                has_cond = True
                attn_kwargs = dict(
                    dim=dim,
                    dim_cond=dim_cond,
                    dim_head=attn_dim_head,
                    heads=attn_heads,
                    dropout=attn_dropout,
                    flash=flash_attn,
                )
                encoder_layer = Sequential(Residual(SpaceAttention(**attn_kwargs)), Residual(FeedForward(dim)))
                decoder_layer = Sequential(Residual(SpaceAttention(**attn_kwargs)), Residual(FeedForward(dim)))
            elif layer_type == "cond_linear_attend_space":
                has_cond = True
                attn_kwargs = dict(
                    dim=dim,
                    dim_cond=dim_cond,
                    dim_head=attn_dim_head,
                    heads=attn_heads,
                    dropout=attn_dropout,
                    flash=flash_attn,
                )
                encoder_layer = Sequential(
                    Residual(LinearSpaceAttention(**attn_kwargs)), Residual(FeedForward(dim, dim_cond=dim_cond))
                )
                decoder_layer = Sequential(
                    Residual(LinearSpaceAttention(**attn_kwargs)), Residual(FeedForward(dim, dim_cond=dim_cond))
                )
            elif layer_type == "cond_attend_time":
                has_cond = True
                attn_kwargs = dict(
                    dim=dim,
                    dim_cond=dim_cond,
                    dim_head=attn_dim_head,
                    heads=attn_heads,
                    dropout=attn_dropout,
                    causal=True,
                    flash=flash_attn,
                )
                encoder_layer = Sequential(
                    Residual(TokenShift(TimeAttention(**attn_kwargs))),
                    Residual(TokenShift(FeedForward(dim, dim_cond=dim_cond))),
                )
                decoder_layer = Sequential(
                    Residual(TokenShift(TimeAttention(**attn_kwargs))),
                    Residual(TokenShift(FeedForward(dim, dim_cond=dim_cond))),
                )
            else:
                raise ValueError(f"unknown layer type {layer_type}")
            self.encoder_layers.append(encoder_layer)
            if len(self.decoder_layers) == 0:
                self.decoder_layers.append(decoder_layer)
            else:
                self.decoder_layers.insert(0, decoder_layer)

            dim = dim_out
            has_cond_across_layers.append(has_cond)

        self.encoder_layers.append(
            Sequential(
                Rearrange("b c ... -> b ... c"),
                paddle.nn.LayerNorm(normalized_shape=dim),
                Rearrange("b ... c -> b c ..."),
            )
        )
        self.time_downsample_factor = time_downsample_factor
        self.time_padding = time_downsample_factor - 1
        self.fmap_size = layer_fmap_size
        self.has_cond_across_layers = has_cond_across_layers
        self.has_cond = any(has_cond_across_layers)
        self.encoder_cond_in = paddle.nn.Identity()
        self.decoder_cond_in = paddle.nn.Identity()
        if has_cond:
            self.dim_cond = dim_cond
            self.encoder_cond_in = Sequential(
                paddle.nn.Linear(in_features=dim_cond, out_features=int(dim_cond * dim_cond_expansion_factor)),
                paddle.nn.Silu(),
            )
            self.decoder_cond_in = Sequential(
                paddle.nn.Linear(in_features=dim_cond, out_features=int(dim_cond * dim_cond_expansion_factor)),
                paddle.nn.Silu(),
            )
        self.use_fsq = use_fsq
        if not use_fsq:
            assert exists(codebook_size) and not exists(
                fsq_levels
            ), "if use_fsq is set to False, `codebook_size` must be set (and not `fsq_levels`)"
            self.quantizers = LFQ(
                dim=dim,
                codebook_size=codebook_size,
                num_codebooks=num_codebooks,
                entropy_loss_weight=lfq_entropy_loss_weight,
                commitment_loss_weight=lfq_commitment_loss_weight,
                diversity_gamma=lfq_diversity_gamma,
            )
        else:
            assert not exists(codebook_size) and exists(
                fsq_levels
            ), "if use_fsq is set to True, `fsq_levels` must be set (and not `codebook_size`). the effective codebook size is the cumulative product of all the FSQ levels"
            self.quantizers = FSQ(fsq_levels, dim=dim, num_codebooks=num_codebooks)
        self.quantizer_aux_loss_weight = quantizer_aux_loss_weight
        self.register_buffer(name="zero", tensor=paddle.to_tensor(data=0.0), persistable=False)
        use_vgg = channels in {1, 3, 4} and perceptual_loss_weight > 0.0
        self.vgg = None
        self.perceptual_loss_weight = perceptual_loss_weight
        if use_vgg:

            if not exists(vgg):
                vgg = paddle.vision.models.vgg16(pretrained=True)
                vgg.classifier = Sequential(*vgg.classifier[:-2])
            self.vgg = vgg
        self.use_vgg = use_vgg
        self.use_gan = use_gan
        discr_kwargs = default(discr_kwargs, dict(dim=dim, image_size=image_size, channels=channels, max_dim=512))
        self.discr = Discriminator(**discr_kwargs)
        self.adversarial_loss_weight = adversarial_loss_weight
        self.grad_penalty_loss_weight = grad_penalty_loss_weight
        self.has_gan = use_gan and adversarial_loss_weight > 0.0
        self.has_multiscale_gan = use_gan and multiscale_adversarial_loss_weight > 0.0
        self.multiscale_discrs = paddle.nn.LayerList(sublayers=[*multiscale_discrs])
        self.multiscale_adversarial_loss_weight = multiscale_adversarial_loss_weight
        self.has_multiscale_discrs = (
            use_gan and multiscale_adversarial_loss_weight > 0.0 and len(multiscale_discrs) > 0
        )

    @property
    def device(self):
        return self.zero.place

    @classmethod
    def init_and_load_from(cls, path, strict=True):
        path = Path(path)
        assert path.exists()
        pkg = paddle.load(path=str(path))
        assert "config" in pkg, "model configs were not found in this saved checkpoint"
        config = pickle.loads(pkg["config"])
        tokenizer = cls(**config)
        tokenizer.load(path, strict=strict)
        return tokenizer

    def get_param(self, model):
        return [p for n, p in model.named_parameters()]

    def parameters(self):
        param = []
        param.extend(self.get_param(self.conv_in))
        param.extend(self.get_param(self.conv_in_first_frame))
        param.extend(self.get_param(self.conv_out_first_frame))
        param.extend(self.get_param(self.conv_out))
        param.extend(self.get_param(self.encoder_layers))
        param.extend(self.get_param(self.decoder_layers))
        param.extend(self.get_param(self.encoder_cond_in))
        param.extend(self.get_param(self.decoder_cond_in))
        param.extend(self.get_param(self.quantizers))
        return param

    def discr_parameters(self):
        return self.get_param(self.discr)

    def copy_for_eval(self):
        vae_copy = copy.deepcopy()
        maybe_del_attr_(vae_copy, "discr")
        maybe_del_attr_(vae_copy, "vgg")
        maybe_del_attr_(vae_copy, "multiscale_discrs")
        vae_copy.eval()
        return vae_copy

    @beartype
    def encode(
        self,
        video: paddle.Tensor,
        quantize=False,
        cond: Optional[paddle.Tensor] = None,
        video_contains_first_frame=True,
    ):
        encode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame
        if video_contains_first_frame:
            video_len = tuple(video.shape)[2]
            video = pad_at_dim(video, (self.time_padding, 0), value=0.0, dim=2)
            video_packed_shape = [tuple([self.time_padding]), tuple([]), tuple([video_len - 1])]
        assert not self.has_cond or exists(
            cond
        ), "`cond` must be passed into tokenizer forward method since conditionable layers were specified"
        if exists(cond):
            assert tuple(cond.shape) == (tuple(video.shape)[0], self.dim_cond)
            cond = self.encoder_cond_in(cond)
            cond_kwargs = dict(cond=cond)
        if encode_first_frame_separately:
            pad, first_frame, video = unpack(video, video_packed_shape, "b c * h w")
            first_frame = self.conv_in_first_frame(first_frame)
        video = self.conv_in(video)

        if encode_first_frame_separately:
            video, _ = pack([first_frame, video], "b c * h w")
            video = pad_at_dim(video, (self.time_padding, 0), dim=2)

        for fn, has_cond in zip(self.encoder_layers, self.has_cond_across_layers):
            layer_kwargs = dict()
            if has_cond:
                layer_kwargs = cond_kwargs
            video = fn(video, **layer_kwargs)

        maybe_quantize = identity if not quantize else self.quantizers
        return maybe_quantize(video)

    @beartype
    def decode_from_code_indices(
        self, codes: paddle.Tensor, cond: Optional[paddle.Tensor] = None, video_contains_first_frame=True
    ):
        assert codes.dtype in (paddle.int64, paddle.int32)
        if codes.ndim == 2:
            video_code_len = tuple(codes.shape)[-1]
            assert divisible_by(
                video_code_len, self.fmap_size**2
            ), f"flattened video ids must have a length ({video_code_len}) that is divisible by the fmap size ({self.fmap_size}) squared ({self.fmap_size ** 2})"
            codes = rearrange(codes, "b (f h w) -> b f h w", h=self.fmap_size, w=self.fmap_size)
        quantized = self.quantizers.indices_to_codes(codes)
        return self.decode(quantized, cond=cond, video_contains_first_frame=video_contains_first_frame)

    @beartype
    def decode(self, quantized: paddle.Tensor, cond: Optional[paddle.Tensor] = None, video_contains_first_frame=True):
        decode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame
        batch = tuple(quantized.shape)[0]
        assert not self.has_cond or exists(
            cond
        ), "`cond` must be passed into tokenizer forward method since conditionable layers were specified"
        if exists(cond):
            assert tuple(cond.shape) == (batch, self.dim_cond)
            cond = self.decoder_cond_in(cond)
            cond_kwargs = dict(cond=cond)
        x = quantized
        for fn, has_cond in zip(self.decoder_layers, reversed(self.has_cond_across_layers)):
            layer_kwargs = dict()
            if has_cond:
                layer_kwargs = cond_kwargs
            x = fn(x, **layer_kwargs)
        if decode_first_frame_separately:
            xff, x = (
                x[:, :, (self.time_padding)],
                x[:, :, self.time_padding + 1 :],
            )
            out = self.conv_out(x)
            outff = self.conv_out_first_frame(xff)
            video, _ = pack([outff, out], "b c * h w")
        else:
            video = self.conv_out(x)
            if video_contains_first_frame:
                video = video[:, :, self.time_padding :]
        return video

    @paddle.no_grad()
    def tokenize(self, video):
        self.eval()
        return self.forward(video, return_codes=True)

    @beartype
    def forward(
        self,
        video_or_images: paddle.Tensor,
        cond: Optional[paddle.Tensor] = None,
        return_loss=False,
        return_codes=False,
        return_recon=False,
        return_discr_loss=False,
        return_recon_loss_only=False,
        apply_gradient_penalty=True,
        video_contains_first_frame=True,
        adversarial_loss_weight=None,
        multiscale_adversarial_loss_weight=None,
    ):
        adversarial_loss_weight = default(adversarial_loss_weight, self.adversarial_loss_weight)
        multiscale_adversarial_loss_weight = default(
            multiscale_adversarial_loss_weight, self.multiscale_adversarial_loss_weight
        )
        assert return_loss + return_codes + return_discr_loss <= 1
        assert video_or_images.ndim in {4, 5}
        assert tuple(video_or_images.shape)[-2:] == (self.image_size, self.image_size)
        is_image = video_or_images.ndim == 4
        if is_image:
            video = rearrange(video_or_images, "b c ... -> b c 1 ...")
            video_contains_first_frame = True
        else:
            video = video_or_images
        batch, channels, frames = tuple(video.shape)[:3]

        assert divisible_by(
            frames - int(video_contains_first_frame), self.time_downsample_factor
        ), f"number of frames {frames} minus the first frame ({frames - int(video_contains_first_frame)}) must be divisible by the total downsample factor across time {self.time_downsample_factor}"

        x = self.encode(video, cond=cond, video_contains_first_frame=video_contains_first_frame)

        if self.use_fsq:
            quantized, codes = self.quantizers(x)
            aux_losses = self.zero
            quantizer_loss_breakdown = None
        else:
            (quantized, codes, aux_losses), quantizer_loss_breakdown = self.quantizers(x, return_loss_breakdown=True)
        if return_codes and not return_recon:
            return codes

        recon_video = self.decode(quantized, cond=cond, video_contains_first_frame=video_contains_first_frame)
        if return_codes:
            return codes, recon_video
        if not (return_loss or return_discr_loss or return_recon_loss_only):
            return recon_video

        recon_loss = paddle.nn.functional.mse_loss(input=video, label=recon_video)

        if return_recon_loss_only:
            return recon_loss, recon_video
        if return_discr_loss:
            assert self.has_gan
            assert exists(self.discr)
            frame_indices = paddle.randn(shape=(batch, frames)).topk(k=1, axis=-1)[1]

            real = pick_video_frame(video, frame_indices)
            if apply_gradient_penalty:
                out_11 = real
                out_11.stop_gradient = not True
                real = out_11
            fake = pick_video_frame(recon_video, frame_indices)
            real_logits = self.discr(real)
            fake_logits = self.discr(fake.detach())
            discr_loss = hinge_discr_loss(fake_logits, real_logits)
            multiscale_discr_losses = []
            if self.has_multiscale_discrs:
                for discr in self.multiscale_discrs:
                    multiscale_real_logits = discr(video)
                    multiscale_fake_logits = discr(recon_video.detach())
                    multiscale_discr_loss = hinge_discr_loss(multiscale_fake_logits, multiscale_real_logits)
                    multiscale_discr_losses.append(multiscale_discr_loss)
            else:
                multiscale_discr_losses.append(self.zero)
            if apply_gradient_penalty:
                gradient_penalty_loss = gradient_penalty(real, real_logits)
            else:
                gradient_penalty_loss = self.zero
            total_loss = (
                discr_loss
                + gradient_penalty_loss * self.grad_penalty_loss_weight
                + sum(multiscale_discr_losses) * self.multiscale_adversarial_loss_weight
            )
            discr_loss_breakdown = DiscrLossBreakdown(discr_loss, multiscale_discr_losses, gradient_penalty_loss)
            return total_loss, discr_loss_breakdown
        if self.use_vgg:
            frame_indices = paddle.randn(shape=(batch, frames)).topk(k=1, axis=-1)[1]

            input_vgg_input = pick_video_frame(video, frame_indices)
            recon_vgg_input = pick_video_frame(recon_video, frame_indices)
            if channels == 1:
                input_vgg_input = repeat(input_vgg_input, "b 1 h w -> b c h w", c=3)
                recon_vgg_input = repeat(recon_vgg_input, "b 1 h w -> b c h w", c=3)
            elif channels == 4:
                input_vgg_input = input_vgg_input[:, :3]
                recon_vgg_input = recon_vgg_input[:, :3]

            input_vgg_feats = self.vgg(input_vgg_input)
            recon_vgg_feats = self.vgg(recon_vgg_input)
            perceptual_loss = paddle.nn.functional.mse_loss(input=input_vgg_feats, label=recon_vgg_feats)

        else:
            perceptual_loss = self.zero

        last_dec_layer = self.conv_out.conv.weight
        norm_grad_wrt_perceptual_loss = None

        if self.training and self.use_vgg and (self.has_gan or self.has_multiscale_discrs):
            norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p=2)

        recon_video_frames = None

        if self.has_gan:
            frame_indices = paddle.randn(shape=(batch, frames)).topk(k=1, axis=-1)[1]

            recon_video_frames = pick_video_frame(recon_video, frame_indices)
            fake_logits = self.discr(recon_video_frames)
            gen_loss = hinge_gen_loss(fake_logits)
            adaptive_weight = 1.0

            if exists(norm_grad_wrt_perceptual_loss):
                norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p=2)
                adaptive_weight = norm_grad_wrt_perceptual_loss / norm_grad_wrt_gen_loss.clip(min=0.001)
                adaptive_weight.clip_(max=1000.0)
                if paddle.isnan(x=adaptive_weight).astype("bool").any():
                    adaptive_weight = 1.0
        else:
            gen_loss = self.zero
            adaptive_weight = 0.0

        multiscale_gen_losses = []
        multiscale_gen_adaptive_weights = []
        if self.has_multiscale_gan and self.has_multiscale_discrs:
            if not exists(recon_video_frames):
                recon_video_frames = pick_video_frame(recon_video, frame_indices)
            for discr in self.multiscale_discrs:
                fake_logits = recon_video_frames
                multiscale_gen_loss = hinge_gen_loss(fake_logits)
                multiscale_gen_losses.append(multiscale_gen_loss)
                multiscale_adaptive_weight = 1.0
                if exists(norm_grad_wrt_perceptual_loss):
                    norm_grad_wrt_gen_loss = grad_layer_wrt_loss(multiscale_gen_loss, last_dec_layer).norm(p=2)
                    multiscale_adaptive_weight = norm_grad_wrt_perceptual_loss / norm_grad_wrt_gen_loss.clip(min=1e-05)
                    multiscale_adaptive_weight.clip_(max=1000.0)
                multiscale_gen_adaptive_weights.append(multiscale_adaptive_weight)
        total_loss = (
            recon_loss
            + aux_losses * self.quantizer_aux_loss_weight
            + perceptual_loss * self.perceptual_loss_weight
            + gen_loss * adaptive_weight * adversarial_loss_weight
        )

        if self.has_multiscale_discrs:
            weighted_multiscale_gen_losses = sum(
                loss * weight for loss, weight in zip(multiscale_gen_losses, multiscale_gen_adaptive_weights)
            )
            total_loss = total_loss + weighted_multiscale_gen_losses * multiscale_adversarial_loss_weight
        loss_breakdown = LossBreakdown(
            recon_loss,
            aux_losses,
            quantizer_loss_breakdown,
            perceptual_loss,
            gen_loss,
            adaptive_weight,
            multiscale_gen_losses,
            multiscale_gen_adaptive_weights,
        )

        return total_loss, loss_breakdown
