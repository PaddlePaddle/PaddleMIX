import sys
import paddle
""" Sin-cos, fourier, rotary position embedding modules and functions

Hacked together by / Copyright 2022 Ross Wightman
"""
import math
from typing import List, Tuple, Optional, Union


def pixel_freq_bands(num_bands: int,
                     max_freq: float=224.0,
                     linear_bands: bool=True,
                     dtype: paddle.dtype='float32',
                     device=None):
    if linear_bands:
        bands = paddle.linspace(
            start=1.0, stop=max_freq / 2, num=num_bands).astype(dtype)
    else:
        bands = 2**paddle.linspace(
            start=0, stop=math.log(max_freq, 2) - 1,
            num=num_bands).astype(dtype)
    return bands * paddle.to_tensor(math.pi).astype(dtype)


def freq_bands(num_bands: int,
               temperature: float=10000.0,
               step: int=2,
               dtype: paddle.dtype='float32',
               device=None) -> paddle.Tensor:
    bands = 1.0 / temperature**(paddle.arange(
        start=0, end=num_bands, step=step).astype(dtype) / num_bands)
    return bands


def build_sincos2d_pos_embed(feat_shape: List[int],
                             dim: int=64,
                             temperature: float=10000.0,
                             reverse_coord: bool=False,
                             interleave_sin_cos: bool=False,
                             dtype: paddle.dtype='float32',
                             device=None) -> paddle.Tensor:
    """

    Args:
        feat_shape:
        dim:
        temperature:
        reverse_coord: stack grid order W, H instead of H, W
        interleave_sin_cos: sin, cos, sin, cos stack instead of sin, sin, cos, cos
        dtype:
        device:

    Returns:

    """
    assert dim % 4 == 0, 'Embed dimension must be divisible by 4 for sin-cos 2D position embedding'
    pos_dim = dim // 4
    bands = freq_bands(
        pos_dim, temperature=temperature, step=1, dtype=dtype, device=device)
    if reverse_coord:
        feat_shape = feat_shape[::-1]
    x = paddle.stack(x=paddle.meshgrid(
        [paddle.arange(end=s).astype(dtype)
         for s in feat_shape])).flatten(start_axis=1)
    perm_4 = list(range(x.ndim))
    perm_4[0] = 1
    perm_4[1] = 0
    grid = x.transpose(perm=perm_4)
    pos2 = grid.unsqueeze(axis=-1) * bands.unsqueeze(axis=0)
    stack_dim = 2 if interleave_sin_cos else 1
    pos_emb = paddle.stack(
        x=[paddle.sin(x=pos2), paddle.cos(x=pos2)],
        axis=stack_dim).flatten(start_axis=1)
    return pos_emb


def build_fourier_pos_embed(feat_shape: List[int],
                            bands: Optional[paddle.Tensor]=None,
                            num_bands: int=64,
                            max_res: int=224,
                            temperature: float=10000.0,
                            linear_bands: bool=False,
                            include_grid: bool=False,
                            in_pixels: bool=True,
                            ref_feat_shape: Optional[List[int]]=None,
                            dtype: paddle.dtype='float32',
                            device=None) -> List[paddle.Tensor]:
    """

    Args:
        feat_shape: Feature shape for embedding.
        bands: Pre-calculated frequency bands.
        num_bands: Number of frequency bands (determines output dim).
        max_res: Maximum resolution for pixel based freq.
        temperature: Temperature for non-pixel freq.
        linear_bands: Linear band spacing for pixel based freq.
        include_grid: Include the spatial grid in output.
        in_pixels: Output in pixel freq.
        ref_feat_shape: Reference feature shape for resize / fine-tune.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    if bands is None:
        if in_pixels:
            bands = pixel_freq_bands(
                num_bands,
                float(max_res),
                linear_bands=linear_bands,
                dtype=dtype,
                device=device)
        else:
            bands = freq_bands(
                num_bands,
                temperature=temperature,
                step=1,
                dtype=dtype,
                device=device)
    else:
        if device is None:
            device = bands.place
        if dtype is None:
            dtype = bands.dtype
    if in_pixels:
        t = [
            paddle.linspace(
                start=-1.0, stop=1.0, num=s).astype(dtype) for s in feat_shape
        ]
    else:
        t = [paddle.arange(end=s).astype(dtype) for s in feat_shape]
    if ref_feat_shape is not None:
        t = [(x / f * r) for x, f, r in zip(t, feat_shape, ref_feat_shape)]
    grid = paddle.stack(x=paddle.meshgrid(t), axis=-1)
    grid = grid.unsqueeze(axis=-1)
    pos = grid * bands
    pos_sin, pos_cos = pos.sin(), pos.cos()
    out = [grid, pos_sin, pos_cos] if include_grid else [pos_sin, pos_cos]
    return out


class FourierEmbed(paddle.nn.Layer):
    def __init__(self,
                 max_res: int=224,
                 num_bands: int=64,
                 concat_grid=True,
                 keep_spatial=False):
        super().__init__()
        self.max_res = max_res
        self.num_bands = num_bands
        self.concat_grid = concat_grid
        self.keep_spatial = keep_spatial
        self.register_buffer(
            'bands', pixel_freq_bands(max_res, num_bands), persistable=False)

    def forward(self, x):
        B, C = x.shape[:2]
        feat_shape = x.shape[2:]
        emb = build_fourier_pos_embed(
            feat_shape,
            self.bands,
            include_grid=self.concat_grid,
            dtype=x.dtype,
            device=x.place)
        emb = paddle.concat(x=emb, axis=-1)
        x = emb
        perm_5 = list(range(x.ndim))
        perm_5[-1] = -2
        perm_5[-2] = -1
        emb = x.transpose(perm=perm_5).flatten(start_axis=len(feat_shape))
        batch_expand = (B, ) + (-1, ) * (x.ndim - 1)
        if self.keep_spatial:
            x = paddle.concat(
                x=[
                    x, emb.unsqueeze(axis=0).expand(
                        shape=batch_expand).transpose(perm=[0, 3, 1, 2])
                ],
                axis=1)
        else:
            x = paddle.concat(
                x=[
                    x.transpose(perm=[0, 2, 3, 1]),
                    emb.unsqueeze(axis=0).expand(shape=batch_expand)
                ],
                axis=-1)
            x = x.reshape((B, feat_shape.size(), -1))
        return x


def rot(x):
    return paddle.stack(
        x=[-x[(...), 1::2], x[(...), ::2]], axis=-1).reshape(x.shape)


def apply_rot_embed(x: paddle.Tensor, sin_emb, cos_emb):
    if sin_emb.ndim == 3:
        return x * cos_emb.unsqueeze(axis=1).expand_as(
            y=x) + rot(x) * sin_emb.unsqueeze(axis=1).expand_as(y=x)
    return x * cos_emb + rot(x) * sin_emb


def apply_rot_embed_list(x: List[paddle.Tensor], sin_emb, cos_emb):
    if isinstance(x, paddle.Tensor):
        x = [x]
    return [(t * cos_emb + rot(t) * sin_emb) for t in x]


def apply_rot_embed_cat(x: paddle.Tensor, emb):
    sin_emb, cos_emb = emb.split(2, -1)
    if sin_emb.ndim == 3:
        return x * cos_emb.unsqueeze(axis=1).expand_as(
            y=x) + rot(x) * sin_emb.unsqueeze(axis=1).expand_as(y=x)
    return x * cos_emb + rot(x) * sin_emb


def apply_keep_indices_nlc(x, pos_embed, keep_indices):
    pos_embed = pos_embed.unsqueeze(axis=0).expand(shape=[x.shape[0], -1, -1])
    pos_embed = pos_embed.take_along_axis(
        axis=1,
        indices=keep_indices.unsqueeze(axis=-1).expand(
            shape=[-1, -1, pos_embed.shape[-1]]))
    return pos_embed


def build_rotary_pos_embed(feat_shape: List[int],
                           bands: Optional[paddle.Tensor]=None,
                           dim: int=64,
                           max_res: int=224,
                           temperature: float=10000.0,
                           linear_bands: bool=False,
                           in_pixels: bool=True,
                           ref_feat_shape: Optional[List[int]]=None,
                           dtype: paddle.dtype='float32',
                           device=None):
    """

    Args:
        feat_shape: Spatial shape of the target tensor for embedding.
        bands: Optional pre-generated frequency bands
        dim: Output dimension of embedding tensor.
        max_res: Maximum resolution for pixel mode.
        temperature: Temperature (inv freq) for non-pixel mode
        linear_bands: Linearly (instead of log) spaced bands for pixel mode
        in_pixels: Pixel vs language (inv freq) mode.
        dtype: Output dtype.
        device: Output device.

    Returns:

    """
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape,
        bands=bands,
        num_bands=dim // 4,
        max_res=max_res,
        temperature=temperature,
        linear_bands=linear_bands,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        device=device,
        dtype=dtype)
    num_spatial_dim = 1
    for x in feat_shape:
        num_spatial_dim *= x
    sin_emb = sin_emb.reshape((num_spatial_dim, -1)).repeat_interleave(
        repeats=2, axis=-1)
    cos_emb = cos_emb.reshape((num_spatial_dim, -1)).repeat_interleave(
        repeats=2, axis=-1)
    return sin_emb, cos_emb


class RotaryEmbedding(paddle.nn.Layer):
    """ Rotary position embedding

    NOTE: This is my initial attempt at impl rotary embedding for spatial use, it has not
    been well tested, and will likely change. It will be moved to its own file.

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(self,
                 dim,
                 max_res=224,
                 temperature=10000,
                 in_pixels=True,
                 linear_bands: bool=False,
                 feat_shape: Optional[List[int]]=None,
                 ref_feat_shape: Optional[List[int]]=None):
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape
        if feat_shape is None:
            if in_pixels:
                bands = pixel_freq_bands(
                    dim // 4, float(max_res), linear_bands=linear_bands)
            else:
                bands = freq_bands(dim // 4, temperature=temperature, step=1)
                print(bands)
            self.register_buffer('bands', bands, persistable=False)
            self.pos_embed_sin = None
            self.pos_embed_cos = None
        else:
            emb_sin, emb_cos = build_rotary_pos_embed(
                feat_shape=feat_shape,
                dim=dim,
                max_res=max_res,
                linear_bands=linear_bands,
                in_pixels=in_pixels,
                ref_feat_shape=self.ref_feat_shape)
            self.bands = None
            self.register_buffer('pos_embed_sin', emb_sin, persistable=False)
            self.register_buffer('pos_embed_cos', emb_cos, persistable=False)

    def get_embed(self, shape: Optional[List[int]]=None):
        if self.bands is not None:
            assert shape is not None
            return build_rotary_pos_embed(
                shape, self.bands, in_pixels=self.in_pixels)
        else:
            return self.pos_embed_sin, self.pos_embed_cos

    def forward(self, x):
        sin_emb, cos_emb = self.get_embed(x.shape[2:])
        return apply_rot_embed(x, sin_emb, cos_emb)


class RotaryEmbeddingCat(paddle.nn.Layer):
    """ Rotary position embedding w/ concatenatd sin & cos

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(self,
                 dim,
                 max_res=224,
                 temperature=10000,
                 in_pixels=True,
                 linear_bands: bool=False,
                 feat_shape: Optional[List[int]]=None,
                 ref_feat_shape: Optional[List[int]]=None):
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape
        if feat_shape is None:
            if in_pixels:
                bands = pixel_freq_bands(
                    dim // 4, float(max_res), linear_bands=linear_bands)
            else:
                bands = freq_bands(dim // 4, temperature=temperature, step=1)
                print(bands)
            self.register_buffer('bands', bands, persistable=False)
            self.embed = None
        else:
            embeds = build_rotary_pos_embed(
                feat_shape=feat_shape,
                dim=dim,
                max_res=max_res,
                linear_bands=linear_bands,
                in_pixels=in_pixels,
                ref_feat_shape=self.ref_feat_shape)
            self.bands = None
            self.register_buffer(
                'pos_embed',
                paddle.concat(
                    x=embeds, axis=-1),
                persistable=False)

    def get_embed(self, shape: Optional[List[int]]=None):
        if self.bands is not None:
            assert (shape is not None, 'valid shape needed')
            embeds = build_rotary_pos_embed(
                shape, self.bands, in_pixels=self.in_pixels)
            return paddle.concat(x=embeds, axis=-1)
        else:
            return self.pos_embed

    def forward(self, x):
        pos_embed = self.get_embed(x.shape[2:])
        return apply_rot_embed_cat(x, pos_embed)
