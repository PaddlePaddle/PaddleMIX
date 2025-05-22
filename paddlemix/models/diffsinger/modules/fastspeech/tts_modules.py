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
import sys

import paddle

from paddlemix.models.diffsinger.utils import paddle_aux
from paddlemix.models.diffsinger.modules.commons.common_layers import (
    EncSALayer,
    SinusoidalPositionalEmbedding,
)
from paddlemix.models.diffsinger.modules.commons.espnet_positional_embedding import (
    RelPositionalEncoding,
)

DEFAULT_MAX_SOURCE_POSITIONS = 2000
DEFAULT_MAX_TARGET_POSITIONS = 2000


class TransformerEncoderLayer(paddle.nn.Layer):
    def __init__(self, hidden_size, dropout, kernel_size=None, act="gelu", num_heads=2):
        super().__init__()
        self.op = EncSALayer(
            hidden_size,
            num_heads,
            dropout=dropout,
            attention_dropout=0.0,
            relu_dropout=dropout,
            kernel_size=kernel_size,
            act=act,
        )

    def forward(self, x, **kwargs):
        return self.op(x, **kwargs)


class LayerNorm(paddle.nn.LayerNorm):
    """Layer normalization module.
    :param int nout: output dim size
    :param int dim: dimension to be normalized
    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.
        :param torch.Tensor x: input tensor
        :return: layer normalized tensor
        :rtype torch.Tensor
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return (
            super(LayerNorm, self)
            .forward(x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 1, -1)))
            .transpose(
                perm=paddle_aux.transpose_aux_func(
                    super(LayerNorm, self)
                    .forward(x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 1, -1)))
                    .ndim,
                    1,
                    -1,
                )
            )
        )


class DurationPredictor(paddle.nn.Layer):
    """Duration predictor module.
    This is a module of duration predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`, those are calculated in linear domain.
    """

    def __init__(
        self, in_dims, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0, dur_loss_type="mse"
    ):
        """Initialize duration predictor module.
        Args:
            in_dims (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = paddle.nn.LayerList()
        self.kernel_size = kernel_size
        for idx in range(n_layers):
            in_chans = in_dims if idx == 0 else n_chans
            self.conv.append(
                paddle.nn.Sequential(
                    paddle.nn.Identity(),
                    paddle.nn.Conv1D(
                        in_channels=in_chans,
                        out_channels=n_chans,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    ),
                    paddle.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    paddle.nn.Dropout(p=dropout_rate),
                )
            )
        self.loss_type = dur_loss_type
        if self.loss_type in ["mse", "huber"]:
            self.out_dims = 1
        else:
            raise NotImplementedError()
        self.linear = paddle.nn.Linear(in_features=n_chans, out_features=self.out_dims)

    def out2dur(self, xs):
        if self.loss_type in ["mse", "huber"]:
            dur = xs.squeeze(axis=-1).exp() - self.offset
        else:
            raise NotImplementedError()
        return dur

    def forward(self, xs, x_masks=None, infer=True):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (BoolTensor, optional): Batch of masks indicating padded part (B, Tmax).
            infer (bool): Whether inference
        Returns:
            (train) FloatTensor, (infer) LongTensor: Batch of predicted durations in linear domain (B, Tmax).
        """
        xs = xs.transpose(perm=paddle_aux.transpose_aux_func(xs.ndim, 1, -1))
        masks = 1 - x_masks.astype(dtype="float32")
        masks_ = masks[:, None, :]
        for f in self.conv:
            xs = f(xs)
            if x_masks is not None:
                xs = xs * masks_
        xs = self.linear(xs.transpose(perm=paddle_aux.transpose_aux_func(xs.ndim, 1, -1)))
        xs = xs * masks[:, :, None]
        dur_pred = self.out2dur(xs)
        if infer:
            dur_pred = dur_pred.clip(min=0.0)
        return dur_pred


class VariancePredictor(paddle.nn.Layer):
    def __init__(self, vmin, vmax, in_dims, n_layers=5, n_chans=512, kernel_size=5, dropout_rate=0.1):
        """Initialize variance predictor module.
        Args:
            in_dims (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(VariancePredictor, self).__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.conv = paddle.nn.LayerList()
        self.kernel_size = kernel_size
        for idx in range(n_layers):
            in_chans = in_dims if idx == 0 else n_chans
            self.conv.append(
                paddle.nn.Sequential(
                    paddle.nn.Conv1D(
                        in_channels=in_chans,
                        out_channels=n_chans,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    ),
                    paddle.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    paddle.nn.Dropout(p=dropout_rate),
                )
            )
        self.linear = paddle.nn.Linear(in_features=n_chans, out_features=1)
        self.embed_positions = SinusoidalPositionalEmbedding(in_dims, 0, init_size=4096)
        self.pos_embed_alpha = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.to_tensor(data=[1], dtype="float32")
        )

    def out2value(self, xs):
        return (xs + 1) / 2 * (self.vmax - self.vmin) + self.vmin

    def forward(self, xs, infer=True):
        """
        :param xs: [B, T, H]
        :param infer: whether inference
        :return: [B, T]
        """
        positions = self.pos_embed_alpha * self.embed_positions(xs[..., 0])
        xs = xs + positions
        xs = xs.transpose(perm=paddle_aux.transpose_aux_func(xs.ndim, 1, -1))
        for f in self.conv:
            xs = f(xs)
        xs = self.linear(xs.transpose(perm=paddle_aux.transpose_aux_func(xs.ndim, 1, -1))).squeeze(axis=-1)
        if infer:
            xs = self.out2value(xs)
        return xs


class PitchPredictor(paddle.nn.Layer):
    def __init__(
        self, vmin, vmax, num_bins, deviation, in_dims, n_layers=5, n_chans=384, kernel_size=5, dropout_rate=0.1
    ):
        """Initialize pitch predictor module.
        Args:
            in_dims (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super(PitchPredictor, self).__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.interval = (vmax - vmin) / (num_bins - 1)
        self.sigma = deviation / self.interval
        self.register_buffer(name="x", tensor=paddle.arange(end=num_bins).astype(dtype="float32").reshape(1, 1, -1))
        self.base_pitch_embed = paddle.nn.Linear(in_features=1, out_features=in_dims)
        self.conv = paddle.nn.LayerList()
        self.kernel_size = kernel_size
        for idx in range(n_layers):
            in_chans = in_dims if idx == 0 else n_chans
            self.conv.append(
                paddle.nn.Sequential(
                    paddle.nn.Conv1D(
                        in_channels=in_chans,
                        out_channels=n_chans,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    ),
                    paddle.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    paddle.nn.Dropout(p=dropout_rate),
                )
            )
        self.linear = paddle.nn.Linear(in_features=n_chans, out_features=num_bins)
        self.embed_positions = SinusoidalPositionalEmbedding(in_dims, 0, init_size=4096)
        self.pos_embed_alpha = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.to_tensor(data=[1], dtype="float32")
        )

    def bins_to_values(self, bins):
        return bins * self.interval + self.vmin

    def out2pitch(self, probs):
        logits = probs.sigmoid()
        bins = paddle.sum(x=self.x * logits, axis=2) / paddle.sum(x=logits, axis=2)
        pitch = self.bins_to_values(bins)
        return pitch

    def forward(self, xs, base):
        """
        :param xs: [B, T, H]
        :param base: [B, T]
        :return: [B, T, N]
        """
        xs = xs + self.base_pitch_embed(base[..., None])
        positions = self.pos_embed_alpha * self.embed_positions(xs[..., 0])
        xs = xs + positions
        xs = xs.transpose(perm=paddle_aux.transpose_aux_func(xs.ndim, 1, -1))
        for f in self.conv:
            xs = f(xs)
        xs = self.linear(xs.transpose(perm=paddle_aux.transpose_aux_func(xs.ndim, 1, -1)))
        return self.out2pitch(xs) + base, xs


class RhythmRegulator(paddle.nn.Layer):
    def __init__(self, eps=1e-05):
        super().__init__()
        self.eps = eps

    def forward(self, ph_dur, ph2word, word_dur):
        """
        Example (no batch dim version):
            1. ph_dur = [4,2,3,2]
            2. word_dur = [3,4,2], ph2word = [1,2,2,3]
            3. word_dur_in = [4,5,2]
            4. alpha_w = [0.75,0.8,1], alpha_ph = [0.75,0.8,0.8,1]
            5. ph_dur_out = [3,1.6,2.4,2]
        :param ph_dur: [B, T_ph]
        :param ph2word: [B, T_ph]
        :param word_dur: [B, T_w]
        """
        ph_dur = ph_dur.astype(dtype="float32") * (ph2word > 0)
        word_dur = word_dur.astype(dtype="float32")
        word_dur_in = paddle.zeros(
            shape=[tuple(ph_dur.shape)[0], ph2word.max() + 1], dtype=ph_dur.dtype
        ).put_along_axis(axis=1, indices=ph2word, values=ph_dur, reduce="add")[:, 1:]
        alpha_w = word_dur / word_dur_in.clip(min=self.eps)
        alpha_ph = paddle.take_along_axis(
            arr=paddle.nn.functional.pad(x=alpha_w, pad=[1, 0], pad_from_left_axis=False),
            axis=1,
            indices=ph2word,
            broadcast=False,
        )
        ph_dur_out = ph_dur * alpha_ph
        return ph_dur_out.round().astype(dtype="int64")


class LengthRegulator(paddle.nn.Layer):
    def forward(self, dur, dur_padding=None, alpha=None):
        """
        Example (no batch dim version):
            1. dur = [2,2,3]
            2. token_idx = [[1],[2],[3]], dur_cumsum = [2,4,7], dur_cumsum_prev = [0,2,4]
            3. token_mask = [[1,1,0,0,0,0,0],
                             [0,0,1,1,0,0,0],
                             [0,0,0,0,1,1,1]]
            4. token_idx * token_mask = [[1,1,0,0,0,0,0],
                                         [0,0,2,2,0,0,0],
                                         [0,0,0,0,3,3,3]]
            5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]

        :param dur: Batch of durations of each frame (B, T_txt)
        :param dur_padding: Batch of padding of each frame (B, T_txt)
        :param alpha: duration rescale coefficient
        :return:
            mel2ph (B, T_speech)
        """
        assert alpha is None or alpha > 0
        if alpha is not None:
            dur = paddle.round(dur.astype(dtype="float32") * alpha).astype(dtype="int64")
        if dur_padding is not None:
            dur = dur * (1 - dur_padding.astype(dtype="int64"))
        token_idx = paddle.arange(start=1, end=tuple(dur.shape)[1] + 1)[None, :, None].to(dur.place)
        dur_cumsum = paddle.cumsum(x=dur, axis=1)
        # dur_cumsum_prev = paddle.nn.functional.pad(x=dur_cumsum, pad=[1, -1
        #     ], mode='constant', value=0, pad_from_left_axis=False)
        dur_cumsum_prev = paddle.concat([paddle.zeros_like(dur_cumsum[:, :1]), dur_cumsum[:, :-1]], axis=1)

        pos_idx = paddle.arange(end=dur.sum(axis=-1).max())[None, None].to(dur.place)
        token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
        mel2ph = (token_idx * token_mask.astype(dtype="int64")).sum(axis=1)
        return mel2ph


class StretchRegulator(paddle.nn.Layer):
    def forward(self, mel2ph, dur=None):
        """
        Example (no batch dim version):
            1. dur = [2,4,3]
            2. mel2ph = [1,1,2,2,2,2,3,3,3]
            3. mel2dur = [2,2,4,4,4,4,3,3,3]
            4. bound_mask = [0,1,0,0,0,1,0,0,1]
            5. 1 - bound_mask * mel2dur = [1,-1,1,1,1,-3,1,1,-2] => pad => [0,1,-1,1,1,1,-3,1,1]
            6. stretch_denorm = [0,1,0,1,2,3,0,1,2]

        :param dur: Batch of durations of each frame (B, T_txt)
        :param mel2ph: Batch of mel2ph (B, T_speech)
        :return:
            stretch (B, T_speech)
        """
        if dur is None:
            dur = mel2ph_to_dur(mel2ph, mel2ph.max())
        dur = paddle.nn.functional.pad(x=dur, pad=[1, 0], value=1, pad_from_left_axis=False)
        mel2dur = paddle.take_along_axis(arr=dur, axis=1, indices=mel2ph, broadcast=False)
        bound_mask = paddle.greater_than(x=mel2ph[:, 1:], y=paddle.to_tensor(mel2ph[:, :-1]))
        bound_mask = paddle.nn.functional.pad(
            x=bound_mask, pad=[0, 1], mode="constant", value=True, pad_from_left_axis=False
        )
        stretch_delta = 1 - bound_mask * mel2dur
        stretch_delta = paddle.nn.functional.pad(
            x=stretch_delta, pad=[1, -1], mode="constant", value=0, pad_from_left_axis=False
        )
        stretch_denorm = paddle.cumsum(x=stretch_delta, axis=1)
        stretch = stretch_denorm / mel2dur
        return stretch * (mel2ph > 0)


def mel2ph_to_dur(mel2ph, T_txt, max_dur=None):
    B, _ = tuple(mel2ph.shape)
    dur = paddle.zeros(shape=[B, T_txt + 1], dtype=mel2ph.dtype).put_along_axis(
        axis=1, indices=mel2ph, values=paddle.ones_like(x=mel2ph), reduce="add"
    )
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clip(max=max_dur)
    return dur


class FastSpeech2Encoder(paddle.nn.Layer):
    def __init__(
        self,
        hidden_size,
        num_layers,
        ffn_kernel_size=9,
        ffn_act="gelu",
        dropout=None,
        num_heads=2,
        use_pos_embed=True,
        rel_pos=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        embed_dim = self.hidden_size = hidden_size
        self.dropout = dropout
        self.use_pos_embed = use_pos_embed
        self.layers = paddle.nn.LayerList(
            sublayers=[
                TransformerEncoderLayer(
                    self.hidden_size, self.dropout, kernel_size=ffn_kernel_size, act=ffn_act, num_heads=num_heads
                )
                for _ in range(self.num_layers)
            ]
        )
        self.layer_norm = paddle.nn.LayerNorm(normalized_shape=embed_dim)
        self.embed_scale = math.sqrt(hidden_size)
        self.padding_idx = 0
        self.rel_pos = rel_pos
        if self.rel_pos:
            self.embed_positions = RelPositionalEncoding(hidden_size, dropout_rate=0.0)
        else:
            self.embed_positions = SinusoidalPositionalEmbedding(
                hidden_size, self.padding_idx, init_size=DEFAULT_MAX_TARGET_POSITIONS
            )

    def forward_embedding(self, main_embed, extra_embed=None, padding_mask=None):
        x = self.embed_scale * main_embed
        if extra_embed is not None:
            x = x + extra_embed
        if self.use_pos_embed:
            if self.rel_pos:
                x = self.embed_positions(x)
            else:
                positions = self.embed_positions(~padding_mask)
                x = x + positions
        x = paddle.nn.functional.dropout(x=x, p=self.dropout, training=self.training)
        return x

    def forward(self, main_embed, extra_embed, padding_mask, attn_mask=None, return_hiddens=False):
        x = self.forward_embedding(main_embed, extra_embed, padding_mask=padding_mask)
        nonpadding_mask_TB = (
            1
            - padding_mask.transpose(perm=paddle_aux.transpose_aux_func(padding_mask.ndim, 0, 1)).astype(
                dtype="float32"
            )[:, :, None]
        )
        x = x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 0, 1)) * nonpadding_mask_TB
        hiddens = []
        for layer in self.layers:
            x = layer(x, encoder_padding_mask=padding_mask, attn_mask=attn_mask) * nonpadding_mask_TB
            hiddens.append(x)
        x = self.layer_norm(x) * nonpadding_mask_TB
        if return_hiddens:
            x = paddle.stack(x=hiddens, axis=0)
            x = x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 1, 2))
        else:
            x = x.transpose(perm=paddle_aux.transpose_aux_func(x.ndim, 0, 1))
        return x
