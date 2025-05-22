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

from dataclasses import dataclass, field
from typing import List

import paddle

__all__ = ["ModelArgs", "VectorQuantizer", "VQModel"]


@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256
    dropout_p: float = 0.0


class Encoder(paddle.nn.Layer):
    def __init__(
        self,
        in_channels=3,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        norm_type="group",
        dropout=0.0,
        resamp_with_conv=True,
        z_channels=256,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = paddle.nn.Conv2D(in_channels=in_channels, out_channels=ch, kernel_size=3, stride=1, padding=1)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.conv_blocks = paddle.nn.LayerList()
        for i_level in range(self.num_resolutions):
            conv_block = paddle.nn.Layer()
            res_block = paddle.nn.LayerList()
            attn_block = paddle.nn.LayerList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            if i_level != self.num_resolutions - 1:
                conv_block.downsample = Downsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)
        self.mid = paddle.nn.LayerList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = paddle.nn.Conv2D(
            in_channels=block_in, out_channels=z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        h = self.conv_in(x)
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)
        for mid_block in self.mid:
            h = mid_block(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(paddle.nn.Layer):
    def __init__(
        self,
        z_channels=256,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        norm_type="group",
        dropout=0.0,
        resamp_with_conv=True,
        out_channels=3,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = ch * ch_mult[self.num_resolutions - 1]
        self.conv_in = paddle.nn.Conv2D(
            in_channels=z_channels, out_channels=block_in, kernel_size=3, stride=1, padding=1
        )
        self.mid = paddle.nn.LayerList()
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type))
        self.conv_blocks = paddle.nn.LayerList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = paddle.nn.Layer()
            res_block = paddle.nn.LayerList()
            attn_block = paddle.nn.LayerList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(block_in, block_out, dropout=dropout, norm_type=norm_type))
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            if i_level != 0:
                conv_block.upsample = Upsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = paddle.nn.Conv2D(
            in_channels=block_in, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )

    @property
    def last_layer(self):
        return self.conv_out.weight

    def forward(self, z):
        h = self.conv_in(z)
        for mid_block in self.mid:
            h = mid_block(h)
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VectorQuantizer(paddle.nn.Layer):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.embedding = paddle.nn.Embedding(num_embeddings=self.n_e, embedding_dim=self.e_dim)
        # self.embedding.weight.data = paddle.cast(paddle.uniform(self.embedding.weight.data.shape,min=-1.0 / self.n_e, max=1.0 /self.n_e,dtype='float32'),dtype=paddle.bfloat16)
        if self.l2_norm:
            self.embedding.weight.data = paddle.nn.functional.normalize(x=self.embedding.weight.data, p=2, axis=-1)
        if self.show_usage:
            self.register_buffer(name="codebook_used", tensor=paddle.zeros(shape=[65536]))

    def forward(self, z):
        z = paddle.einsum("b c h w -> b h w c", z).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        if self.l2_norm:
            z = paddle.nn.functional.normalize(x=z, p=2, axis=-1)
            z_flattened = paddle.nn.functional.normalize(x=z_flattened, p=2, axis=-1)
            embedding = paddle.nn.functional.normalize(x=self.embedding.weight, p=2, axis=-1)
        else:
            embedding = self.embedding.weight
        d = (
            paddle.sum(x=z_flattened**2, axis=1, keepdim=True)
            + paddle.sum(x=embedding**2, axis=1)
            - 2 * paddle.einsum("bd,dn->bn", z_flattened, paddle.einsum("n d -> d n", embedding))
        )
        min_encoding_indices = paddle.argmin(x=d, axis=1)
        z_q = embedding[min_encoding_indices].view(tuple(z.shape))
        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        if self.training:
            vq_loss = paddle.mean(x=(z_q - z.detach()) ** 2)
            commit_loss = self.beta * paddle.mean(x=(z_q.detach() - z) ** 2)
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)
        z_q = z + (z_q - z).detach()
        z_q = paddle.einsum("b h w c -> b c h w", z_q)
        return z_q, (vq_loss, commit_loss, entropy_loss), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        if self.l2_norm:
            embedding = paddle.nn.functional.normalize(x=self.embedding.weight, p=2, axis=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]
        if shape is not None:
            if channel_first:
                z_q = z_q.reshape([shape[0], shape[2], shape[3], shape[1]])
                z_q = z_q.transpose(perm=[0, 3, 1, 2]).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q


class ResnetBlock(paddle.nn.Layer):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type="group"):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = paddle.nn.Dropout(p=dropout)
        self.conv2 = paddle.nn.Conv2D(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = paddle.nn.Conv2D(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = paddle.nn.Conv2D(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock(paddle.nn.Layer):
    def __init__(self, in_channels, norm_type="group"):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = paddle.nn.Conv2D(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = tuple(q.shape)
        q = q.reshape([b, c, h * w])
        q = q.transpose(perm=[0, 2, 1])
        k = k.reshape([b, c, h * w])
        w_ = paddle.bmm(x=q, y=k)
        w_ = w_ * int(c) ** -0.5
        w_ = paddle.nn.functional.softmax(x=w_, axis=2)
        v = v.reshape([b, c, h * w])
        w_ = w_.transpose(perm=[0, 2, 1])
        h_ = paddle.bmm(x=v, y=w_)
        h_ = h_.reshape([b, c, h, w])
        h_ = self.proj_out(h_)
        return x + h_


def nonlinearity(x):
    return x * paddle.nn.functional.sigmoid(x=x)


def Normalize(in_channels, norm_type="group"):
    assert norm_type in ["group", "batch"]
    if norm_type == "group":
        return paddle.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, epsilon=1e-06, weight_attr=True, bias_attr=True
        )
    elif norm_type == "batch":
        return paddle.nn.SyncBatchNorm(num_features=in_channels)


class Upsample(paddle.nn.Layer):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = paddle.nn.Conv2D(
                in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        origin_dtype = x.dtype
        if x.dtype != paddle.float32:
            x = paddle.nn.functional.interpolate(x=x.astype(paddle.float32), scale_factor=2.0, mode="nearest").astype(
                origin_dtype
            )
        else:
            x = paddle.nn.functional.interpolate(x=x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(paddle.nn.Layer):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = paddle.nn.Conv2D(
                in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = 0, 1, 0, 1
            x = paddle.nn.functional.pad(x=x, pad=pad, mode="constant", value=0, pad_from_left_axis=False)
            x = self.conv(x)
        else:
            x = paddle.nn.functional.avg_pool2d(kernel_size=2, stride=2, x=x, exclusive=False)
        return x


def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    flat_affinity = affinity.reshape([-1, tuple(affinity.shape)[-1]])
    flat_affinity /= temperature
    probs = paddle.nn.functional.softmax(x=flat_affinity, axis=-1)
    log_probs = paddle.nn.functional.log_softmax(x=flat_affinity + 1e-05, axis=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = paddle.mean(x=target_probs, axis=0)
    avg_entropy = -paddle.sum(x=avg_probs * paddle.log(x=avg_probs + 1e-05))
    sample_entropy = -paddle.mean(x=paddle.sum(x=target_probs * log_probs, axis=-1))
    loss = sample_entropy - avg_entropy
    return loss


class VQModel(paddle.nn.Layer):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(ch_mult=config.encoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, z_channels=config.z_channels, dropout=config.dropout_p)
        self.quantize = VectorQuantizer(
            config.codebook_size,
            config.codebook_embed_dim,
            config.commit_loss_beta,
            config.entropy_loss_ratio,
            config.codebook_l2_norm,
            config.codebook_show_usage,
        )
        self.quant_conv = paddle.nn.Conv2D(
            in_channels=config.z_channels, out_channels=config.codebook_embed_dim, kernel_size=1
        )
        self.post_quant_conv = paddle.nn.Conv2D(
            in_channels=config.codebook_embed_dim, out_channels=config.z_channels, kernel_size=1
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff
