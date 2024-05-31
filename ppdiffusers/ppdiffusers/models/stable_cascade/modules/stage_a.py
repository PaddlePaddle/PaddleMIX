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

import paddle
from torchtools.nn import VectorQuantize


class ResBlock(paddle.nn.Layer):
    def __init__(self, c, c_hidden):
        super().__init__()
        self.norm1 = paddle.nn.LayerNorm(normalized_shape=c, weight_attr=False, bias_attr=False, epsilon=1e-06)
        self.depthwise = paddle.nn.Sequential(
            paddle.nn.Pad2D(padding=1, mode="replicate"),
            paddle.nn.Conv2D(in_channels=c, out_channels=c, kernel_size=3, groups=c),
        )
        self.norm2 = paddle.nn.LayerNorm(normalized_shape=c, weight_attr=False, bias_attr=False, epsilon=1e-06)
        self.channelwise = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=c, out_features=c_hidden),
            paddle.nn.GELU(),
            paddle.nn.Linear(in_features=c_hidden, out_features=c),
        )
        out_19 = paddle.create_parameter(
            shape=paddle.zeros(shape=[6]).shape,
            dtype=paddle.zeros(shape=[6]).numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(paddle.zeros(shape=[6])),
        )
        out_19.stop_gradient = not True
        self.gammas = out_19

        def _basic_init(module):
            if isinstance(module, paddle.nn.Linear) or isinstance(module, paddle.nn.Conv2D):
                init_XavierUniform = paddle.nn.initializer.XavierUniform()
                init_XavierUniform(module.weight)
                if module.bias is not None:
                    init_Constant = paddle.nn.initializer.Constant(value=0)
                    init_Constant(module.bias)

        self.apply(_basic_init)

    def _norm(self, x, norm):
        return norm(x.transpose(perm=[0, 2, 3, 1])).transpose(perm=[0, 3, 1, 2])

    def forward(self, x):
        mods = self.gammas
        x_temp = self._norm(x, self.norm1) * (1 + mods[0]) + mods[1]
        x = x + self.depthwise(x_temp) * mods[2]
        x_temp = self._norm(x, self.norm2) * (1 + mods[3]) + mods[4]
        x = x + self.channelwise(x_temp.transpose(perm=[0, 2, 3, 1])).transpose(perm=[0, 3, 1, 2]) * mods[5]
        return x


class StageA(paddle.nn.Layer):
    def __init__(
        self,
        levels=2,
        bottleneck_blocks=12,
        c_hidden=384,
        c_latent=4,
        codebook_size=8192,
        scale_factor=0.43,
    ):
        super().__init__()
        self.c_latent = c_latent
        self.scale_factor = scale_factor
        c_levels = [(c_hidden // 2**i) for i in reversed(range(levels))]
        self.in_block = paddle.nn.Sequential(
            paddle.nn.PixelUnshuffle(downscale_factor=2),
            paddle.nn.Conv2D(in_channels=3 * 4, out_channels=c_levels[0], kernel_size=1),
        )
        down_blocks = []
        for i in range(levels):
            if i > 0:
                down_blocks.append(
                    paddle.nn.Conv2D(
                        in_channels=c_levels[i - 1],
                        out_channels=c_levels[i],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                )
            block = ResBlock(c_levels[i], c_levels[i] * 4)
            down_blocks.append(block)
        down_blocks.append(
            paddle.nn.Sequential(
                paddle.nn.Conv2D(
                    in_channels=c_levels[-1],
                    out_channels=c_latent,
                    kernel_size=1,
                    bias_attr=False,
                ),
                paddle.nn.BatchNorm2D(num_features=c_latent),
            )
        )
        self.down_blocks = paddle.nn.Sequential(*down_blocks)
        self.down_blocks[0]
        self.codebook_size = codebook_size
        self.vquantizer = VectorQuantize(c_latent, k=codebook_size)
        up_blocks = [
            paddle.nn.Sequential(paddle.nn.Conv2D(in_channels=c_latent, out_channels=c_levels[-1], kernel_size=1))
        ]
        for i in range(levels):
            for j in range(bottleneck_blocks if i == 0 else 1):
                block = ResBlock(c_levels[levels - 1 - i], c_levels[levels - 1 - i] * 4)
                up_blocks.append(block)
            if i < levels - 1:
                up_blocks.append(
                    paddle.nn.Conv2DTranspose(
                        in_channels=c_levels[levels - 1 - i],
                        out_channels=c_levels[levels - 2 - i],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                )
        self.up_blocks = paddle.nn.Sequential(*up_blocks)
        self.out_block = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels=c_levels[0], out_channels=3 * 4, kernel_size=1),
            paddle.nn.PixelShuffle(upscale_factor=2),
        )

    def encode(self, x, quantize=False):
        x = self.in_block(x)
        x = self.down_blocks(x)
        if quantize:
            qe, (vq_loss, commit_loss), indices = self.vquantizer.forward(x, dim=1)
            return (
                qe / self.scale_factor,
                x / self.scale_factor,
                indices,
                vq_loss + commit_loss * 0.25,
            )
        else:
            return x / self.scale_factor, None, None, None

    def decode(self, x):
        x = x * self.scale_factor
        x = self.up_blocks(x)
        x = self.out_block(x)
        return x

    def forward(self, x, quantize=False):
        qe, x, _, vq_loss = self.encode(x, quantize)
        x = self.decode(qe)
        return x, vq_loss


class Discriminator(paddle.nn.Layer):
    def __init__(self, c_in=3, c_cond=0, c_hidden=512, depth=6):
        super().__init__()
        d = max(depth - 3, 3)
        layers = [
            paddle.nn.utils.spectral_norm(
                layer=paddle.nn.Conv2D(
                    in_channels=c_in,
                    out_channels=c_hidden // 2**d,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
            ),
            paddle.nn.LeakyReLU(negative_slope=0.2),
        ]
        for i in range(depth - 1):
            c_in = c_hidden // 2 ** max(d - i, 0)
            c_out = c_hidden // 2 ** max(d - 1 - i, 0)
            layers.append(
                paddle.nn.utils.spectral_norm(
                    layer=paddle.nn.Conv2D(
                        in_channels=c_in,
                        out_channels=c_out,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                )
            )
            layers.append(paddle.nn.InstanceNorm2D(num_features=c_out, momentum=1 - 0.1))
            layers.append(paddle.nn.LeakyReLU(negative_slope=0.2))
        self.encoder = paddle.nn.Sequential(*layers)
        self.shuffle = paddle.nn.Conv2D(
            in_channels=c_hidden + c_cond if c_cond > 0 else c_hidden,
            out_channels=1,
            kernel_size=1,
        )
        self.logits = paddle.nn.Sigmoid()

    def forward(self, x, cond=None):
        x = self.encoder(x)
        if cond is not None:
            cond = cond.reshape([cond.shape[0], cond.shape[1], 1, 1]).expand(shape=[-1, -1, x.shape[-2], x.shape[-1]])
            x = paddle.concat(x=[x, cond], axis=1)
        x = self.shuffle(x)
        x = self.logits(x)
        return x
