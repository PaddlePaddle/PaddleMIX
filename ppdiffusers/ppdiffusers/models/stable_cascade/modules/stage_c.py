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

import numpy as np
import paddle
import paddle.nn as nn
import paddle_aux  # noqa

from .common import AttnBlock, FeedForwardBlock, LayerNorm2d, ResBlock, TimestepBlock


def load(path="../x.npy"):
    return paddle.to_tensor(np.load(path))


def diff(a, b):
    return (a - b).abs().mean()


class UpDownBlock2d(nn.Layer):
    def __init__(self, c_in, c_out, mode, enabled=True):
        super().__init__()
        assert mode in ["up", "down"]
        interpolation = (
            nn.Upsample(
                scale_factor=2 if mode == "up" else 0.5,
                mode="bilinear",
                align_corners=True,
            )
            if enabled
            else nn.Identity()
        )
        mapping = nn.Conv2D(in_channels=c_in, out_channels=c_out, kernel_size=1)
        self.blocks = nn.LayerList(sublayers=[interpolation, mapping] if mode == "up" else [mapping, interpolation])

    def forward(self, x):
        for block in self.blocks:
            x = block(x.astype(paddle.float32))
        return x


class StageC(nn.Layer):
    def __init__(
        self,
        c_in=16,
        c_out=16,
        c_r=64,
        patch_size=1,
        c_cond=2048,
        c_hidden=[2048, 2048],
        nhead=[32, 32],
        blocks=[[8, 24], [24, 8]],
        block_repeat=[[1, 1], [1, 1]],
        level_config=["CTA", "CTA"],
        c_clip_text=1280,
        c_clip_text_pooled=1280,
        c_clip_img=768,
        c_clip_seq=4,
        kernel_size=3,
        dropout=[0.1, 0.1],
        # dropout=[0, 0],
        self_attn=True,
        t_conds=["sca", "crp"],
        switch_level=[False],
    ):
        super().__init__()
        self.c_r = c_r
        self.t_conds = t_conds
        self.c_clip_seq = c_clip_seq
        if not isinstance(dropout, list):
            dropout = [dropout] * len(c_hidden)
        if not isinstance(self_attn, list):
            self_attn = [self_attn] * len(c_hidden)
        # CONDITIONING
        self.clip_txt_mapper = nn.Linear(c_clip_text, c_cond)
        self.clip_txt_pooled_mapper = nn.Linear(c_clip_text_pooled, c_cond * c_clip_seq)
        self.clip_img_mapper = nn.Linear(c_clip_img, c_cond * c_clip_seq)
        self.clip_norm = nn.LayerNorm(c_cond, weight_attr=False, bias_attr=False, epsilon=1e-6)

        self.embedding = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            nn.Conv2D(c_in * (patch_size**2), c_hidden[0], kernel_size=1),
            LayerNorm2d(c_hidden[0], weight_attr=False, bias_attr=False, epsilon=1e-6),
        )

        def get_block(block_type, c_hidden, nhead, c_skip=0, dropout=0, self_attn=True):
            if block_type == "C":
                return ResBlock(c_hidden, c_skip, kernel_size=kernel_size, dropout=dropout)
            elif block_type == "A":
                return AttnBlock(c_hidden, c_cond, nhead, self_attn=self_attn, dropout=dropout)
            elif block_type == "F":
                return FeedForwardBlock(c_hidden, dropout=dropout)
            elif block_type == "T":
                return TimestepBlock(c_hidden, c_r, conds=t_conds)
            else:
                raise Exception(f"Block type {block_type} not supported")

        self.down_blocks = nn.LayerList()
        self.down_downscalers = nn.LayerList()
        self.down_repeat_mappers = nn.LayerList()
        for i in range(len(c_hidden)):
            if i > 0:
                self.down_downscalers.append(
                    nn.Sequential(
                        LayerNorm2d(
                            c_hidden[i - 1],
                            weight_attr=False,
                            bias_attr=False,
                            epsilon=1e-06,
                        ),
                        UpDownBlock2d(
                            c_hidden[i - 1],
                            c_hidden[i],
                            mode="down",
                            enabled=switch_level[i - 1],
                        ),
                    )
                )
            else:
                self.down_downscalers.append(nn.Identity())
            down_block = nn.LayerList()
            for _ in range(blocks[0][i]):
                for block_type in level_config[i]:
                    block = get_block(
                        block_type,
                        c_hidden[i],
                        nhead[i],
                        dropout=dropout[i],
                        self_attn=self_attn[i],
                    )
                    down_block.append(block)
            self.down_blocks.append(down_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.LayerList()
                for _ in range(block_repeat[0][i] - 1):
                    block_repeat_mappers.append(nn.Conv2D(c_hidden[i], c_hidden[i], kernel_size=1))
                self.down_repeat_mappers.append(block_repeat_mappers)
        self.up_blocks = nn.LayerList()
        self.up_upscalers = nn.LayerList()
        self.up_repeat_mappers = nn.LayerList()
        for i in reversed(range(len(c_hidden))):
            if i > 0:
                self.up_upscalers.append(
                    nn.Sequential(
                        LayerNorm2d(c_hidden[i], weight_attr=False, bias_attr=False, epsilon=1e-6),
                        UpDownBlock2d(
                            c_hidden[i],
                            c_hidden[i - 1],
                            mode="up",
                            enabled=switch_level[i - 1],
                        ),
                    )
                )
            else:
                self.up_upscalers.append(nn.Identity())
            up_block = nn.LayerList()
            for j in range(blocks[1][::-1][i]):
                for k, block_type in enumerate(level_config[i]):
                    c_skip = c_hidden[i] if i < len(c_hidden) - 1 and j == k == 0 else 0
                    block = get_block(
                        block_type,
                        c_hidden[i],
                        nhead[i],
                        c_skip=c_skip,
                        dropout=dropout[i],
                        self_attn=self_attn[i],
                    )
                    up_block.append(block)
            self.up_blocks.append(up_block)
            if block_repeat is not None:
                block_repeat_mappers = nn.LayerList()
                for _ in range(block_repeat[1][::-1][i] - 1):
                    block_repeat_mappers.append(nn.Conv2D(c_hidden[i], c_hidden[i], kernel_size=1))
                self.up_repeat_mappers.append(block_repeat_mappers)
        self.clf = nn.Sequential(
            LayerNorm2d(c_hidden[0], weight_attr=False, bias_attr=False, epsilon=1e-06),
            nn.Conv2D(c_hidden[0], c_out * (patch_size**2), kernel_size=1),
            nn.PixelShuffle(upscale_factor=patch_size),
        )
        self.apply(self._init_weights)
        init_Normal = nn.initializer.Normal(std=0.02)
        init_Normal(self.clip_txt_mapper.weight)
        init_Normal = nn.initializer.Normal(std=0.02)
        init_Normal(self.clip_txt_pooled_mapper.weight)
        init_Normal = nn.initializer.Normal(std=0.02)
        init_Normal(self.clip_img_mapper.weight)
        init_Xavier = nn.initializer.XavierUniform()
        self.embedding[1].weight = self.create_parameter(
            shape=self.embedding[1].weight.shape, default_initializer=init_Xavier
        )
        init_Constant = nn.initializer.Constant(value=0)
        init_Constant(self.clf[1].weight)

        for level_list in (self.down_blocks, self.up_blocks):
            for level_block in level_list:
                for block in level_block:
                    if isinstance(block, ResBlock) or isinstance(block, FeedForwardBlock):
                        block.channelwise[-1].weight.multiply(np.sqrt(1 / sum(blocks[0])))
                    elif isinstance(block, TimestepBlock):
                        for layer in block.sublayers():
                            if isinstance(layer, nn.Linear):
                                init_Constant = nn.initializer.Constant(value=0)
                                init_Constant(layer.weight)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2D, nn.Linear)):
            init_XavierUniform = nn.initializer.XavierUniform()
            init_XavierUniform(m.weight)
            if m.bias is not None:
                init_Constant = nn.initializer.Constant(value=0)
                init_Constant(m.bias)

    def gen_r_embedding(self, r, max_positions=10000):
        r = r * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = paddle.arange(end=half_dim).astype(dtype="float32").mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = paddle.concat(x=[emb.sin(), emb.cos()], axis=1)
        if self.c_r % 2 == 1:
            emb = nn.functional.pad(emb, [0, 1], mode="constant")
        return emb

    def gen_c_embeddings(self, clip_txt, clip_txt_pooled, clip_img):
        clip_txt = self.clip_txt_mapper(clip_txt)
        if len(clip_txt_pooled.shape) == 2:
            clip_txt_pool = clip_txt_pooled.unsqueeze(axis=1)
        if len(clip_img.shape) == 2:
            clip_img = paddle.unsqueeze(clip_img, axis=1)

        clip_txt_pool = self.clip_txt_pooled_mapper(clip_txt_pooled).reshape(
            [clip_txt_pooled.shape[0], clip_txt_pooled.shape[1] * self.c_clip_seq, -1]
        )

        clip_img = self.clip_img_mapper(clip_img).reshape([clip_img.shape[0], clip_img.shape[1] * self.c_clip_seq, -1])

        clip = paddle.concat(x=[clip_txt, clip_txt_pool, clip_img], axis=1)
        clip = self.clip_norm(clip)

        return clip

    def _down_encode(self, x, r_embed, clip, cnet=None):
        level_outputs = []
        block_group = zip(self.down_blocks, self.down_downscalers, self.down_repeat_mappers)
        for down_block, downscaler, repmap in block_group:
            x = downscaler(x)
            for i in range(len(repmap) + 1):
                for block in down_block:
                    if (
                        isinstance(block, ResBlock)
                        or hasattr(block, "_fsdp_wrapped_module")
                        and isinstance(block._fsdp_wrapped_module, ResBlock)
                    ):
                        if cnet is not None:
                            next_cnet = cnet()
                            if next_cnet is not None:
                                x = x + nn.functional.interpolate(
                                    next_cnet,
                                    size=x.shape[-2:],
                                    mode="bilinear",
                                    align_corners=True,
                                )
                        x = block(x)

                    elif (
                        isinstance(block, AttnBlock)
                        or hasattr(block, "_fsdp_wrapped_module")
                        and isinstance(block._fsdp_wrapped_module, AttnBlock)
                    ):
                        x = block(x, clip)

                    elif (
                        isinstance(block, TimestepBlock)
                        or hasattr(block, "_fsdp_wrapped_module")
                        and isinstance(block._fsdp_wrapped_module, TimestepBlock)
                    ):
                        x = block(x, r_embed)
                    else:
                        x = block(x)

                if i < len(repmap):
                    x = repmap[i](x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, r_embed, clip, cnet=None):
        x = level_outputs[0]
        block_group = zip(self.up_blocks, self.up_upscalers, self.up_repeat_mappers)
        count_i = 0
        for i, (up_block, upscaler, repmap) in enumerate(block_group):
            count_i += 1
            count_j = 0
            for j in range(len(repmap) + 1):
                count_j += 1
                count_k = 0
                for k, block in enumerate(up_block):
                    count_k += 1

                    if (
                        isinstance(block, ResBlock)
                        or hasattr(block, "_fsdp_wrapped_module")
                        and isinstance(block._fsdp_wrapped_module, ResBlock)
                    ):
                        skip = level_outputs[i] if k == 0 and i > 0 else None
                        if skip is not None and (x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]):
                            x = nn.functional.interpolate(
                                x=x.astype(paddle.float32),
                                size=skip.shape[-2:],
                                mode="bilinear",
                                align_corners=True,
                            )
                        x = block(x, skip)
                    elif (
                        isinstance(block, AttnBlock)
                        or hasattr(block, "_fsdp_wrapped_module")
                        and isinstance(block._fsdp_wrapped_module, AttnBlock)
                    ):
                        x = block(x, clip)
                    elif (
                        isinstance(block, TimestepBlock)
                        or hasattr(block, "_fsdp_wrapped_module")
                        and isinstance(block._fsdp_wrapped_module, TimestepBlock)
                    ):
                        x = block(x, r_embed)
                    else:
                        x = block(x)

                if j < len(repmap):
                    x = repmap[j](x)

            x = upscaler(x)

        return x

    def forward(self, x, r, clip_text, clip_text_pooled, clip_img, cnet=None, **kwargs):

        r_embed = self.gen_r_embedding(r)
        for c in self.t_conds:
            t_cond = kwargs.get(c, paddle.zeros_like(r))
            r_embed = paddle.concat(x=[r_embed, self.gen_r_embedding(t_cond)], axis=1)
        clip = self.gen_c_embeddings(clip_text, clip_text_pooled, clip_img)

        x = self.embedding(x)
        level_outputs = self._down_encode(x, r_embed, clip, cnet)
        x = self._up_decode(level_outputs, r_embed, clip, cnet)
        x = self.clf(x)
        # x.register_hook(lambda grad: print("@@@ before-clf-x @@@", grad.shape, grad.abs().mean()))

        return x

    def update_weights_ema(self, src_model, beta=0.999):
        for self_params, src_params in zip(self.parameters(), src_model.parameters()):
            self_params.data = self_params.data * beta + src_params.data.clone() * (1 - beta)
        for self_buffers, src_buffers in zip(self.buffers(), src_model.buffers()):
            self_buffers.data = self_buffers.data * beta + src_buffers.data.clone() * (1 - beta)
