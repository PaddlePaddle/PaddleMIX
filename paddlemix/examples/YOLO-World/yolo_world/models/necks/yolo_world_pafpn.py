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
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from ppdet.modeling import ShapeSpec
from ppdet.modeling.backbones.csp_darknet import DWConv
from ppdet.modeling.backbones.yolov8_csp_darknet import C2fLayer
from yolo_world.models.utils.util import BaseConv, make_round


class MaxSigmoidAttnBlock(nn.Layer):
    """Max Sigmoid attention block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        guide_channels,
        embed_channels,
        kernel_size=3,
        padding=1,
        num_heads=1,
        depthwise=False,
        with_scale=False,
        act="silu",
    ):
        super().__init__()

        assert (
            out_channels % num_heads == 0 and embed_channels % num_heads == 0
        ), "out_channels and embed_channels should be divisible by num_heads."
        self.num_heads = num_heads
        self.head_channels = out_channels // num_heads
        Conv = DWConv if depthwise else BaseConv

        self.embed_conv = (
            nn.Sequential(
                nn.Conv2D(
                    in_channels=in_channels,
                    out_channels=embed_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                ),
                nn.BatchNorm2D(num_features=embed_channels, momentum=0.03, epsilon=0.001),
            )
            if embed_channels != in_channels
            else None
        )

        self.guide_fc = nn.Linear(guide_channels, embed_channels)

        self.bias = self.create_parameter(
            shape=[num_heads],
            default_initializer=nn.initializer.Constant(value=0.0),
            is_bias=True,
        )

        if with_scale:
            self.scale = self.create_parameter(
                shape=[1, num_heads, 1, 1],
                default_initializer=nn.initializer.Constant(value=1.0),
            )
        else:
            self.scale = 1.0

        self.project_conv = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            ksize=kernel_size,
            stride=1,
            with_act=False,
        )

    def forward(self, x, guide):
        B, _, H, W = x.shape

        guide = self.guide_fc(guide)
        guide = guide.reshape([B, -1, self.num_heads, self.head_channels])
        embed = self.embed_conv(x) if self.embed_conv is not None else x
        embed = embed.reshape([B, self.num_heads, self.head_channels, H, W])

        batch, num_heads, head_channels, height, width = embed.shape
        _, num_embeddings, _, _ = guide.shape
        embed = paddle.transpose(embed, perm=[0, 1, 3, 4, 2])
        embed = embed.reshape([B, num_heads, -1, head_channels])
        guide = paddle.transpose(guide, perm=[0, 2, 3, 1])
        attn_weight = paddle.matmul(embed, guide)
        attn_weight = attn_weight.reshape([batch, num_heads, height, width, num_embeddings])

        attn_weight = attn_weight.max(axis=-1)[0]
        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = attn_weight + self.bias[None, :, None, None]
        attn_weight = attn_weight.sigmoid() * self.scale

        x = self.project_conv(x)
        x = x.reshape([B, self.num_heads, -1, H, W])
        x = x * attn_weight.unsqueeze(2)
        x = x.reshape([B, -1, H, W])
        return x


class MaxSigmoidCSPLayerWithTwoConv(C2fLayer):
    """Sigmoid-attention based CSP layer with two convolution layers."""

    def __init__(
        self,
        in_channels,
        out_channels,
        guide_channels,
        embed_channels,
        num_heads=1,
        expansion=0.5,
        num_blocks=1,
        depthwise=False,
        with_scale=False,
        shortcut=True,
        act="silu",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            shortcut=shortcut,
            expansion=expansion,
            depthwise=depthwise,
            act=act,
        )

        self.conv1 = BaseConv(in_channels=in_channels, out_channels=2 * self.c, ksize=1, stride=1, act=act)

        self.conv2 = BaseConv(
            in_channels=(3 + num_blocks) * self.c,
            out_channels=out_channels,
            ksize=1,
            stride=1,
            act=act,
        )

        self.attn_block = MaxSigmoidAttnBlock(
            self.c,
            self.c,
            guide_channels=guide_channels,
            embed_channels=embed_channels,
            num_heads=num_heads,
            depthwise=depthwise,
            with_scale=with_scale,
            act=act,
        )

    def forward(self, x, guide):
        """Forward process."""
        x_main = self.conv1(x)
        x_main = list(x_main.split((self.c, self.c), 1))
        x_main.extend(blocks(x_main[-1]) for blocks in self.bottlenecks)
        x_main.append(self.attn_block(x_main[-1], guide))
        return self.conv2(paddle.concat(x_main, 1))


class ImagePoolingAttentionModule(nn.Layer):
    def __init__(
        self,
        image_channels,
        text_channels,
        embed_channels,
        with_scale=False,
        num_feats=3,
        num_heads=8,
        pool_size=3,
    ):
        super().__init__()

        self.text_channels = text_channels
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.num_feats = num_feats
        self.head_channels = embed_channels // num_heads
        self.pool_size = pool_size
        if with_scale:
            self.scale = self.create_parameter(shape=[1], default_initializer=paddle.nn.initializer.Constant(0.0))
        else:
            self.scale = 1.0

        self.projections = nn.LayerList(
            [
                BaseConv(
                    in_channels=in_channels,
                    out_channels=embed_channels,
                    ksize=1,
                    stride=1,
                    with_act=False,
                )
                for in_channels in image_channels
            ]
        )
        self.query = nn.Sequential(nn.LayerNorm(text_channels), nn.Linear(text_channels, embed_channels))
        self.key = nn.Sequential(nn.LayerNorm(embed_channels), nn.Linear(embed_channels, embed_channels))
        self.value = nn.Sequential(nn.LayerNorm(embed_channels), nn.Linear(embed_channels, embed_channels))
        self.proj = nn.Linear(embed_channels, text_channels)

        self.image_pools = nn.LayerList([nn.AdaptiveMaxPool2D((pool_size, pool_size)) for _ in range(num_feats)])

    def forward(self, text_features, image_features):
        B = image_features[0].shape[0]
        assert len(image_features) == self.num_feats
        num_patches = self.pool_size**2
        mlvl_image_features = [
            pool(proj(x)).view(B, -1, num_patches)
            for (x, proj, pool) in zip(image_features, self.projections, self.image_pools)
        ]
        mlvl_image_features = paddle.transpose(paddle.concat(mlvl_image_features, axis=-1), perm=[0, 2, 1])
        q = self.query(text_features)
        k = self.key(mlvl_image_features)
        v = self.value(mlvl_image_features)

        q = q.reshape([B, -1, self.num_heads, self.head_channels])
        k = k.reshape([B, -1, self.num_heads, self.head_channels])
        v = v.reshape([B, -1, self.num_heads, self.head_channels])

        q = paddle.transpose(q, perm=[0, 2, 1, 3])
        k = paddle.transpose(k, perm=[0, 2, 1, 3])
        attn_weight = paddle.matmul(q, k)

        attn_weight = attn_weight / (self.head_channels**0.5)
        attn_weight = F.softmax(attn_weight, axis=-1)

        v = paddle.transpose(v, perm=[0, 2, 1, 3])
        x = paddle.matmul(attn_weight, v)
        x = paddle.transpose(x, perm=[0, 2, 1, 3])
        x = self.proj(x.reshape([B, -1, self.embed_channels]))
        return x * self.scale + text_features


@register
class YOLOWorldPAFPN(nn.Layer):
    """
    Path Aggregation Network used in YOLO World
    Following YOLOv8 PAFPN, including text to image fusion
    """

    __shared__ = ["depth_mult", "width_mult"]

    def __init__(
        self,
        in_channels,
        guide_channels,
        embed_channels,
        num_heads,
        depth_mult=1.0,
        width_mult=1.0,
        num_csp_blocks=3,
        act="silu",
    ):
        super(YOLOWorldPAFPN, self).__init__()
        self.in_channels = in_channels
        self._out_channels = in_channels

        # top-down
        self.top_down_layers_0 = MaxSigmoidCSPLayerWithTwoConv(
            in_channels=(in_channels[1] + in_channels[2]),
            out_channels=self._out_channels[1],
            guide_channels=guide_channels,
            embed_channels=make_round(embed_channels[1], width_mult),
            num_heads=make_round(num_heads[1], width_mult),
            num_blocks=make_round(num_csp_blocks, depth_mult),
            shortcut=False,
        )

        self.top_down_layers_1 = MaxSigmoidCSPLayerWithTwoConv(
            in_channels=(in_channels[0] + in_channels[1]),
            out_channels=self._out_channels[0],
            guide_channels=guide_channels,
            embed_channels=make_round(embed_channels[0], width_mult),
            num_heads=make_round(num_heads[0], width_mult),
            num_blocks=make_round(num_csp_blocks, depth_mult),
            shortcut=False,
        )

        # bottom-up
        self.downsample_layers_0 = BaseConv(in_channels[0], in_channels[0], 3, stride=2, act=act)

        self.bottom_up_layers_0 = MaxSigmoidCSPLayerWithTwoConv(
            in_channels=self._out_channels[0] + in_channels[1],
            out_channels=self._out_channels[1],
            guide_channels=guide_channels,
            embed_channels=make_round(embed_channels[1], width_mult),
            num_heads=make_round(num_heads[1], width_mult),
            num_blocks=make_round(num_csp_blocks, depth_mult),
            shortcut=False,
        )

        self.downsample_layers_1 = BaseConv(in_channels[1], in_channels[1], 3, stride=2, act=act)

        self.bottom_up_layers_1 = MaxSigmoidCSPLayerWithTwoConv(
            in_channels=self._out_channels[1] + in_channels[2],
            out_channels=self._out_channels[2],
            guide_channels=guide_channels,
            embed_channels=make_round(embed_channels[2], width_mult),
            num_heads=make_round(num_heads[2], width_mult),
            num_blocks=make_round(num_csp_blocks, depth_mult),
            shortcut=False,
        )

    def forward(self, img_feats, txt_feats):

        assert len(img_feats) == len(self.in_channels)

        [c3, c4, c5] = img_feats

        # top-down FPN
        up_feat1 = F.interpolate(c5, scale_factor=2.0, mode="nearest")
        f_concat1 = paddle.concat([up_feat1, c4], 1)
        f_out0 = self.top_down_layers_0(f_concat1, txt_feats)

        up_feat2 = F.interpolate(f_out0, scale_factor=2.0, mode="nearest")
        f_concat2 = paddle.concat([up_feat2, c3], 1)
        f_out1 = self.top_down_layers_1(f_concat2, txt_feats)

        # bottom-up PAN
        down_feat1 = self.downsample_layers_0(f_out1)
        p_concat1 = paddle.concat([down_feat1, f_out0], 1)
        pan_out1 = self.bottom_up_layers_0(p_concat1, txt_feats)

        down_feat2 = self.downsample_layers_1(pan_out1)
        p_concat2 = paddle.concat([down_feat2, c5], 1)
        pan_out0 = self.bottom_up_layers_1(p_concat2, txt_feats)

        return [f_out1, pan_out1, pan_out0]

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "in_channels": [i.channels for i in input_shape],
        }

    @property
    def out_shape(self):
        return [ShapeSpec(channels=c) for c in self._out_channels]
