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
"""
Various positional encodings for the transformer.
"""
import math

import paddle
import paddle.nn as nn


class PositionEmbeddingSineHW(nn.Layer):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        num_pos_feats=64,
        temperatureH=10000,
        temperatureW=10000,
        normalize=False,
        scale=None,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask: paddle.Tensor):

        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.astype(paddle.float32).cumsum(1)
        x_embed = not_mask.astype(paddle.float32).cumsum(2)

        # import ipdb; ipdb.set_trace()

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_tx = paddle.arange(self.num_pos_feats)
        dim_tx = self.temperatureW ** (2 * (paddle.floor_divide(dim_tx, paddle.to_tensor(2))) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_tx

        dim_ty = paddle.arange(self.num_pos_feats)
        dim_ty = self.temperatureH ** (2 * (paddle.floor_divide(dim_ty, paddle.to_tensor(2))) / self.num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = paddle.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), axis=4).flatten(3)
        pos_y = paddle.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), axis=4).flatten(3)
        pos = paddle.concat((pos_y, pos_x), axis=3).transpose([0, 3, 1, 2])

        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSineHW(
            N_steps,
            temperatureH=args.pe_temperatureH,
            temperatureW=args.pe_temperatureW,
            normalize=True,
        )
    # elif args.position_embedding in ("v3", "learned"):
    # position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
