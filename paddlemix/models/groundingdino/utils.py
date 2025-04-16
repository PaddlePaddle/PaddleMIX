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

import copy
import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


def inverse_sigmoid(x, eps=1e-3):
    x = x.clip(min=0, max=1)
    x1 = x.clip(min=eps)
    x2 = (1 - x).clip(min=eps)
    return paddle.log(x1 / x2)


def _get_clones(module, N, layer_share=False):

    if layer_share:
        return nn.LayerList([module for i in range(N)])
    else:
        return nn.LayerList([copy.deepcopy(module) for i in range(N)])


def get_sine_pos_embed(
    pos_tensor: paddle.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
):
    """generate sine position embedding from a position tensor
    Args:
        pos_tensor (paddle.Tensor): shape: [..., n].
        num_pos_feats (int): projected shape for each float in the tensor.
        temperature (int): temperature in the sine/cosine function.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is [x,y], the results will be [pos(y), pos(x)]. Defaults to True.
    Returns:
        pos_embed (torch.Tensor): shape: [..., n*num_pos_feats].
    """
    scale = 2 * math.pi
    dim_t = paddle.arange(num_pos_feats)
    dim_t = temperature ** (2.0 * paddle.floor_divide(dim_t, paddle.to_tensor(2)) / num_pos_feats)

    def sine_func(x: paddle.Tensor):
        sin_x = x * scale / dim_t
        sin_x = paddle.stack((sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), axis=3).flatten(2)
        return sin_x

    pos_res = [sine_func(x) for x in paddle.split(pos_tensor, [1] * pos_tensor.shape[-1], axis=-1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = paddle.concat(pos_res, axis=-1)
    return pos_res


def gen_encoder_output_proposals(
    memory: paddle.Tensor,
    memory_padding_mask: paddle.Tensor,
    spatial_shapes: paddle.Tensor,
    learnedwh=None,
):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
        - learnedwh: 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].reshape([N_, H_, W_, 1])
        valid_H = paddle.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = paddle.sum(~mask_flatten_[:, 0, :, 0], 1)

        # import ipdb; ipdb.set_trace()

        grid_y, grid_x = paddle.meshgrid(
            paddle.linspace(0, H_ - 1, H_, dtype=paddle.float32),
            paddle.linspace(0, W_ - 1, W_, dtype=paddle.float32),
        )
        grid = paddle.concat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # H_, W_, 2

        scale = paddle.concat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).reshape([N_, 1, 1, 2])
        grid = (grid.unsqueeze(0).tile([N_, 1, 1, 1]) + 0.5) / scale.astype(grid.dtype)

        if learnedwh is not None:
            wh = paddle.ones_like(grid) * learnedwh.sigmoid() * (2.0**lvl)
        else:
            wh = paddle.ones_like(grid) * 0.05 * (2.0**lvl)

        proposal = paddle.concat((grid, wh), -1).reshape([N_, -1, 4])
        proposals.append(proposal)
        _cur += H_ * W_

    output_proposals = paddle.concat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = paddle.log(output_proposals / (1 - output_proposals))  # unsigmoid
    output_proposals = masked_fill(output_proposals, memory_padding_mask.unsqueeze(-1), float("inf"))
    output_proposals = masked_fill(output_proposals, ~output_proposals_valid, float("inf"))

    output_memory = memory
    output_memory = masked_fill(output_memory, memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = masked_fill(output_memory, ~output_proposals_valid, float(0))

    return output_memory, output_proposals


class RandomBoxPerturber:
    def __init__(self, x_noise_scale=0.2, y_noise_scale=0.2, w_noise_scale=0.2, h_noise_scale=0.2) -> None:
        self.noise_scale = paddle.to_tensor([x_noise_scale, y_noise_scale, w_noise_scale, h_noise_scale])

    def __call__(self, refanchors: paddle.Tensor) -> paddle.Tensor:
        nq, bs, query_dim = refanchors.shape

        noise_raw = paddle.rand(shape=refanchors.shape, dtype=refanchors.dtype)
        noise_scale = self.noise_scale[:query_dim]

        new_refanchors = refanchors * (1 + (noise_raw - 0.5) * noise_scale)
        return new_refanchors.clip(0, 1)


def sigmoid_focal_loss(
    inputs,
    targets,
    num_boxes,
    alpha: float = 0.25,
    gamma: float = 2,
    no_reduction=False,
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if no_reduction:
        return loss

    return loss.mean(1).sum() / num_boxes


class MLP(nn.Layer):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation, d_model=256, batch_dim=0):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def gen_sineembed_for_position(pos_tensor):

    scale = 2 * math.pi
    dim_t = paddle.arange(128)
    dim_t = 10000 ** (2 * (paddle.floor_divide(dim_t, paddle.to_tensor(2))) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = paddle.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), axis=3).flatten(2)
    pos_y = paddle.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), axis=3).flatten(2)
    if pos_tensor.shape[-1] == 2:
        pos = paddle.concat((pos_y, pos_x), axis=2)
    elif pos_tensor.shape[-1] == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = paddle.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), axis=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = paddle.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), axis=3).flatten(2)

        pos = paddle.concat((pos_y, pos_x, pos_w, pos_h), axis=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.shape[-1]))
    return pos


class ContrastiveEmbed(nn.Layer):
    def __init__(self, max_text_len=256):
        """
        Args:
            max_text_len: max length of text.
        """
        super().__init__()
        self.max_text_len = max_text_len

    def forward(self, x, text_dict):
        """_summary_

        Args:
            x (_type_): _description_
            text_dict (_type_): _description_
            {
                'encoded_text': encoded_text, # bs, 195, d_model
                'text_token_mask': text_token_mask, # bs, 195
                        # True for used tokens. False for padding tokens
            }
        Returns:
            _type_: _description_
        """
        assert isinstance(text_dict, dict)

        y = text_dict["encoded_text"]
        text_token_mask = text_dict["text_token_mask"]

        res = x @ y.transpose([0, 2, 1])
        masked_fill(res, ~text_token_mask[:, None, :], float("-inf"))

        # padding to max_text_len
        new_res = paddle.full((*res.shape[:-1], self.max_text_len), float("-inf"))
        new_res[..., : res.shape[-1]] = res

        return new_res
