# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The Salesforce Team Authors and The HuggingFace Team. All rights reserved.
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


import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


@paddle.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: paddle.distributed.all_gather has no gradient.
    """
    if paddle.distributed.get_world_size() < 2:
        return tensor

    tensors_gather = []
    paddle.distributed.all_gather(tensors_gather, tensor, sync_op=True)

    output = paddle.concat(tensors_gather, axis=0)
    return output


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = paddle.to_tensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]), dtype="int64"
    )
    return paddle.index_select(x, dim, order_index)


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = paddle.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)
    return paddle.concat(tensor_all, axis=0)


class CrossEntropyLoss(nn.Layer):
    """
    Softmax Cross entropy loss
    """

    def __init__(self, reduction="mean", label_smoothing=None):
        super().__init__()
        if label_smoothing is not None:
            assert label_smoothing >= 0 and label_smoothing <= 1, "label_smoothing must be in [0, 1]"
        self.epsilon = label_smoothing
        self.reduction = reduction

    def _labelsmoothing(self, target, class_num):
        if len(target.shape) == 1 or target.shape[-1] != class_num:
            one_hot_target = F.one_hot(target, class_num)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self.epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, class_num])
        return soft_target

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]
        if self.epsilon is not None:
            class_num = x.shape[-1]
            label = self._labelsmoothing(label, class_num)

            x = -F.log_softmax(x, axis=-1)
            loss = paddle.sum(x * label, axis=-1)
        else:
            if label.shape[-1] == x.shape[-1]:
                loss = paddle.sum(-label * F.log_softmax(x, axis=-1), axis=-1)
            else:
                if label.dtype == paddle.int32:
                    label = paddle.cast(label, "int64")
                loss = F.cross_entropy(x, label=label, soft_label=False)

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        else:
            return loss


class GatherLayer(paddle.autograd.PyLayer):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as paddle.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [paddle.zeros_like(x) for _ in range(paddle.distributed.get_world_size())]
        paddle.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = paddle.stack(grads)
        paddle.distributed.all_reduce(all_gradients)
        return all_gradients[paddle.distributed.get_rank()]


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)
