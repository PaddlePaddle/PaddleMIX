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

import paddle
import paddle.distributed as dist


class AllGather(paddle.autograd.PyLayer):
    """An autograd function that performs allgather on a tensor.
    Performs all_gather operation on the provided tensors.
    *** Warning ***: paddle.distributed.all_gather has no gradient.
    """

    @staticmethod
    def forward(ctx, tensor, group=None):
        if group is not None:
            rank = group.rank
            world_size = group.world_size
        else:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        tensors_gather = [paddle.empty_like(x=tensor) for _ in range(world_size)]
        paddle.distributed.all_gather(tensors_gather, tensor, group=group)
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return paddle.concat(x=tensors_gather, axis=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)]


allgather = AllGather.apply
