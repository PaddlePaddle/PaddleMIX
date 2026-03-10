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
        if group is None:
            world_size = dist.get_world_size()
        else:
            world_size = group.world_size
        tensors_gather = [paddle.empty_like(x=tensor) for _ in range(world_size)]
        paddle.distributed.all_gather(tensors_gather, tensor, group=group)  #
        ctx.group = group
        ctx.batch_size = tensor.shape[0]
        return paddle.concat(x=tensors_gather, axis=0)

    @staticmethod
    def backward(ctx, grad_output):
        num_or_sections = grad_output.shape[0] // ctx.batch_size
        grad_lst = paddle.split(grad_output, num_or_sections=num_or_sections)
        grad = paddle.zeros_like(x=grad_lst[0])
        dist.reduce_scatter(grad, grad_lst, group=ctx.group)
        return grad


allgather = AllGather.apply
