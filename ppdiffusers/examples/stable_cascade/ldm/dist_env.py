# Copyright (c) 2024  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import random

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker


def set_hyrbid_parallel_seed(basic_seed, data_world_rank, mp_rank, pp_rank=0):
    device_id = paddle.device.get_device()
    assert "gpu" in device_id

    random.seed(basic_seed + data_world_rank)
    np.random.seed(basic_seed + data_world_rank)
    paddle.seed(basic_seed + data_world_rank)

    # local_seed/ global_seed is used to control dropout in ModelParallel
    local_seed = 1024 + basic_seed + mp_rank * 100 + data_world_rank
    global_seed = 2048 + basic_seed + data_world_rank
    tracker = get_rng_state_tracker()
    tracker.add("global_seed", global_seed)
    tracker.add("local_seed", local_seed)


def setdistenv(args):
    world_size = dist.get_world_size()
    if world_size > 1:
        args.dp_degree = max(args.dp_degree, 1)
        args.sharding_parallel_degree = max(args.sharding_parallel_degree, 1)
        args.tensor_parallel_degree = max(args.tensor_parallel_degree, 1)
        args.sep_parallel_degree = max(args.sep_parallel_degree, 1)
        args.pipeline_parallel_degree = max(args.pipeline_parallel_degree, 1)

        assert (
            world_size % (args.tensor_parallel_degree * args.pipeline_parallel_degree) == 0
        ), f"Total world_size:{world_size} should be devided by tensor_parallel_degree: {args.tensor_parallel_degree} and pipeline_parallel_degree: {args.pipeline_parallel_degree}."

        args.dp_degree = world_size // (
            args.tensor_parallel_degree * args.sharding_parallel_degree * args.pipeline_parallel_degree
        )
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": args.dp_degree,
            "mp_degree": args.tensor_parallel_degree,
            "sharding_degree": args.sharding_parallel_degree,
            "pp_degree": args.pipeline_parallel_degree,
        }
        # strategy.find_unused_parameters = True

        # set control in tensor parallel
        strategy.tensor_parallel_configs = {"tensor_init_seed": args.seed}

        fleet.init(is_collective=True, strategy=strategy)

        args.rank = dist.get_rank()
        # obtain rank message of hybrid parallel
        hcg = fleet.get_hybrid_communicate_group()
        args.mp_rank = hcg.get_model_parallel_rank()
        args.dp_rank = hcg.get_data_parallel_rank()
        args.sharding_rank = hcg.get_sharding_parallel_rank()

        args.data_world_rank = args.dp_rank * args.sharding_parallel_degree + args.sharding_rank
        args.data_world_size = world_size // abs(args.tensor_parallel_degree * args.pipeline_parallel_degree)
    else:
        args.data_world_rank = 0
        args.data_world_size = 1
        args.mp_rank = 0
        args.rank = 0

    # seed control in hybrid parallel
    set_hyrbid_parallel_seed(args.seed, args.data_world_rank, args.mp_rank)
