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

import collections
from functools import reduce

import paddle
from paddle.distributed import fleet
from paddlenlp.trainer.integrations import TrainerCallback


class EmaCallback(TrainerCallback):
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        self.decay = decay
        self.num_updates = 0 if use_num_upates else -1

        # Initialize communication for distributed training
        if paddle.distributed.get_world_size() > 1 and hasattr(fleet.fleet, "_hcg"):
            hcg = fleet.get_hybrid_communicate_group()
            self._sharding_world_size = max(1, hcg.get_sharding_parallel_world_size())
            self._sharding_rank = max(0, hcg.get_sharding_parallel_rank())     
        else:
            self._sharding_world_size = 1
            self._sharding_rank = 0
        # Mapping parameters to their names and preparing for sharding
        param_2_name = {id(p): name for name, p in model.named_parameters() if not p.stop_gradient}

        # Partition parameters according to their ranks
        rank_2_param = self._partition_parameters([p for p in model.parameters() if not p.stop_gradient])

        # Map parameter names to their respective ranks
        param_2_rank = {param_2_name[id(param)]: rank for rank, params in rank_2_param.items() for param in params}

        # Only handle parameters that are assigned to the current rank
        shard_parameters = rank_2_param[self._sharding_rank]
        self.states = collections.OrderedDict(
            [
                (
                    "data",
                    collections.OrderedDict(
                        (param_2_name[id(param)], param.clone().detach().astype("float32"))
                        for param in shard_parameters
                    ),
                ),
                (
                    "meta",
                    collections.OrderedDict(
                        [("num_updates", self.num_updates), ("decay", self.decay), ("param_2_rank", param_2_rank)]
                    ),
                ),
            ]
        )

    def do_ema(self, model):
        """do ema"""
        decay = self.decay
        if self.num_updates >= 0:
            self.num_updates += 1
            # Improved decay calculation with a clearer form
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        # Access to the EMA state directly
        ema_data = self.states["data"]
        # Using no_grad context to prevent tracking history in autograd
        with paddle.no_grad():
            # Create a dictionary of the model's parameters for quick lookup
            model_params = {name: param for name, param in model.named_parameters() if not param.stop_gradient}

            # Update EMA values for each parameter
            for name, ema_param in ema_data.items():
                if name in model_params:  # Ensure the parameter exists in the model
                    origin_param = model_params[name]
                    # In-place updates for EMA parameters
                    ema_param.scale_(decay)
                    ema_param.add_(one_minus_decay * origin_param.astype("float32"))

    def _partition_parameters(self, parameters):
        """
        Partitions parameters among sharding ranks.

        Return:
        Dict[int, List]
        """
        mapping = {}
        for rank_ in range(self._sharding_world_size):
            mapping[rank_] = []
        sizes = [0] * self._sharding_world_size
        parameters.sort(key=lambda p: reduce(lambda x, y: x * y, p.shape), reverse=True)
        for param in parameters:
            rank = sizes.index(min(sizes))
            mapping[rank].append(param)
            numel = reduce(lambda x, y: x * y, param.shape)
            assert numel > 0, "param [{}] should larger than 0, but it is [{}]".format(param.name, numel)
            sizes[rank] += numel
        return mapping

    def on_step_end(self, args, state, control, **kwargs):
        """on_step_end"""
        model = kwargs["model"]
        self.do_ema(model)

    def state_dict(self):
        """state_dict"""
        # NOTE(shenliang03): need save/update self.num_updates to recovery
        self.states["meta"]["num_updates"] = self.num_updates
        self.states["meta"]["decay"] = self.decay
        return self.states

    def set_state_dict(self, state_dict):
        """set_state_dict"""
        data = self.states["data"]
        # meta = self.states["meta"]

        for name, param in state_dict["data"].items():
            data[name].set_value(param)

        self.num_updates = state_dict["meta"]["num_updates"]
        self.decay = state_dict["meta"]["decay"]
