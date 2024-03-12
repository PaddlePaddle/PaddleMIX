# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import operator as op

SCALER_NAME = "scaler.pd"
MODEL_NAME = "paddle_model"
SAFE_MODEL_NAME = "model"
RNG_STATE_NAME = "random_states"
OPTIMIZER_NAME = "optimizer"
SCHEDULER_NAME = "scheduler"
SAMPLER_NAME = "sampler"
WEIGHTS_NAME = f"{MODEL_NAME}.bin"
WEIGHTS_INDEX_NAME = f"{WEIGHTS_NAME}.index.json"
SAFE_WEIGHTS_NAME = f"{SAFE_MODEL_NAME}.safetensors"
SAFE_WEIGHTS_INDEX_NAME = f"{SAFE_WEIGHTS_NAME}.index.json"

DEEPSPEED_MULTINODE_LAUNCHERS = []
TORCH_DYNAMO_MODES = ["default", "reduce-overhead", "max-autotune"]

STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}

PADDLE_LAUNCH_PARAMS = [
    "nnodes",
    "nproc_per_node",
    "m",
    "log_dir",
    "node_rank",
    "master_addr",
    "master_port",
]

CUDA_DISTRIBUTED_TYPES = ["MULTI_GPU"]
PADDLE_DISTRIBUTED_OPERATION_TYPES = CUDA_DISTRIBUTED_TYPES + [
    "MULTI_GPU",
]
