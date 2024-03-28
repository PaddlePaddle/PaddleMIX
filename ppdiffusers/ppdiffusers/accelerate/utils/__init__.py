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

from .constants import (
    MODEL_NAME,
    OPTIMIZER_NAME,
    PADDLE_DISTRIBUTED_OPERATION_TYPES,
    PADDLE_LAUNCH_PARAMS,
    RNG_STATE_NAME,
    SAFE_MODEL_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    SAMPLER_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)
from .dataclasses import (
    AutocastKwargs,
    ComputeEnvironment,
    DistributedDataParallelKwargs,
    DistributedType,
    FP16OPTLevel,
    GradientAccumulationPlugin,
    GradScalerKwargs,
    InitProcessGroupKwargs,
    KwargsHandler,
    LoggerType,
    PrecisionType,
    ProjectConfiguration,
    SageMakerDistributedType,
    TensorInformation,
)
from .environment import (
    are_libraries_initialized,
    get_int_from_env,
    parse_choice_from_env,
    parse_flag_from_env,
    str_to_bool,
)
from .imports import (
    is_aim_available,
    is_bf16_available,
    is_boto3_available,
    is_clearml_available,
    is_comet_ml_available,
    is_cuda_available,
    is_datasets_available,
    is_dvclive_available,
    is_mlflow_available,
    is_pandas_available,
    is_rich_available,
    is_tensorboard_available,
    is_transformers_available,
    is_visualdl_available,
    is_wandb_available,
)
from .launch import (
    PrepareForLaunch,
    _filter_args,
    prepare_multi_gpu_env,
    prepare_simple_launcher_cmd_env,
)
from .memory import find_executable_batch_size, release_memory
from .modeling import (
    get_mixed_precision_context_manager,
    named_module_tensors,
    shard_checkpoint,
)
from .offload import (
    OffloadedWeightsLoader,
    PrefixedDataset,
    extract_submodules_state_dict,
    load_offloaded_weight,
    offload_state_dict,
    offload_weight,
    save_offload_index,
)
from .operations import (
    CannotPadNestedTensorWarning,
    broadcast,
    broadcast_object_list,
    concatenate,
    convert_outputs_to_fp32,
    convert_to_fp32,
    find_batch_size,
    find_device,
    gather,
    gather_object,
    get_data_structure,
    honor_type,
    initialize_tensors,
    is_namedtuple,
    is_paddle_tensor,
    is_tensor_information,
    listify,
    pad_across_processes,
    recursively_apply,
    reduce,
    send_to_device,
    slice_tensors,
)
from .other import (
    check_os_kernel,
    clear_environment,
    convert_bytes,
    extract_model_from_parallel,
    get_pretty_name,
    is_port_in_use,
    merge_dicts,
    patch_environment,
    save,
    wait_for_everyone,
)
from .random import set_seed
from .tqdm import tqdm
from .versions import compare_versions, is_paddle_version
