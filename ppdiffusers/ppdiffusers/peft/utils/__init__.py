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

# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all

# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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

from .other import (
    COMMON_LAYERS_PATTERN,
    CONFIG_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    TORCH_WEIGHTS_NAME,
    TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    WEIGHTS_NAME,
    ModulesToSaveWrapper,
    _freeze_adapter,
    _get_batch_size,
    _get_submodules,
    _is_valid_match,
    _prepare_prompt_learning_config,
    _set_adapter,
    _set_trainable,
    get_auto_gptq_quant_linear,
    get_quantization_config,
    infer_device,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    shift_tokens_right,
    transpose,
)

# from .config import PeftConfig, PeftType, PromptLearningConfig, TaskType
from .peft_types import PeftType, TaskType
from .save_and_load import (
    get_peft_model_state_dict,
    load_peft_weights,
    set_peft_model_state_dict,
)
