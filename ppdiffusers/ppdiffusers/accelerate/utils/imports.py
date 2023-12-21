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

import importlib
import importlib.metadata
import os
import warnings

import paddle
from packaging import version

from .environment import parse_flag_from_env
from .versions import compare_versions

# Cache this result has it's a C FFI call which can be pretty time-consuming
_paddle_distributed_available = paddle.distributed.is_available()


def _is_package_available(pkg_name):
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    if package_exists:
        try:
            _ = importlib.metadata.metadata(pkg_name)
            return True
        except importlib.metadata.PackageNotFoundError:
            return False


def is_paddle_distributed_available() -> bool:
    return _paddle_distributed_available


def is_cuda_available():
    """
    Checks if `cuda` is available via an `nvml-based` check which won't trigger the drivers and leave cuda
    uninitialized.
    """
    try:
        os.environ["PADDLE_NVML_BASED_CUDA_CHECK"] = str(1)
        available = paddle.device.is_compiled_with_cuda()
    finally:
        os.environ.pop("PADDLE_NVML_BASED_CUDA_CHECK", None)
    return available


def is_bf16_available(ignore_tpu=False):
    "Checks if bf16 is supported, optionally ignoring the TPU"
    if paddle.device.is_compiled_with_cuda():
        prop = paddle.device.cuda.get_device_properties()
        cc = prop.major * 10 + prop.minor
        if cc >= 80:
            return True
        else:
            return False
    return False


def is_transformers_available():
    return _is_package_available("transformers")


def is_paddlenlp_available():
    return _is_package_available("paddlenlp")


def is_datasets_available():
    return _is_package_available("datasets")


def is_aim_available():
    package_exists = _is_package_available("aim")
    if package_exists:
        aim_version = version.parse(importlib.metadata.version("aim"))
        return compare_versions(aim_version, "<", "4.0.0")
    return False


def is_tensorboard_available():
    return _is_package_available("tensorboard") or _is_package_available("tensorboardX")


def is_wandb_available():
    return _is_package_available("wandb")


def is_visualdl_available():
    return _is_package_available("visualdl")


def is_comet_ml_available():
    return _is_package_available("comet_ml")


def is_boto3_available():
    return _is_package_available("boto3")


def is_rich_available():
    if _is_package_available("rich"):
        if "ACCELERATE_DISABLE_RICH" in os.environ:
            warnings.warn(
                "`ACCELERATE_DISABLE_RICH` is deprecated and will be removed in v0.22.0 and deactivated by default. Please use `ACCELERATE_ENABLE_RICH` if you wish to use `rich`."
            )
            return not parse_flag_from_env("ACCELERATE_DISABLE_RICH", False)
        return parse_flag_from_env("ACCELERATE_ENABLE_RICH", False)
    return False


def is_tqdm_available():
    return _is_package_available("tqdm")


def is_clearml_available():
    return _is_package_available("clearml")


def is_pandas_available():
    return _is_package_available("pandas")


def is_mlflow_available():
    if _is_package_available("mlflow"):
        return True

    if importlib.util.find_spec("mlflow") is not None:
        try:
            _ = importlib.metadata.metadata("mlflow-skinny")
            return True
        except importlib.metadata.PackageNotFoundError:
            return False
    return False


def is_dvclive_available():
    return _is_package_available("dvclive")
