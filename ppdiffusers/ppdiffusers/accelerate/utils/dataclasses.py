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

"""
General namespace and dataclass related classes
"""

import copy
import enum
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, List, Optional

import paddle


class KwargsHandler:
    """
    Internal mixin that implements a `to_kwargs()` method for a dataclass.
    """

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def to_kwargs(self):
        """
        Returns a dictionary containing the attributes with values different from the default of this class.
        """
        # import clear_environment here to avoid circular import problem
        from .other import clear_environment

        with clear_environment():
            default_dict = self.__class__().to_dict()
        this_dict = self.to_dict()
        return {k: v for k, v in this_dict.items() if default_dict[k] != v}


@dataclass
class AutocastKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize how `torch.autocast` behaves. Please refer to the
    documentation of this [context manager](https://pytorch.org/docs/stable/amp.html#torch.autocast) for more
    information on each argument.

    Example:

    ```python
    from accelerate import Accelerator
    from accelerate.utils import AutocastKwargs

    kwargs = AutocastKwargs(cache_enabled=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    ```
    """

    enable: bool = True
    custom_white_list: Any = None
    custom_black_list: Any = None
    level: str = "O1"
    dtype: str = "float16"


@dataclass
class DistributedDataParallelKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize how your model is wrapped in a
    `torch.nn.parallel.DistributedDataParallel`. Please refer to the documentation of this
    [wrapper](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) for more
    information on each argument.

    <Tip warning={true}>

    `gradient_as_bucket_view` is only available in PyTorch 1.7.0 and later versions.

    `static_graph` is only available in PyTorch 1.11.0 and later versions.

    </Tip>

    Example:

    ```python
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    ```
    """

    dim: int = 0
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    check_reduction: bool = False
    gradient_as_bucket_view: bool = False
    static_graph: bool = False


@dataclass
class GradScalerKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize the behavior of mixed precision, specifically how the
    `torch.cuda.amp.GradScaler` used is created. Please refer to the documentation of this
    [scaler](https://pytorch.org/docs/stable/amp.html?highlight=gradscaler) for more information on each argument.

    <Tip warning={true}>

    `GradScaler` is only available in PyTorch 1.5.0 and later versions.

    </Tip>

    Example:

    ```python
    from accelerate import Accelerator
    from accelerate.utils import GradScalerKwargs

    kwargs = GradScalerKwargs(backoff_filter=0.25)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    ```
    """

    enable: bool = True
    init_loss_scaling: float = 65536.0
    incr_ratio: float = 2.0
    decr_ratio: float = 0.5
    incr_every_n_steps: int = 2000
    decr_every_n_nan_or_inf: int = 2
    use_dynamic_loss_scaling: bool = True


@dataclass
class InitProcessGroupKwargs(KwargsHandler):
    """
    Use this object in your [`Accelerator`] to customize the initialization of the distributed processes. Please refer
    to the documentation of this
    [method](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) for more
    information on each argument.

    ```python
    from datetime import timedelta
    from accelerate import Accelerator
    from accelerate.utils import InitProcessGroupKwargs

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=800))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    ```
    """

    backend: Optional[str] = "nccl"
    init_method: Optional[str] = None
    timeout: timedelta = timedelta(seconds=1800)


class EnumWithContains(enum.EnumMeta):
    "A metaclass that adds the ability to check if `self` contains an item with the `in` operator"

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class BaseEnum(enum.Enum, metaclass=EnumWithContains):
    "An enum class that can get the value of an item with `str(Enum.key)`"

    def __str__(self):
        return self.value

    @classmethod
    def list(cls):
        "Method to list all the possible items in `cls`"
        return list(map(str, cls))


class DistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment.

    Values:

        - **NO** -- Not a distributed environment, just a single process.
        - **MULTI_GPU** -- Distributed on multiple GPUs.
    """

    # Subclassing str as well as Enum allows the `DistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    MULTI_GPU = "MULTI_GPU"


class SageMakerDistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment.

    Values:

        - **NO** -- Not a distributed environment, just a single process.
        - **DATA_PARALLEL** -- using sagemaker distributed data parallelism.
        - **MODEL_PARALLEL** -- using sagemaker distributed model parallelism.
    """

    # Subclassing str as well as Enum allows the `SageMakerDistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    DATA_PARALLEL = "DATA_PARALLEL"
    MODEL_PARALLEL = "MODEL_PARALLEL"


class ComputeEnvironment(str, enum.Enum):
    """
    Represents a type of the compute environment.

    Values:

        - **LOCAL_MACHINE** -- private/custom cluster hardware.
        - **AMAZON_SAGEMAKER** -- Amazon SageMaker as compute environment.
    """

    # Subclassing str as well as Enum allows the `ComputeEnvironment` to be JSON-serializable out of the box.
    LOCAL_MACHINE = "LOCAL_MACHINE"
    AMAZON_SAGEMAKER = "AMAZON_SAGEMAKER"


class LoggerType(BaseEnum):
    """Represents a type of supported experiment tracker

    Values:

        - **ALL** -- all available trackers in the environment that are supported
        - **TENSORBOARD** -- TensorBoard as an experiment tracker
        - **WANDB** -- wandb as an experiment tracker
        - **COMETML** -- comet_ml as an experiment tracker
        - **DVCLIVE** -- dvclive as an experiment tracker
        - **VISUALDL** -- visualdl as an experiment tracker
    """

    ALL = "all"
    AIM = "aim"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"
    COMETML = "comet_ml"
    MLFLOW = "mlflow"
    CLEARML = "clearml"
    DVCLIVE = "dvclive"
    VISUALDL = "visualdl"


class PrecisionType(BaseEnum):
    """Represents a type of precision used on floating point values

    Values:

        - **NO** -- using full precision (FP32)
        - **FP16** -- using half precision
        - **BF16** -- using brain floating point precision
    """

    NO = "no"
    FP16 = "fp16"
    BF16 = "bf16"


class FP16OPTLevel(BaseEnum):
    """For fp16/bf16: AMP optimization level selected in ['O0', 'O1', and 'O2'].
    See details at https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/amp/auto_cast_cn.html

    Values:

        - **O0** -- O0
        - **O1** -- O1
        - **O2** -- O2
    """

    O0 = "O0"
    O1 = "O1"
    O2 = "O2"


# data classes


@dataclass
class TensorInformation:
    shape: List
    dtype: paddle.dtype


@dataclass
class ProjectConfiguration:
    """
    Configuration for the Accelerator object based on inner-project needs.
    """

    project_dir: str = field(default=None, metadata={"help": "A path to a directory for storing data."})
    logging_dir: str = field(
        default=None,
        metadata={
            "help": "A path to a directory for storing logs of locally-compatible loggers. If None, defaults to `project_dir`."
        },
    )
    automatic_checkpoint_naming: bool = field(
        default=False,
        metadata={"help": "Whether saved states should be automatically iteratively named."},
    )

    total_limit: int = field(
        default=None,
        metadata={"help": "The maximum number of total saved states to keep."},
    )

    iteration: int = field(
        default=0,
        metadata={"help": "The current save iteration."},
    )

    save_on_each_node: bool = field(
        default=False,
        metadata={
            "help": (
                "When doing multi-node distributed training, whether to save models and checkpoints on each node, or"
                " only on the main one"
            )
        },
    )

    def set_directories(self, project_dir: str = None):
        "Sets `self.project_dir` and `self.logging_dir` to the appropriate values."
        self.project_dir = project_dir
        if self.logging_dir is None:
            self.logging_dir = project_dir

    def __post_init__(self):
        self.set_directories(self.project_dir)


@dataclass
class GradientAccumulationPlugin(KwargsHandler):
    """
    A plugin to configure gradient accumulation behavior.
    """

    num_steps: int = field(default=None, metadata={"help": "The number of steps to accumulate gradients for."})
    adjust_scheduler: bool = field(
        default=True,
        metadata={
            "help": "Whether to adjust the scheduler steps to account for the number of steps being accumulated. Should be `True` if the used scheduler was not adjusted for gradient accumulation."
        },
    )
    sync_with_dataloader: bool = field(
        default=True,
        metadata={
            "help": "Whether to synchronize setting the gradients when at the end of the dataloader. Should only be set to `False` if you know what you're doing."
        },
    )
