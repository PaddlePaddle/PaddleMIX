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

from .models import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    AutoencoderKLTemporalDecoder,
    AutoencoderTiny,
    ConsistencyDecoderVAE,
    ControlNetModel,
    Kandinsky3UNet,
    ModelMixin,
    MotionAdapter,
    MultiAdapter,
    PriorTransformer,
    T2IAdapter,
    T5FilmDecoder,
    Transformer2DModel,
    UNet1DModel,
    UNet2DConditionModel,
    UNet2DModel,
    UNet3DConditionModel,
    UNetMotionModel,
    UNetSpatioTemporalConditionModel,
    VQModel,
)
from .optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_scheduler,
)
from .patches import *
from .pipelines import StableDiffusionPipeline, StableDiffusionXLPipeline
from .schedulers import (
    CMStochasticIterativeScheduler,
    DDIMInverseScheduler,
    DDIMParallelScheduler,
    DDIMScheduler,
    DDPMParallelScheduler,
    DDPMScheduler,
    DDPMWuerstchenScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepInverseScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    IPNDMScheduler,
    KarrasVeScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LCMScheduler,
    PNDMScheduler,
    RePaintScheduler,
    SchedulerMixin,
    ScoreSdeVeScheduler,
    UnCLIPScheduler,
    UniPCMultistepScheduler,
    VQDiffusionScheduler,
)
from .version import VERSION as __version__
