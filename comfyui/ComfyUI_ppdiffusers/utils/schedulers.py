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

from ppdiffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    FlowMatchEulerDiscreteScheduler,
)


def get_scheduler(scheduler_type):
    scheduler_type = scheduler_type.lower()
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    original_scheduler_config = scheduler.config
    if scheduler_type == "pndm":
        scheduler = PNDMScheduler.from_config(original_scheduler_config, skip_prk_steps=True)
    elif scheduler_type == "lms":
        scheduler = LMSDiscreteScheduler.from_config(original_scheduler_config)
    elif scheduler_type == "heun":
        scheduler = HeunDiscreteScheduler.from_config(original_scheduler_config)
    elif scheduler_type == "euler":
        scheduler = EulerDiscreteScheduler.from_config(original_scheduler_config)
    elif scheduler_type == "euler-ancestral":
        scheduler = EulerAncestralDiscreteScheduler.from_config(original_scheduler_config)
    elif scheduler_type == "dpm-multi":
        scheduler = DPMSolverMultistepScheduler.from_config(original_scheduler_config)
    elif scheduler_type == "dpm-single":
        scheduler = DPMSolverSinglestepScheduler.from_config(original_scheduler_config)
    elif scheduler_type == "kdpm2-ancestral":
        scheduler = KDPM2AncestralDiscreteScheduler.from_config(original_scheduler_config)
    elif scheduler_type == "kdpm2":
        scheduler = KDPM2DiscreteScheduler.from_config(original_scheduler_config)
    elif scheduler_type == "unipc-multi":
        scheduler = UniPCMultistepScheduler.from_config(original_scheduler_config)
    elif scheduler_type == "flowmatch-euler":
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(original_scheduler_config)
    elif scheduler_type == "ddim":
        scheduler = DDIMScheduler.from_config(
            original_scheduler_config,
            steps_offset=1,
            clip_sample=False,
            set_alpha_to_one=False,
        )
    elif scheduler_type == "ddpm":
        scheduler = DDPMScheduler.from_config(original_scheduler_config)
    elif scheduler_type == "deis-multi":
        scheduler = DEISMultistepScheduler.from_config(original_scheduler_config)
    else:
        raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")
    return scheduler
