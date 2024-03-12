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

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import PIL
from PIL import Image

from ...utils import (
    PPDIFFUSERS_SLOW_IMPORT,
    BaseOutput,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_paddle_available,
    is_paddlenlp_available,
)


@dataclass
class SafetyConfig(object):
    WEAK = {
        "sld_warmup_steps": 15,
        "sld_guidance_scale": 20,
        "sld_threshold": 0.0,
        "sld_momentum_scale": 0.0,
        "sld_mom_beta": 0.0,
    }
    MEDIUM = {
        "sld_warmup_steps": 10,
        "sld_guidance_scale": 1000,
        "sld_threshold": 0.01,
        "sld_momentum_scale": 0.3,
        "sld_mom_beta": 0.4,
    }
    STRONG = {
        "sld_warmup_steps": 7,
        "sld_guidance_scale": 2000,
        "sld_threshold": 0.025,
        "sld_momentum_scale": 0.5,
        "sld_mom_beta": 0.7,
    }
    MAX = {
        "sld_warmup_steps": 0,
        "sld_guidance_scale": 5000,
        "sld_threshold": 1.0,
        "sld_momentum_scale": 0.5,
        "sld_mom_beta": 0.7,
    }


_dummy_objects = {}
_additional_imports = {}
_import_structure = {}

_additional_imports.update({"SafetyConfig": SafetyConfig})

try:
    if not (is_paddlenlp_available() and is_paddle_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_paddle_and_paddlenlp_objects

    _dummy_objects.update(get_objects_from_module(dummy_paddle_and_paddlenlp_objects))
else:
    _import_structure.update(
        {
            "pipeline_output": ["StableDiffusionSafePipelineOutput"],
            "pipeline_stable_diffusion_safe": ["StableDiffusionPipelineSafe"],
            "safety_checker": ["StableDiffusionSafetyChecker"],
        }
    )


if TYPE_CHECKING or PPDIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_paddlenlp_available() and is_paddle_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_paddle_and_paddlenlp_objects import *
    else:
        from .pipeline_output import StableDiffusionSafePipelineOutput
        from .pipeline_stable_diffusion_safe import StableDiffusionPipelineSafe
        from .safety_checker import SafeStableDiffusionSafetyChecker

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )

    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)
    for name, value in _additional_imports.items():
        setattr(sys.modules[__name__], name, value)
