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

from typing import TYPE_CHECKING

from ...utils import (
    PPDIFFUSERS_SLOW_IMPORT,
    OptionalDependencyNotAvailable,
    _LazyModule,
    get_objects_from_module,
    is_paddle_available,
    is_paddlenlp_available,
)

_dummy_objects = {}
_import_structure = {}

try:
    if not (is_paddlenlp_available() and is_paddle_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils import dummy_paddle_and_paddlenlp_objects

    _dummy_objects.update(get_objects_from_module(dummy_paddle_and_paddlenlp_objects))
else:
    _import_structure["modeling_roberta_series"] = ["RobertaSeriesModelWithTransformation", "RobertaSeriesConfig"]
    _import_structure["pipeline_alt_diffusion"] = ["AltDiffusionPipeline"]
    _import_structure["pipeline_alt_diffusion_img2img"] = ["AltDiffusionImg2ImgPipeline"]

    _import_structure["pipeline_output"] = ["AltDiffusionPipelineOutput"]

if TYPE_CHECKING or PPDIFFUSERS_SLOW_IMPORT:
    try:
        if not (is_paddlenlp_available() and is_paddle_available()):
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        from ...utils.dummy_paddle_and_paddlenlp_objects import *

    else:
        from .modeling_roberta_series import (
            RobertaSeriesConfig,
            RobertaSeriesModelWithTransformation,
        )
        from .pipeline_alt_diffusion import AltDiffusionPipeline
        from .pipeline_alt_diffusion_img2img import AltDiffusionImg2ImgPipeline
        from .pipeline_output import AltDiffusionPipelineOutput

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
