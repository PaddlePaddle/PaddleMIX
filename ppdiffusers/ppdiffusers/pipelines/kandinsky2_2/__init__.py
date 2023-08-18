# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from ...utils import (
    OptionalDependencyNotAvailable,
    is_paddle_available,
    is_paddlenlp_available,
)

try:
    if not (is_paddlenlp_available() and is_paddle_available()):
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_paddle_and_paddlenlp_objects import *
else:
    from .pipeline_kandinsky2_2 import KandinskyV22Pipeline
    from .pipeline_kandinsky2_2_combined import (
        KandinskyV22CombinedPipeline,
        KandinskyV22Img2ImgCombinedPipeline,
        KandinskyV22InpaintCombinedPipeline,
    )
    from .pipeline_kandinsky2_2_controlnet import KandinskyV22ControlnetPipeline
    from .pipeline_kandinsky2_2_controlnet_img2img import (
        KandinskyV22ControlnetImg2ImgPipeline,
    )
    from .pipeline_kandinsky2_2_img2img import KandinskyV22Img2ImgPipeline
    from .pipeline_kandinsky2_2_inpainting import KandinskyV22InpaintPipeline
    from .pipeline_kandinsky2_2_prior import KandinskyV22PriorPipeline
    from .pipeline_kandinsky2_2_prior_emb2emb import KandinskyV22PriorEmb2EmbPipeline
