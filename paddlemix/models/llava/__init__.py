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

import pkg_resources

from .language_model.llava_llama import *
from .language_model.tokenizer import *
from .mm_utils import *
from .multimodal_encoder.clip_encoder import *
from .multimodal_encoder.siglip_encoder import *
from .multimodal_projector.builder import *

version = pkg_resources.get_distribution("paddlenlp").version
try:
    if version.startswith("3"):
        from .language_model.llava_qwen import *
    else:
        print(
            f"paddlenlp version {version} is not 3.x, skipping import Qwen2Model for llava-next-interleave/llava_onevision/llava_critic."
        )

except ImportError:
    print("paddlenlp is not installed.")
