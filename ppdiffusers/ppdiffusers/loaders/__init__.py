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

from typing import TYPE_CHECKING

from ..utils import PPDIFFUSERS_SLOW_IMPORT, _LazyModule
from ..utils.import_utils import is_paddle_available, is_paddlenlp_available

_import_structure = {}

if is_paddle_available():
    _import_structure["single_file"] = ["FromOriginalControlnetMixin", "FromOriginalVAEMixin"]
    _import_structure["unet"] = ["UNet2DConditionLoadersMixin"]
    _import_structure["utils"] = ["AttnProcsLayers"]

    if is_paddlenlp_available():
        _import_structure["single_file"].extend(["FromSingleFileMixin", "FromCkptMixin"])
        _import_structure["lora"] = [
            "LoraLoaderMixin",
            "StableDiffusionXLLoraLoaderMixin",
            "SD3LoraLoaderMixin",
            "WanLoraLoaderMixin",
        ]
        _import_structure["textual_inversion"] = ["TextualInversionLoaderMixin"]
        _import_structure["ip_adapter"] = ["IPAdapterMixin"]
        # NOTE: this will removed in the future
        _import_structure["deprecate"] = ["text_encoder_lora_state_dict", "text_encoder_attn_modules"]

if TYPE_CHECKING or PPDIFFUSERS_SLOW_IMPORT:
    if is_paddle_available():
        from .single_file import FromOriginalControlnetMixin, FromOriginalVAEMixin
        from .unet import UNet2DConditionLoadersMixin
        from .utils import AttnProcsLayers

        if is_paddlenlp_available():
            # NOTE: this will removed in the future
            from .deprecate import (
                text_encoder_attn_modules,
                text_encoder_lora_state_dict,
            )
            from .ip_adapter import IPAdapterMixin
            from .lora import (
                FluxLoraLoaderMixin,
                LoraLoaderMixin,
                SD3LoraLoaderMixin,
                StableDiffusionXLLoraLoaderMixin,
                WanLoraLoaderMixin,
            )
            from .single_file import FromCkptMixin, FromSingleFileMixin
            from .textual_inversion import TextualInversionLoaderMixin
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
