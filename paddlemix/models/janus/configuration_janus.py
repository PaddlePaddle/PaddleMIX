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

from typing import Dict

from paddlenlp.transformers import LlamaConfig
from paddlenlp.transformers.configuration_utils import PretrainedConfig


class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: Dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__
        self.params = dict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    cls: str = ""
    params: Dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__
        self.params = dict(kwargs.get("params", {}))


class GenVisionConfig(PretrainedConfig):
    model_type = "gen_vision"
    cls: str = ""
    params: Dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__
        self.params = dict(kwargs.get("params", {}))


class GenAlignerConfig(PretrainedConfig):
    model_type = "gen_aligner"
    cls: str = ""
    params: Dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__
        self.params = dict(kwargs.get("params", {}))


class GenHeadConfig(PretrainedConfig):
    model_type = "gen_head"
    cls: str = ""
    params: Dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__
        self.params = dict(kwargs.get("params", {}))


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig
    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig
    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)
        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)
        gen_vision_config = kwargs.get("gen_vision_config", {})
        self.gen_vision_config = GenVisionConfig(**gen_vision_config)
        gen_aligner_config = kwargs.get("gen_aligner_config", {})
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)
        gen_head_config = kwargs.get("gen_head_config", {})
        self.gen_head_config = GenHeadConfig(**gen_head_config)
        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)


# Janus Flow Config
class VisionUnderstandEncoderConfig(PretrainedConfig):
    model_type = "vision_und_enc"
    cls: str = ""
    params: Dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__
        self.params = dict(kwargs.get("params", {}))


class VisionGenerationEncoderConfig(PretrainedConfig):
    model_type = "vision_gen_enc"
    cls: str = ""
    params: Dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__
        self.params = dict(kwargs.get("params", {}))


class VisionGenerationDecoderConfig(PretrainedConfig):
    model_type = "vision_gen_dec"
    cls: str = ""
    params: Dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__
        self.params = dict(kwargs.get("params", {}))


class JanusFlowMultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_und_enc_config: VisionUnderstandEncoderConfig
    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_und_enc_config = kwargs.get("vision_und_enc_config", {})
        self.vision_und_enc_config = VisionUnderstandEncoderConfig(**vision_und_enc_config)
        vision_gen_enc_config = kwargs.get("vision_gen_enc_config", {})
        self.vision_gen_enc_config = VisionGenerationEncoderConfig(**vision_gen_enc_config)
        vision_gen_dec_config = kwargs.get("vision_gen_dec_config", {})
        self.vision_gen_dec_config = VisionGenerationDecoderConfig(**vision_gen_dec_config)
        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)
