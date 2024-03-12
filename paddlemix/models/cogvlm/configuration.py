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

from typing import Literal

from paddlenlp import transformers


class CogModelConfig(transformers.PretrainedConfig):
    _auto_class = "AutoConfig"

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        cross_hidden_size=1024,
        cross_compute_hidden_size=1024,
        cross_image_size=1120,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        template_version: Literal["base", "chat"] = "chat",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_cache=True,
        model_type="cogagent",
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.cross_hidden_size = cross_hidden_size
        self.cross_compute_hidden_size = cross_compute_hidden_size
        self.cross_image_size = cross_image_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.template_version = template_version
        self.use_cache = use_cache
        self.model_type = model_type
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
