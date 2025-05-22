# -*- coding: utf-8 -*-

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# @Time    : 2025/4/19 下午8:37
# @Author  : zhaop-l(zhaopuzxjc@126.com)

from paddlenlp.transformers.configuration_utils import PretrainedConfig

from ppdiffusers.utils import logging

logger = logging.get_logger(__name__)


class CustomLlamaConfig(PretrainedConfig):
    """
    Args:
        vocab_size (`int`, *optional*, defaults to 50432):
            Vocabulary size of the WeLMV3 model. Defines the number of
                different tokens that can be represented by the
            `inputs_ids` passed when calling [`WeLMV3Model`].
        hidden_size (`int`, *optional*, defaults to 6144):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 44):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads for each attention layer in the
            Transformer encoder.
        num_kv_heads (`int`, *optional*, defaults to 4):
            Number of GQA groups.
        intermediate_size (`int`, *optional*, defaults to 24576):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the
            Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the
            encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        rotary_pct (`float`, *optional*, defaults to 0.25):
            percentage of hidden dimensions to allocate to rotary embeddings
        rotary_emb_base (`int`, *optional*, defaults to 10000)
            base for computing rotary embeddings frequency
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used
            with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
        initializer_range (`float`, *optional*, defaults to 1e-5):
            The standard deviation of the truncated_normal_initializer for
            initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values
            attentions (not used by all models). Only relevant if
            `config.is_decoder=True`.
    """

    model_type = "custom_llama"

    def __init__(
        self,
        vocab_size=102400,
        hidden_size=2560,
        num_layers=32,
        num_attention_heads=20,
        num_kv_heads=4,
        ffn_hidden_size=2560 * 4,
        hidden_act="swiglu",
        rotary_pct=1.0,
        rotary_emb_base=10000,
        rotary_compress=1.0,
        max_position_embeddings=4096,
        initializer_range=0.02,
        layernorm_epsilon=1e-05,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=2,
        rms_norm=None,
        norm_type="layer_norm",
        qkv_proj_bias=True,
        out_proj_bias=True,
        mlp_fc1_bias=True,
        mlp_fc2_bias=True,
        **kwargs
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.ffn_hidden_size = ffn_hidden_size
        self.hidden_act = hidden_act
        self.rotary_pct = rotary_pct
        self.rotary_emb_base = rotary_emb_base
        self.rotary_compress = rotary_compress
        self.initializer_range = initializer_range
        self.layernorm_epsilon = layernorm_epsilon
        self.use_cache = use_cache
        if rms_norm is not None:
            self.norm_type = "rms_norm" if rms_norm else "layer_norm"
        else:
            self.norm_type = norm_type
        self.qkv_proj_bias = qkv_proj_bias
        self.out_proj_bias = out_proj_bias
        self.mlp_fc1_bias = mlp_fc1_bias
        self.mlp_fc2_bias = mlp_fc2_bias
        self.num_hidden_layers = num_layers
