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

from paddlenlp.transformers.configuration_utils import PretrainedConfig

class GPT2Config(PretrainedConfig):
    
    model_type = "gpt2"
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
      self,
      vocab_size=50257,
      n_positions=1024,
      n_embd=768,
      n_layer=12,
      n_head=12,
      n_inner=None,
      activation_function="gelu_new",
      resid_pdrop=0.1,
      embd_pdrop=0.1,
      attn_pdrop=0.1,
      layer_norm_epsilon=1e-5,
      initializer_range=0.02,
      summary_type="cls_index",
      summary_use_proj=True,
      summary_activation=None,
      summary_proj_to_labels=True,
      summary_first_dropout=0.1,
      scale_attn_weights=True,
      use_cache=True,
      bos_token_id=50256,
      eos_token_id=50256,
      scale_attn_by_inverse_layer_idx=False,
      reorder_and_upcast_attn=False,
      **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx
        self.reorder_and_upcast_attn = reorder_and_upcast_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

def get_gpt2_config():
    return {
  "activation_function": "gelu_new",
  "architectures": [
    "GPT2LMHeadModel"
  ],
  "attn_pdrop": 0.1,
  "bos_token_id": 50256,
  "embd_pdrop": 0.1,
  "eos_token_id": 50256,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "gpt2",
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_inner": None,
  "n_layer": 12,
  "n_positions": 1024,
  "reorder_and_upcast_attn": False,
  "resid_pdrop": 0.1,
  "scale_attn_by_inverse_layer_idx": False,
  "scale_attn_weights": True,
  "summary_activation": None,
  "summary_first_dropout": 0.1,
  "summary_proj_to_labels": True,
  "summary_type": "cls_index",
  "summary_use_proj": True,
  "task_specific_params": {
    "text-generation": {
      "do_sample": True,
      "max_length": 50
    }
  },
  "transformers_version": "4.35.2",
  "use_cache": True,
  "vocab_size": 50257
}
