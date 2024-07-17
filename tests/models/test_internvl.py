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

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import unittest

import numpy as np
import paddle

from paddlemix.models.internvl2.configuration import InternVL2Config, InternVisionConfig, InternLM2Config
from paddlemix.models.internvl2.modeling import InternVL2ForCausalLM
from test_configuration_common import ConfigTester
from test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from tests.testing_utils import slow


class InternVL2ForCausalLMTester:
    def __init__(self, parent):
        self.parent = parent

    def get_config(self):
        # InternVL2-8b
        test_llm_config = {
            "vocab_size": 92553,
            "bias": False,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 14336,
            "max_length": 20,
            "max_position_embeddings": 32768,
            "model_type": "internvl2",
            "num_attention_heads": 32,
            "num_hidden_layers": 32, # for testing
            "num_key_value_heads": 8,
            "pad_token_id": 2,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {"factor": 2.0, "type": "dynamic"},
            "rope_theta": 1000000,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.37.2",
            "use_cache": True,
            "attn_implementation": 'flash_attention',
        }
        test_vision_config = {
            "model_type": "intern_vit_6b",
            "image_size": 448,
            "qkv_bias": True,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "qk_normalization": False,
            "num_hidden_layers": 24, # for testing
            "hidden_act": "gelu",
            "norm_type": "layer_norm",
            "layer_norm_eps": 1e-06,
            "initializer_range": 0.02,
            "initializer_factor": 1.0,
            "use_flash_attn": True,
            "torch_dtype": "bfloat16",
            "use_bfloat16": True,
        }
        vision = InternVisionConfig(**test_vision_config)
        llm = InternLM2Config(**test_llm_config)
        return InternVL2Config(vision_config=vision, llm_config=llm)

    def prepare_config_and_inputs(self):
        images = ([floats_tensor([3, 448, 448])],)
        tokenized_out = {
            "input_ids": ids_tensor([1, 258], 5000),
            "token_type_ids": random_attention_mask([1, 258]),
            "attention_mask": random_attention_mask([1, 258]),
            "position_ids": ids_tensor([1, 258], vocab_size=100),
        }

        config = self.get_config()

        return config, images, tokenized_out

    def prepare_config_and_inputs_for_common(self):
        config, images, tokenized_out = self.prepare_config_and_inputs()

        inputs_dict = {
            "images": images,
            "input_ids": tokenized_out["input_ids"],
            "attention_mask": tokenized_out["attention_mask"],
            "token_type_ids": tokenized_out["token_type_ids"],
            "position_ids": tokenized_out["position_ids"],
        }

        return config, inputs_dict

    def create_and_check_model(self, images, input_ids, attention_mask, token_type_ids, position_ids):
        model = InternVL2ForCausalLM(config=self.get_config())
        model.eval()
        with paddle.no_grad():
            result = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )

        self.parent.assertIsNotNone(result)


class InternVL2ForCausalLMTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (InternVL2ForCausalLM, )
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    use_test_model_name_list = False
    use_test_inputs_embeds: bool = False

    def setUp(self):
        self.model_tester = InternVL2ForCausalLM(self)
        self.config_tester = ConfigTester(
            self,
            config_class=InternVL2Config,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    def test_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(**inputs_dict)

    @slow
    def test_model_from_pretrained(self):
        model = InternVL2ForCausalLM.from_pretrained("..../")
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()