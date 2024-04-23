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

from paddlemix.models.internlm_xcomposer2.configuration import InternLMXcomposer2Config
from paddlemix.models.internlm_xcomposer2.modeling import InternLMXComposer2ForCausalLM
from tests.models.test_configuration_common import ConfigTester
from tests.models.test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from tests.testing_utils import slow


class InternLMXComposer2ForCausalLMTester:
    def __init__(self, parent):
        self.parent = parent

    def get_config(self):
        test_config = {
            "bias": False,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 14336,
            "max_length": 4096,
            "max_position_embeddings": 32768,
            "model_type": "internlmxcomposer2",
            "num_attention_heads": 32,
            "num_hidden_layers": 1,
            "num_key_value_heads": 8,
            "pad_token_id": 2,
            "rms_norm_eps": 1e-05,
            "rope_scaling": {"factor": 1.0, "type": "dynamic"},
            "rope_theta": 1000000,
            "tie_word_embeddings": False,
            "torch_dtype": "float32",
            "transformers_version": "4.33.1",
            "use_cache": False,
            "vocab_size": 92544,
            "img_size": 224,
            "vision_tower": "openai/clip-vit-large-patch14-336",
            "vision_projector": {"mlp_depth": 2, "mm_hidden_size": 1024, "hidden_size": 4096},
        }

        return InternLMXcomposer2Config(**test_config)

    def prepare_config_and_inputs(self):
        images = ([floats_tensor([3, 224, 224])],)
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
        model = InternLMXComposer2ForCausalLM(config=self.get_config())
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


class InternLMXComposer2ForCausalLMTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (InternLMXComposer2ForCausalLM,)
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    use_test_model_name_list = False
    use_test_inputs_embeds: bool = False

    def setUp(self):
        self.model_tester = InternLMXComposer2ForCausalLMTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=InternLMXcomposer2Config,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            out_1 = first.numpy()
            out_2 = second.numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 5e-5)

        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                inputs = self._prepare_for_class(inputs_dict, model_class)
                first = model(**inputs)
                second = model(**inputs)

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    def test_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(**inputs_dict)

    @slow
    def test_model_from_pretrained(self):
        model = InternLMXComposer2ForCausalLM.from_pretrained("..../")
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
