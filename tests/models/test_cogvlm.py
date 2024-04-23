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
import tempfile
import unittest

import numpy as np
import paddle

from paddlemix.models.blip2.Qformer import BertLMHeadModel
from paddlemix.models.cogvlm.configuration import CogModelConfig
from paddlemix.models.cogvlm.modeling import CogModelForCausalLM
from tests.models.test_configuration_common import ConfigTester
from tests.models.test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from tests.testing_utils import slow


class CogAgentForCausalLMTester:
    def __init__(self, parent):
        self.parent = parent

    def get_config(self):
        test_config = {
            "model_type": "cogagent",
            "bos_token_id": 1,
            "cross_compute_hidden_size": 1024,
            "cross_hidden_size": 1024,
            "cross_image_size": 1120,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 2,
            "initializer_range": 0.02,
            "intermediate_size": 2,
            "max_position_embeddings": 2048,
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "pad_token_id": 0,
            "paddlenlp_version": None,
            "rms_norm_eps": 1e-05,
            "template_version": "chat",
            "tie_word_embeddings": False,
            "transformers_version": "4.36.0.dev0",
            "vision_config": {
                "dropout_prob": 0.0,
                "hidden_act": "gelu",
                "hidden_size": 8,
                "image_size": 224,
                "in_channels": 3,
                "intermediate_size": 2,
                "layer_norm_eps": 1e-06,
                "num_heads": 1,
                "num_hidden_layers": 1,
                "num_positions": 257,
                "patch_size": 14,
            },
            "vocab_size": 32000,
        }
        return CogModelConfig(**test_config)

    def prepare_config_and_inputs(self):
        images = ([floats_tensor([3, 224, 224])],)
        cross_images = ([floats_tensor([3, 1120, 1120])],)
        tokenized_out = {
            "input_ids": ids_tensor([1, 258], 5000),
            "token_type_ids": random_attention_mask([1, 258]),
            "attention_mask": random_attention_mask([1, 258]),
            "position_ids": ids_tensor([1, 258], vocab_size=100),
        }

        config = self.get_config()

        return config, images, cross_images, tokenized_out

    def prepare_config_and_inputs_for_common(self):
        config, images, cross_images, tokenized_out = self.prepare_config_and_inputs()

        inputs_dict = {
            "images": images,
            "cross_images": cross_images,
            "input_ids": tokenized_out["input_ids"],
            "attention_mask": tokenized_out["attention_mask"],
            "token_type_ids": tokenized_out["token_type_ids"],
            "position_ids": tokenized_out["position_ids"],
        }

        return config, inputs_dict

    def create_and_check_model(self, images, cross_images, input_ids, attention_mask, token_type_ids, position_ids):
        model = CogModelForCausalLM(config=self.get_config())
        model.eval()
        with paddle.no_grad():
            result = model(
                images=images,
                cross_images=cross_images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
            )

        self.parent.assertIsNotNone(result)


class CogAgentForCausalLMTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (CogModelForCausalLM,)
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    use_test_model_name_list = False
    use_test_inputs_embeds: bool = False

    def setUp(self):
        self.model_tester = CogAgentForCausalLMTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=CogModelConfig,
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
        model = CogModelForCausalLM.from_pretrained("THUDM/cogagent-chat")
        self.assertIsNotNone(model)


class CogVLMForCausalLMTester(CogAgentForCausalLMTester):
    def get_config(self):
        test_config = {
            "model_type": "cogvlm",
            "bos_token_id": 1,
            # "cross_compute_hidden_size": 1024,
            # "cross_hidden_size": 1024,
            # "cross_image_size": 1120,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 2,
            "initializer_range": 0.02,
            "intermediate_size": 2,
            "max_position_embeddings": 2048,
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "pad_token_id": 0,
            "paddlenlp_version": None,
            "rms_norm_eps": 1e-05,
            "template_version": "chat",
            "tie_word_embeddings": False,
            "transformers_version": "4.36.0.dev0",
            "vision_config": {
                "dropout_prob": 0.0,
                "hidden_act": "gelu",
                "hidden_size": 8,
                "image_size": 224,
                "in_channels": 3,
                "intermediate_size": 2,
                "layer_norm_eps": 1e-06,
                "num_heads": 1,
                "num_hidden_layers": 1,
                "num_positions": 257,
                "patch_size": 14,
            },
            "vocab_size": 32000,
        }
        return CogModelConfig(**test_config)


class CogVLMForCausalLMTest(CogAgentForCausalLMTest):
    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_save_load(out1, out2):
            # make sure we don't have nans
            out_2 = out2.numpy()
            out_2[np.isnan(out_2)] = 0

            out_1 = out1.numpy()
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 5e-5)

        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            if isinstance(model, BertLMHeadModel):
                model = model.bert
            model.eval()
            with paddle.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, save_function=paddle.save)
                model = model_class.from_pretrained(tmpdirname)
                model.eval()
                with paddle.no_grad():
                    second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            # support tuple of tensor
            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_save_load(tensor1, tensor2)
            else:
                check_save_load(first, second)

    @slow
    def test_model_from_pretrained(self):
        model = CogModelForCausalLM.from_pretrained("THUDM/cogvlm-chat")
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
