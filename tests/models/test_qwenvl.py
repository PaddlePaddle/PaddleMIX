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

os.environ["FLAGS_use_cuda_managed_memory"] = "true"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import unittest

import paddle
from paddlenlp.transformers.qwen.configuration import QWenConfig

from paddlemix import QWenLMHeadModel, QwenVLProcessor, QWenVLTokenizer
from tests.models.test_configuration_common import ConfigTester
from tests.models.test_modeling_common import ModelTesterMixin
from tests.testing_utils import slow


class QWenLMHeadModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.model_name_or_path = "qwen-vl/qwen-vl-chat-7b"
        self.tokenizer = QWenVLTokenizer.from_pretrained(self.model_name_or_path, dtype="float32")
        self.processor = QwenVLProcessor(tokenizer=self.tokenizer)

    def get_config(self):
        config = {
            "_name_or_path": "./",
            "architectures": ["QWenLMHeadModel"],
            "llm_pretrained_model_name_or_path": "qwen/qwen-7b",
            "attn_dropout_prob": 0.0,
            "auto_map": {"AutoConfig": "QWenConfig", "AutoModelForCausalLM": "QWenLMHeadModel"},
            "emb_dropout_prob": 0.0,
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 1,
            "kv_channels": 1,
            "layer_norm_epsilon": 1e-06,
            "max_position_embeddings": 2,
            "model_type": "qwen",
            "no_bias": True,
            "num_attention_heads": 1,
            "num_hidden_layers": 1,
            "onnx_safe": None,
            "rotary_emb_base": 1,
            "rotary_pct": 1.0,
            "scale_attn_weights": True,
            "seq_length": 2,
            "tie_word_embeddings": False,
            "tokenizer_type": "QWenTokenizer",
            "dtype": "float32",
            "transformers_version": "4.31.0",
            "use_cache": True,
            "recompute": True,
            "use_dynamic_ntk": True,
            "use_flash_attn": False,
            "use_logn_attn": True,
            "use_flash_attention": True,
            "use_fused_rms_norm": True,
            "use_fused_rope": True,
            "visual": {
                "heads": 1,
                "image_size": 448,
                "image_start_id": 151857,
                "layers": 1,
                "mlp_ratio": 1,
                "output_dim": 128,
                "patch_size": 14,
                "width": 1664,
            },
            "vocab_size": 2,
        }

        return QWenConfig(**config)

    def prepare_config_and_inputs(self):
        query = []
        query.append({"image": "https://bj.bcebos.com/v1/paddlenlp/models/community/GroundingDino/000000004505.jpg"})
        query.append({"text": "Generate the caption in English with grounding:"})
        inputs = self.processor(query=query, return_tensors="pd")
        config = self.get_config()

        return config, inputs

    def prepare_config_and_inputs_for_common(self):
        config, inputs = self.prepare_config_and_inputs()
        return config, inputs

    def create_and_check_model(self, kwargs):
        model = QWenLMHeadModel(config=self.get_config())
        model.eval()
        with paddle.no_grad():
            result = model(**kwargs)

        self.parent.assertIsNotNone(result)


class QWenLMHeadModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (QWenLMHeadModel,)
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    use_test_model_name_list = False
    use_test_inputs_embeds: bool = False

    def setUp(self):
        self.model_tester = QWenLMHeadModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=QWenConfig,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        self.model_tester.create_and_check_model(inputs_dict)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model = QWenLMHeadModel.from_pretrained("qwen-vl/qwen-vl-chat-7b")
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
