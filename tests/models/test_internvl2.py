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
os.environ["FLAGS_use_cuda_managed_memory"] = "True"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import unittest
import numpy as np
import tempfile
import inspect
import paddle

from paddlemix.models.internvl2.conversation import get_conv_template
from paddlemix.models.internvl2.internvl_chat import (
    InternVLChatConfig,
    InternVLChatModel,
)
from paddlemix.models.blip2.Qformer import BertLMHeadModel
from tests.models.test_modeling_common import ModelTesterMixin
from paddlemix.examples.internvl2.chat_demo import load_tokenizer

from tests.models.test_modeling_common import ModelTesterMixin
from tests.testing_utils import slow

paddle.set_grad_enabled(False)


def prepare_model_inputs(model, tokenizer, inputs_dict, question="Who are you?"):
    pixel_values = inputs_dict["pixel_values"]

    if "<image>" not in question:
        question = "<image>\n" + question
    # Generate template and image tokens
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    num_patches_list = [pixel_values.shape[0]]
    for num_patches in num_patches_list:
        image_tokens = "<img>" + "<IMG_CONTEXT>" * model.num_image_token * num_patches + "</img>"
        query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, add_special_tokens=True, return_tensors="pd")
    inputs_dict["input_ids"] = model_inputs["input_ids"]
    inputs_dict["attention_mask"] = model_inputs["attention_mask"]
    batch_size = inputs_dict["input_ids"].shape[0]
    position_ids = paddle.arange(inputs_dict["input_ids"].shape[1]).expand((batch_size, inputs_dict["input_ids"].shape[1]))
    inputs_dict["position_ids"] = position_ids

    return inputs_dict

class InternVLChatModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.model_name_or_path = "OpenGVLab/InternVL2_5-2B"
        self.tokenizer = load_tokenizer(self.model_name_or_path)

    def get_config(self):
        # InternVL2-2B
        test_config = {
            "_commit_hash": None,
            "architectures": [
                "InternVLChatModel"
            ],
            "auto_map": {
                "AutoConfig": "configuration_internvl_chat.InternVLChatConfig",
                "AutoModel": "modeling_internvl_chat.InternVLChatModel",
                "AutoModelForCausalLM": "modeling_internvl_chat.InternVLChatModel"
            },
            "downsample_ratio": 0.5,
            "dynamic_image_size": True,
            "force_image_size": 448,
            "llm_config": {
                "_name_or_path": "internlm/internlm2_5-1_8b-chat",
                "add_cross_attention": False,
                "architectures": [
                "InternLM2ForCausalLM"
                ],
                "attn_implementation": "eager",
                "auto_map": {
                "AutoConfig": "configuration_internlm2.InternLM2Config",
                "AutoModel": "modeling_internlm2.InternLM2ForCausalLM",
                "AutoModelForCausalLM": "modeling_internlm2.InternLM2ForCausalLM",
                "AutoModelForSequenceClassification": "modeling_internlm2.InternLM2ForSequenceClassification"
                },
                "bad_words_ids": None,
                "begin_suppress_tokens": None,
                "bias": False,
                "bos_token_id": 1,
                "chunk_size_feed_forward": 0,
                "cross_attention_hidden_size": None,
                "decoder_start_token_id": None,
                "diversity_penalty": 0.0,
                "do_sample": False,
                "early_stopping": False,
                "encoder_no_repeat_ngram_size": 0,
                "eos_token_id": 2,
                "exponential_decay_length_penalty": None,
                "finetuning_task": None,
                "forced_bos_token_id": None,
                "forced_eos_token_id": None,
                "hidden_act": "silu",
                "hidden_size": 2048,
                "id2label": {
                "0": "LABEL_0",
                "1": "LABEL_1"
                },
                "initializer_range": 0.02,
                "intermediate_size": 8192,
                "is_decoder": False,
                "is_encoder_decoder": False,
                "label2id": {
                "LABEL_0": 0,
                "LABEL_1": 1
                },
                "length_penalty": 1.0,
                "max_length": 20,
                "max_position_embeddings": 32768,
                "min_length": 0,
                "model_type": "internlm2",
                "no_repeat_ngram_size": 0,
                "num_attention_heads": 16,
                "num_beam_groups": 1,
                "num_beams": 1,
                "num_hidden_layers": 24,
                "num_key_value_heads": 8,
                "num_return_sequences": 1,
                "output_attentions": False,
                "output_hidden_states": False,
                "output_scores": False,
                "pad_token_id": 2,
                "prefix": None,
                "pretraining_tp": 1,
                "problem_type": None,
                "pruned_heads": {},
                "remove_invalid_values": False,
                "repetition_penalty": 1.0,
                "return_dict": True,
                "return_dict_in_generate": False,
                "rms_norm_eps": 1e-05,
                "rope_scaling": {
                "factor": 2.0,
                "type": "dynamic"
                },
                "rope_theta": 1000000,
                "sep_token_id": None,
                "suppress_tokens": None,
                "task_specific_params": None,
                "temperature": 1.0,
                "tf_legacy_loss": False,
                "tie_encoder_decoder": False,
                "tie_word_embeddings": False,
                "tokenizer_class": None,
                "top_k": 50,
                "top_p": 1.0,
                "dtype": "float16",
                "torchscript": False,
                "typical_p": 1.0,
                "use_bfloat16": False,
                "use_cache": True,
                "vocab_size": 92553
            },
            "max_dynamic_patch": 12,
            "min_dynamic_patch": 1,
            "model_type": "internvl_chat",
            "ps_version": "v2",
            "select_layer": -1,
            "template": "internvl2_5",
            "dtype": "float16",
            "use_backbone_lora": 0,
            "use_llm_lora": 0,
            "use_thumbnail": True,
            "vision_config": {
                "architectures": [
                "InternVisionModel"
                ],
                "attention_dropout": 0.0,
                "drop_path_rate": 0.0,
                "dropout": 0.0,
                "hidden_act": "gelu",
                "hidden_size": 1024,
                "image_size": 448,
                "initializer_factor": 1.0,
                "initializer_range": 0.02,
                "intermediate_size": 4096,
                "layer_norm_eps": 1e-06,
                "model_type": "intern_vit_6b",
                "norm_type": "layer_norm",
                "num_attention_heads": 16,
                "num_channels": 3,
                "num_hidden_layers": 24,
                "output_attentions": False,
                "output_hidden_states": False,
                "patch_size": 14,
                "qk_normalization": False,
                "qkv_bias": True,
                "return_dict": True,
                "dtype": "float16",
                "use_bfloat16": False,
                "use_flash_attn": False
            }
            }

        return InternVLChatConfig(**test_config)

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()
        pixel_values = paddle.to_tensor(np.random.rand(14, 3, 448, 448), dtype='float16')  
        input_ids = paddle.to_tensor(np.random.randint(0, 1000, (2, 1918)), dtype='int64')  
        attention_mask = paddle.to_tensor(np.ones((2, 1918)), dtype='int64')  
        position_ids = paddle.to_tensor(np.arange(1918).reshape(1, -1).repeat(2, axis=0), dtype='int64')  
        image_flags = paddle.to_tensor(np.ones((14, 1)), dtype='int64')  


        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "image_flags": image_flags,
            "labels": None
        }

        return config, inputs_dict
    
    def create_and_check_model(self, pixel_values):
        model = InternVLChatModel.from_pretrained(self.model_name_or_path, dtype="float16")
        model.eval()
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        with paddle.no_grad():
            result = model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question="Who are you?",
                generation_config=generation_config,
            )

        self.parent.assertIsNotNone(result)

class InternVLChatModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (InternVLChatModel,)
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    use_test_model_name_list = False
    use_test_inputs_embeds: bool = False

    def setUp(self):
        # model tester instance
        self.model_tester = InternVLChatModelTester(self)

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            # Handle both tuple outputs and model output objects
            if hasattr(first, 'logits'):
                first = first.logits
                second = second.logits
            out_1 = first.numpy()
            out_2 = second.numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 5e-5)

        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            img_context_token_id = self.model_tester.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            model.img_context_token_id = img_context_token_id
            model.eval()

            # Prepare inputs for chat function
            tokenizer = self.model_tester.tokenizer
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            inputs_dict = prepare_model_inputs(model, tokenizer, inputs_dict, question="Who are you?")

            with paddle.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]
                second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    def test_hidden_states_output(self):
        pass
    
    def test_save_load(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_save_load(out1, out2):
            # make sure we don't have nans
            out_2 = out2.numpy()
            out_2[np.isnan(out_2)] = 0

            out_1 = out1.numpy()
            out_1[np.isnan(out_1)] = 0
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = InternVLChatModel.from_pretrained(self.model_tester.model_name_or_path, dtype="float16")
            model.eval()
            img_context_token_id = self.model_tester.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            model.img_context_token_id = img_context_token_id
            model.eval()

            tokenizer = self.model_tester.tokenizer
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            inputs_dict = prepare_model_inputs(model, tokenizer, inputs_dict, question="Who are you?")

            with paddle.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, save_function=paddle.save)
                model = model_class.from_pretrained(tmpdirname, dtype="float16")
                model.img_context_token_id = img_context_token_id
                model.eval()
                with paddle.no_grad():
                    second = model(**self._prepare_for_class(inputs_dict, model_class))[0]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_save_load(tensor1, tensor2)
            else:
                check_save_load(first, second)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            if isinstance(model, BertLMHeadModel):
                model = model.bert
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]
            expected_arg_names = ['pixel_values']
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        pixel_values = inputs_dict["pixel_values"]
        self.model_tester.create_and_check_model(pixel_values)

    @slow
    def test_model_from_pretrained(self):
        model = InternVLChatModel.from_pretrained(self.model_tester.model_name_or_path)
        self.assertIsNotNone(model)

if __name__ == "__main__":
    unittest.main()