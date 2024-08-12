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

import paddle
from paddlemix.models.internvl2.internlm2 import InternLM2Config, InternLM2Tokenizer
from paddlemix.models.internvl2.internvl_chat import InternVisionConfig, InternVLChatModel, InternVLChatConfig
from test_modeling_common import floats_tensor


class InternVLChatModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.model_name_or_path = "OpenGVLab/InternVL2-2B"
        self.tokenizer = InternLM2Tokenizer.from_pretrained(self.model_name_or_path)
        # TODO
        self.tokenizer.added_tokens_encoder = {'<unk>': 0, '<s>': 1, '</s>': 2, '<|plugin|>': 92538, '<|interpreter|>': 92539, '<|action_end|>': 92540, '<|action_start|>': 92541, '<|im_end|>': 92542, '<|im_start|>': 92543, '<img>': 92544, '</img>': 92545, '<IMG_CONTEXT>': 92546, '<quad>': 92547, '</quad>': 92548, '<ref>': 92549, '</ref>': 92550, '<box>': 92551, '</box>': 92552}
        self.tokenizer.added_tokens_decoder = {v: k for k, v in self.tokenizer.added_tokens_encoder.items()}

    def get_config(self):
        # InternVL2-2B
        test_llm_config = {
            "_name_or_path": "internlm/internlm2_5-7b-chat",
            "add_cross_attention": False,
            "architectures": [
                "InternLM2ForCausalLM"
            ],
            "attn_implementation": "flash_attention_2",
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
            "hidden_size": 4096,
            "id2label": {
                "0": "LABEL_0",
                "1": "LABEL_1"
            },
            "initializer_range": 0.02,
            "intermediate_size": 14336,
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
            "num_attention_heads": 32,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_hidden_layers": 32,
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
            "torch_dtype": "bfloat16",
            "torchscript": False,
            "typical_p": 1.0,
            "use_bfloat16": True,
            "use_cache": True,
            "vocab_size": 92553,
        }

        test_vision_config = {
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
            "torch_dtype": "bfloat16",
            "use_bfloat16": True,
            "use_flash_attn": True,
        }
        llm_config = InternLM2Config(**test_llm_config)
        vision_config = InternVisionConfig(**test_vision_config)
        return InternVLChatConfig(vision_config=vision_config, llm_config=llm_config)

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([7, 3, 448, 448])
        config = self.get_config()
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values = self.prepare_config_and_inputs()

        inputs_dict = {
            "pixel_values": pixel_values,
        }

        return config, inputs_dict

    def create_and_check_model(self, pixel_values):
        model = InternVLChatModel(config=self.get_config())
        model.eval()
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        with paddle.no_grad():
            result = model.chat(
                tokenizer=self.tokenizer,
                pixel_values=pixel_values,
                question='Who are you?',
                generation_config=generation_config,
            )

        self.parent.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
