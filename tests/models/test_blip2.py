# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import inspect
import tempfile
import unittest

import numpy as np
import paddle
import paddle.nn as nn
from paddlenlp.transformers.opt.configuration import OPTConfig

from paddlemix.models.blip2 import (
    Blip2Config,
    Blip2ForConditionalGeneration,
    Blip2QFormerConfig,
    Blip2VisionConfig,
)
from paddlemix.models.blip2.eva_vit import VisionTransformer
from paddlemix.models.blip2.modeling import BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST
from paddlemix.models.blip2.Qformer import BertLMHeadModel
from tests.models.test_configuration_common import ConfigTester
from tests.models.test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from tests.testing_utils import slow


class Blip2VisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        is_training=True,
        num_channels=3,
        depth=39,
        drop_rate=0,
        embed_dim=1408,
        epsilon=1e-06,
        gradient_checkpointing=False,
        img_size=224,
        mlp_ratio=4.3637,
        num_heads=16,
        patch_size=14,
        qkv_bias=True,
        return_dict=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = img_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.embed_dim = embed_dim
        self.drop_rate = drop_rate
        self.epsilon = epsilon
        self.drop_rate = drop_rate
        self.gradient_checkpointing = gradient_checkpointing
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.return_dict = return_dict
        self.depth = depth
        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (img_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return Blip2VisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            drop_rate=self.drop_rate,
            epsilon=self.epsilon,
            gradient_checkpointing=self.gradient_checkpointing,
            mlp_ratio=self.mlp_ratio,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            return_dict=self.return_dict,
            depth=self.depth,
        )

    def create_and_check_model(self, config, pixel_values):
        model = VisionTransformer(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(
            result.shape,
            [self.batch_size, num_patches + 1, self.embed_dim],
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


class Blip2VisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as BLIP-2's vision encoder does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (VisionTransformer,)
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = Blip2VisionModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=Blip2VisionConfig,
            has_text_modality=False,
            hidden_size=37,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="BLIP-2's vision encoder does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            # self.assertIsInstance(model.get_input_embeddings(), (nn.Layer))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = VisionTransformer.from_pretrained(model_name)
            self.assertIsNotNone(model)


class BertLMHeadModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=32,
        is_training=True,
        add_cross_attention=True,
        attention_probs_dropout_prob=0.1,
        cross_attention_freq=2,
        embed_dim=256,
        fuse=False,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=768,
        initializer_range=0.02,
        intermediate_size=3072,
        layer_norm_eps=1e-12,
        max_position_embedding=512,
        model_type="bert",
        num_attention_heads=12,
        num_hidden_layers=12,
        num_query_tokens=32,
        pad_token_id=0,
        pool_act="tanh",
        type_vocab_size=2,
        vocab_size=30522,
        text_hidden_size=2560,
        encoder_width=1408,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.add_cross_attention = add_cross_attention
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.cross_attention_freq = cross_attention_freq
        self.embed_dim = embed_dim
        self.fuse = fuse
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embedding = max_position_embedding
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_query_tokens = num_query_tokens
        self.pad_token_id = pad_token_id
        self.pool_act = pool_act
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size
        self.encoder_width = encoder_width
        self.text_hidden_size = text_hidden_size

    def prepare_config_and_inputs(self):
        query_embeds = floats_tensor([self.batch_size, self.seq_length, self.hidden_size])
        encoder_hidden_states = floats_tensor([self.batch_size, self.embed_dim + 1, self.encoder_width])
        encoder_attention_mask = random_attention_mask([self.batch_size, self.embed_dim + 1])
        config = self.get_config()

        return config, query_embeds, encoder_hidden_states, encoder_attention_mask

    def get_config(self):
        return Blip2QFormerConfig(
            add_cross_attention=self.add_cross_attention,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            cross_attention_freq=self.cross_attention_freq,
            embed_dim=self.embed_dim,
            fuse=self.fuse,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            hidden_size=self.hidden_size,
            initializer_range=self.initializer_range,
            intermediate_size=self.intermediate_size,
            layer_norm_eps=self.layer_norm_eps,
            max_position_embedding=self.max_position_embedding,
            model_type=self.model_type,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            num_query_tokens=self.num_query_tokens,
            pad_token_id=self.pad_token_id,
            pool_act=self.pool_act,
            type_vocab_size=self.type_vocab_size,
            vocab_size=self.vocab_size,
        )

    def create_and_check_model(self, config, query_embeds, encoder_hidden_states, encoder_attention_mask):
        model = BertLMHeadModel(
            config=config, encoder_width=self.encoder_width, text_hidden_size=self.text_hidden_size
        )
        model = model.bert
        model.eval()
        result = model(
            query_embeds=query_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            [self.batch_size, self.seq_length, self.hidden_size],
        )

        model = BertLMHeadModel(
            config=config, encoder_width=self.encoder_width, text_hidden_size=self.text_hidden_size
        )
        model.eval()
        with paddle.no_grad():
            result = model(
                query_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )

        self.parent.assertEqual(
            result.last_hidden_state.shape,
            [self.batch_size, self.seq_length, self.hidden_size],
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            query_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
        ) = config_and_inputs
        inputs_dict = {
            "query_embeds": query_embeds,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }
        return config, inputs_dict


class BertLMHeadModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (BertLMHeadModel,)
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = BertLMHeadModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=Blip2QFormerConfig,
            has_text_modality=False,
            hidden_size=37,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config=config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_ids"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_save_load(self):
        pass


class Blip2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=50272,
        hidden_size=2560,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=2048,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        embed_dim=16,
        num_labels=3,
        word_embed_proj_dim=2560,
        type_sequence_label_size=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.type_sequence_label_size = type_sequence_label_size
        self.word_embed_proj_dim = word_embed_proj_dim
        self.is_encoder_decoder = False

    def prepare_config_and_inputs(self):
        config = self.get_config()

        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size, dtype="int64").clip(
            3,
        )
        input_ids[:, -1] = self.eos_token_id  # Eos Token

        attention_mask = input_ids.not_equal(paddle.to_tensor([self.pad_token_id], dtype="int64")).cast("int64")

        return config, input_ids, attention_mask

    def get_config(self):
        return OPTConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            ffn_dim=self.intermediate_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            embed_dim=self.embed_dim,
            is_encoder_decoder=False,
            word_embed_proj_dim=self.word_embed_proj_dim,
        )


class Blip2ModelTester:
    def __init__(
        self,
        parent,
        vision_kwargs=None,
        qformer_kwargs=None,
        text_kwargs=None,
        is_training=True,
        num_query_tokens=10,
    ):
        if vision_kwargs is None:
            vision_kwargs = {}
        if qformer_kwargs is None:
            qformer_kwargs = {}
        if text_kwargs is None:
            text_kwargs = {}

        self.parent = parent
        self.vision_model_tester = Blip2VisionModelTester(parent, **vision_kwargs)
        self.qformer_model_tester = BertLMHeadModelTester(parent, **qformer_kwargs)
        self.text_model_tester = Blip2TextModelTester(parent, **text_kwargs)
        self.is_training = is_training
        self.num_query_tokens = num_query_tokens

    def prepare_config_and_inputs(self):
        _, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        (
            _,
            input_ids,
            attention_mask,
        ) = self.text_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, pixel_values

    def get_config(self):
        return Blip2Config.from_vision_qformer_text_configs(
            vision_config=self.vision_model_tester.get_config(),
            qformer_config=self.qformer_model_tester.get_config(),
            text_config=self.text_model_tester.get_config(),
            num_query_tokens=self.num_query_tokens,
        )

    @unittest.skip(reason="BLIP-2's output needs to unified")
    def create_and_check_for_conditional_generation(self, config, input_ids, attention_mask, pixel_values):
        model = Blip2ForConditionalGeneration(config)
        model.eval()
        with paddle.no_grad():
            result = model(pixel_values, input_ids, attention_mask, return_dict=True)

        self.parent.assertEqual(
            result.logits.shape,
            [
                self.vision_model_tester.batch_size,
                self.text_model_tester.seq_length + self.num_query_tokens,
                self.text_model_tester.vocab_size,
            ],
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            pixel_values,
        ) = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        config.text_config = "facebook/opt-2.7b"
        return config, inputs_dict


class Blip2ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Blip2ForConditionalGeneration,)
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    use_test_model_name_list = False
    use_test_inputs_embeds: bool = False

    def setUp(self):
        self.model_tester = Blip2ModelTester(self)

    def test_for_conditional_generation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_conditional_generation(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            out_1 = first.numpy()
            out_2 = second.numpy()
            out_1 = out_1[~np.isnan(out_1)]
            out_2 = out_2[~np.isnan(out_2)]
            max_diff = np.amax(np.abs(out_1 - out_2))
            self.assertLessEqual(max_diff, 1e-5)

        for model_class in self.all_model_classes:
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                input = self._prepare_for_class(inputs_dict, model_class)
                first = model(**input)["loss"]
                second = model(**input)["loss"]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_load_vision_qformer_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save Blip2Config and check if we can load Blip2VisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = Blip2VisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save Blip2Config and check if we can load Blip2QFormerConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            qformer_config = Blip2QFormerConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.qformer_config.to_dict(), qformer_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        for model_name in BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST:
            model = Blip2ForConditionalGeneration.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_save_load(self):
        pass


if __name__ == "__main__":
    unittest.main()
