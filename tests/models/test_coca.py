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

import inspect
import unittest

import paddle
import paddle.nn as nn

from paddlemix.models.clip.coca_model import CoCa, CoCaConfig
from paddlemix.models.clip.multi_modal_model import (
    MultimodalTransformer,
    MultimodalTransformerConfig,
)
from tests.models.test_configuration_common import ConfigTester
from tests.models.test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)
from tests.testing_utils import slow

CoCa_PRETRAINED_MODEL_ARCHIVE_LIST = ["paddlemix/CoCa/coca_Vit-L-14"]


class MultimodalTransformerModelTester:
    def __init__(
        self,
        parent,
        width: int = 512,
        layers: int = 12,
        heads: int = 8,
        context_length: int = 76,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        quick_gelu: bool = False,
        cast_dtype=None,
        vocab_size: int = 512,
        xattn: bool = False,
        batchsize: int = 4,
    ):
        self.parent = parent
        self.width = width
        self.layers = layers
        self.heads = heads
        self.context_length = context_length
        self.mlp_ratio = mlp_ratio
        self.ls_init_value = ls_init_value
        self.quick_gelu = quick_gelu
        self.cast_dtype = cast_dtype
        self.vocab_size = vocab_size
        self.xattn = xattn
        self.batch_size = batchsize

    def prepare_config_and_inputs(self):
        image_embs = floats_tensor([self.batch_size, 255, self.vocab_size])
        text_embs = floats_tensor([self.batch_size, 76, self.vocab_size])
        config = self.get_config()

        return config, image_embs, text_embs

    def get_config(self):
        return MultimodalTransformerConfig(
            width=self.width,
            layers=self.layers,
            heads=self.heads,
            context_length=self.context_length,
            mlp_ratio=self.mlp_ratio,
            ls_init_value=self.ls_init_value,
            quick_gelu=self.quick_gelu,
            cast_dtype=self.cast_dtype,
            vocab_size=self.vocab_size,
            xattn=self.xattn,
        )

    def create_and_check_model(self, config, image_embs, text_embs):
        model = MultimodalTransformer(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(image_embs, text_embs)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        self.parent.assertEqual(
            result.shape,
            [self.batch_size, self.context_length, self.vocab_size],
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, image_embs, text_embs = config_and_inputs
        inputs_dict = {"image_embs": image_embs, "text_embs": text_embs}
        return config, inputs_dict


class MultimodalTransformerModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (MultimodalTransformer,)
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = MultimodalTransformerModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=MultimodalTransformerConfig,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="CoCa's vision encoder does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="CoCa's text encoder does not use inputs_embeds and output_embeds")
    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Layer))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["image_embs"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)


class CoCaModelTester:
    def __init__(
        self,
        parent,
        vision_cfg={
            "embed_dim": 768,
            "image_size": 224,
            "layers": 24,
            "width": 1024,
            "patch_size": 14,
            "attentional_pool": True,
            "attn_pooler_heads": 8,
            "output_tokens": True,
            "batchsize": 4,
        },
        text_cfg={
            "embed_dim": 768,
            "context_length": 76,
            "vocab_size": 49408,
            "width": 768,
            "heads": 12,
            "layers": 12,
            "embed_cls": True,
            "output_tokens": True,
        },
        multimodal_cfg={
            "context_length": 76,
            "vocab_size": 49408,
            "width": 768,
            "heads": 12,
            "layers": 12,
            "attn_pooler_heads": 12,
        },
        batch_size=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.vision_cfg = vision_cfg
        self.text_cfg = text_cfg
        self.multimodal_cfg = multimodal_cfg

    def prepare_config_and_inputs(self):
        image = floats_tensor([self.batch_size, 3, self.vision_cfg["image_size"], self.vision_cfg["image_size"]])
        text = ids_tensor(
            [self.batch_size, self.text_cfg["context_length"] + 1], self.text_cfg["vocab_size"], dtype="int64"
        )
        config = self.get_config()

        return config, image, text

    def get_config(self):
        return CoCaConfig(
            vision_cfg=self.vision_cfg,
            text_cfg=self.text_cfg,
            multimodal_cfg=self.multimodal_cfg,
        )

    def create_and_check_model(self, config, image, text):
        model = CoCa(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(image, text, skiploss=True)

        self.parent.assertEqual(
            result[1].shape,
            [self.batch_size, self.vision_cfg["embed_dim"]],
        )
        self.parent.assertEqual(
            result[2].shape,
            [self.batch_size, self.text_cfg["embed_dim"]],
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, image, text = config_and_inputs
        inputs_dict = {"image": image, "input_ids": text}
        return config, inputs_dict


class CoCaModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (CoCa,)
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = CoCaModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=CoCaConfig,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="CoCa's vision encoder does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="CoCa's text encoder does not use inputs_embeds and output_embeds")
    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Layer))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["image", "input_ids"]
            self.assertListEqual(arg_names[:2], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in CoCa_PRETRAINED_MODEL_ARCHIVE_LIST:
            model = CoCa.from_pretrained(model_name)
            self.assertIsNotNone(model)

    def test_determinism(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_determinism(first, second):
            import numpy as np

            print(first, second)
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
                first = model(**self._prepare_for_class(inputs_dict, model_class))[1]
                second = model(**self._prepare_for_class(inputs_dict, model_class))[1]

            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_determinism(tensor1, tensor2)
            else:
                check_determinism(first, second)
