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
import unittest

import paddle
import paddle.nn as nn

from paddlemix.models.clip.clip_model import CLIP, CLIPConfig
from paddlemix.models.clip.text_model import TextTransformer, TextTransformerConfig
from paddlemix.models.clip.vit_model import VisionTransformer, VisionTransformerConfig
from tests.models.test_configuration_common import ConfigTester
from tests.models.test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)
from tests.testing_utils import slow

CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = ["paddlemix/CLIP/Vit-L-14"]


class VisionTransformerModelTester:
    def __init__(
        self,
        parent,
        image_size: int = 224,
        patch_size: int = 14,
        width: int = 768,
        layers: int = 12,
        head_width: int = 64,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        patch_dropout: float = 0.0,
        global_average_pool: bool = False,
        attentional_pool: bool = False,
        n_queries: int = 256,
        attn_pooler_heads: int = 8,
        embed_dim: int = 512,
        xattn: bool = False,
        output_tokens: bool = False,
        fusedlinear: bool = False,
        flash_attn: bool = False,
        batchsize: int = 4,
    ):
        self.parent = parent
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.head_width = head_width
        self.mlp_ratio = mlp_ratio
        self.ls_init_value = ls_init_value
        self.patch_dropout = patch_dropout
        self.global_average_pool = global_average_pool
        self.attentional_pool = attentional_pool
        self.n_queries = n_queries
        self.attn_pooler_heads = attn_pooler_heads
        self.embed_dim = embed_dim
        self.output_dim = embed_dim
        self.xattn = xattn
        self.output_tokens = output_tokens
        self.fusedlinear = fusedlinear
        self.flash_attn = flash_attn
        self.batch_size = batchsize

    def prepare_config_and_inputs(self):
        image = floats_tensor([self.batch_size, 3, self.image_size, self.image_size])
        config = self.get_config()

        return config, image

    def get_config(self):
        return VisionTransformerConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            width=self.width,
            layers=self.layers,
            head_width=self.head_width,
            mlp_ratio=self.mlp_ratio,
            ls_init_value=self.ls_init_value,
            patch_dropout=self.patch_dropout,
            global_average_pool=self.global_average_pool,
            attentional_pool=self.attentional_pool,
            n_queries=self.n_queries,
            attn_pooler_heads=self.attn_pooler_heads,
            output_dim=self.output_dim,
            xattn=self.xattn,
            output_tokens=self.output_tokens,
            fusedlinear=self.fusedlinear,
            flash_attn=self.flash_attn,
        )

    def create_and_check_model(self, config, image):
        model = VisionTransformer(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(image)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        self.parent.assertEqual(
            result.shape,
            [self.batch_size, self.output_dim],
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, image = config_and_inputs
        inputs_dict = {"x": image}
        return config, inputs_dict


class VisionTransformerModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (VisionTransformer,)
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = VisionTransformerModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=VisionTransformerConfig,
            image_size=224,
            patch_size=14,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="CLIP's vision encoder does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="CLIP's text encoder does not use inputs_embeds and output_embeds")
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

            expected_arg_names = ["x"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)


class TextTransformerModelTester:
    def __init__(
        self,
        parent,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        ls_init_value: float = None,
        embed_dim: int = 512,
        xattn: bool = False,
        attn_mask: bool = True,
        pad_id: int = 0,
        embed_cls: bool = False,
        output_tokens: bool = False,
        quick_gelu: bool = False,
        mlp_ratio: float = 4.0,
        fusedLN: bool = False,
        fusedlinear: bool = False,
        flash_attn: bool = False,
        batchsize: int = 4,
    ):
        self.parent = parent
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.heads = heads
        self.layers = layers
        self.ls_init_value = ls_init_value
        self.embed_dim = embed_dim
        self.xattn = xattn
        self.attn_mask = attn_mask
        self.pad_id = pad_id
        self.embed_cls = embed_cls
        self.output_tokens = output_tokens
        self.quick_gelu = quick_gelu
        self.mlp_ratio = mlp_ratio
        self.fusedLN = fusedLN
        self.fusedlinear = fusedlinear
        self.flash_attn = flash_attn
        self.batch_size = batchsize

    def prepare_config_and_inputs(self):
        text = ids_tensor([self.batch_size, self.context_length], self.vocab_size)
        config = self.get_config()

        return config, text

    def get_config(self):
        return TextTransformerConfig(
            context_length=self.context_length,
            vocab_size=self.vocab_size,
            width=self.width,
            heads=self.heads,
            layers=self.layers,
            ls_init_value=self.ls_init_value,
            embed_dim=self.embed_dim,
            xattn=self.xattn,
            attn_mask=self.attn_mask,
            pad_id=self.pad_id,
            embed_cls=self.embed_cls,
            output_tokens=self.output_tokens,
            quick_gelu=self.quick_gelu,
            mlp_ratio=self.mlp_ratio,
            fusedLN=self.fusedLN,
            fusedlinear=self.fusedlinear,
            flash_attn=self.flash_attn,
        )

    def create_and_check_model(self, config, text):
        model = TextTransformer(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(text)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        self.parent.assertEqual(
            result.shape,
            [self.batch_size, self.embed_dim],
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, text = config_and_inputs
        inputs_dict = {"text": text}
        return config, inputs_dict


class TextTransformerModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (TextTransformer,)
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = TextTransformerModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=TextTransformerConfig,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="CLIP's text encoder does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="CLIP's text encoder does not use inputs_embeds and output_embeds")
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

            expected_arg_names = ["text"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)


class CLIPModelTester:
    def __init__(
        self,
        parent,
        vision_cfg={
            "image_size": 224,
            "patch_size": 14,
            "width": 768,
            "layers": 12,
            "head_width": 64,
            "mlp_ratio": 4.0,
            "ls_init_value": None,
            "patch_dropout": 0.0,
            "global_average_pool": False,
            "attentional_pool": False,
            "n_queries": 256,
            "attn_pooler_heads": 8,
            "embed_dim": 512,
            "xattn": False,
            "output_tokens": False,
            "fusedlinear": False,
            "flash_attn": False,
            "batchsize": 4,
        },
        text_cfg={
            "context_length": 77,
            "vocab_size": 49408,
            "width": 512,
            "heads": 8,
            "layers": 12,
            "ls_init_value": None,
            "embed_dim": 512,
            "xattn": False,
            "attn_mask": True,
            "pad_id": 0,
            "embed_cls": False,
            "output_tokens": False,
            "quick_gelu": False,
            "mlp_ratio": 4.0,
            "fusedLN": False,
            "fusedlinear": False,
            "flash_attn": False,
            "batchsize": 4,
        },
        batch_size=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.vision_cfg = vision_cfg
        self.text_cfg = text_cfg

    def prepare_config_and_inputs(self):
        image = floats_tensor([self.batch_size, 3, self.vision_cfg["image_size"], self.vision_cfg["image_size"]])
        text = ids_tensor([self.batch_size, self.text_cfg["context_length"]], self.text_cfg["vocab_size"])
        config = self.get_config()

        return config, image, text

    def get_config(self):
        return CLIPConfig(
            vision_cfg=self.vision_cfg,
            text_cfg=self.text_cfg,
        )

    def create_and_check_model(self, config, image, text):
        model = CLIP(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(image, text, skiploss=True)

        self.parent.assertEqual(
            result[0].shape,
            [self.batch_size, self.vision_cfg["embed_dim"]],
        )
        self.parent.assertEqual(
            result[1].shape,
            [self.batch_size, self.text_cfg["embed_dim"]],
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, image, text = config_and_inputs
        inputs_dict = {"image": image, "input_ids": text}
        return config, inputs_dict


class CLIPModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (CLIP,)
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = CLIPModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=CLIPConfig,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="CLIP's vision encoder does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="CLIP's text encoder does not use inputs_embeds and output_embeds")
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
        for model_name in CLIP_PRETRAINED_MODEL_ARCHIVE_LIST:
            model = CLIP.from_pretrained(model_name)
            self.assertIsNotNone(model)
