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
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker

from paddlemix.models.clip.eva_clip_model import EVACLIP, EVACLIPConfig
from paddlemix.models.clip.vit_model import (
    EVAVisionTransformer,
    EVAVisionTransformerConfig,
)
from tests.models.test_configuration_common import ConfigTester
from tests.models.test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
)
from tests.testing_utils import slow

CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = ["paddlemix/EVA/EVA02-CLIP-L-14"]

tracker = get_rng_state_tracker()
tracker.add("global_seed", 6666)
tracker.add("local_seed", 1025)


class EVAVisionTransformerModelTester:
    def __init__(
        self,
        parent,
        image_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1000,
        width=768,
        layers=12,
        head_width: int = 64,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_values=None,
        patch_dropout=0.0,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        rope=False,
        use_mean_pooling=True,
        attentional_pool=False,
        n_queries=256,
        attn_pooler_heads=8,
        init_scale=0.001,
        enable_recompute=False,
        xattn=False,
        postnorm=False,
        pt_hw_seq_len=16,
        intp_freq=False,
        naiveswiglu=False,
        subln=False,
        output_tokens=False,
        token_feats=False,  # whether tokens send to self.head
        fusedLN=False,
        inner_attn_ln=True,  # False in eva-01 clip
        fusedlinear=False,
        flash_attn=False,
        batchsize=4,
    ):
        self.parent = parent
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.width = width
        self.layers = layers
        self.head_width = head_width
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.init_values = init_values
        self.patch_dropout = patch_dropout
        self.use_abs_pos_emb = use_abs_pos_emb
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_shared_rel_pos_bias = use_shared_rel_pos_bias
        self.rope = rope
        self.use_mean_pooling = use_mean_pooling
        self.attentional_pool = attentional_pool
        self.n_queries = n_queries
        self.attn_pooler_heads = attn_pooler_heads
        self.init_scale = init_scale
        self.enable_recompute = enable_recompute
        self.xattn = xattn
        self.postnorm = postnorm
        self.pt_hw_seq_len = pt_hw_seq_len
        self.intp_freq = intp_freq
        self.naiveswiglu = naiveswiglu
        self.subln = subln
        self.output_tokens = output_tokens
        self.token_feats = token_feats
        self.fusedLN = fusedLN
        self.inner_attn_ln = inner_attn_ln
        self.fusedlinear = fusedlinear
        self.flash_attn = flash_attn
        self.batch_size = batchsize

    def prepare_config_and_inputs(self):
        image = floats_tensor([self.batch_size, 3, self.image_size, self.image_size])
        config = self.get_config()

        return config, image

    def get_config(self):
        return EVAVisionTransformerConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            width=self.width,
            layers=self.layers,
            head_width=self.head_width,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
            init_values=self.init_values,
            patch_dropout=self.patch_dropout,
            use_abs_pos_emb=self.use_abs_pos_emb,
            use_rel_pos_bias=self.use_rel_pos_bias,
            use_shared_rel_pos_bias=self.use_shared_rel_pos_bias,
            rope=self.rope,
            use_mean_pooling=self.use_mean_pooling,
            attentional_pool=self.attentional_pool,
            n_queries=self.n_queries,
            attn_pooler_heads=self.attn_pooler_heads,
            init_scale=self.init_scale,
            enable_recompute=self.enable_recompute,
            xattn=self.xattn,
            postnorm=self.postnorm,
            pt_hw_seq_len=self.pt_hw_seq_len,
            intp_freq=self.intp_freq,
            naiveswiglu=self.naiveswiglu,
            subln=self.subln,
            output_tokens=self.output_tokens,
            token_feats=self.token_feats,
            fusedLN=self.fusedLN,
            inner_attn_ln=self.inner_attn_ln,
            fusedlinear=self.fusedlinear,
            flash_attn=self.flash_attn,
        )

    def create_and_check_model(self, config, image):
        model = EVAVisionTransformer(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(image)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        self.parent.assertEqual(
            result.shape,
            [self.batch_size, self.embed_dim],
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, image = config_and_inputs
        inputs_dict = {"x": image}
        return config, inputs_dict


class EVAVisionTransformerModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (EVAVisionTransformer,)
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = EVAVisionTransformerModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=EVAVisionTransformerConfig,
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


class EVACLIPModelTester:
    def __init__(
        self,
        parent,
        vision_cfg={
            "embed_dim": 512,
            "image_size": 224,
            "layers": 12,
            "width": 768,
            "head_width": 64,
            "patch_size": 16,
            "mlp_ratio": 2.6667,
            "eva_model_name": "eva-clip-b-16-X",
            "drop_path_rate": 0.0,
            "xattn": True,
            "fusedLN": True,
            "rope": True,
            "pt_hw_seq_len": 16,
            "intp_freq": True,
            "naiveswiglu": True,
            "subln": True,
            "quick_gelu": False,
            "qkv_bias": True,
            "use_mean_pooling": False,
        },
        text_cfg={
            "embed_dim": 512,
            "context_length": 77,
            "vocab_size": 49408,
            "width": 512,
            "heads": 8,
            "layers": 12,
            "xattn": True,
            "fusedLN": True,
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
        return EVACLIPConfig(
            vision_cfg=self.vision_cfg,
            text_cfg=self.text_cfg,
        )

    def create_and_check_model(self, config, image, text):
        model = EVACLIP(config=config)
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


class EVACLIPModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (EVACLIP,)
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = EVACLIPModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=EVACLIPConfig,
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
            model = EVACLIP.from_pretrained(model_name)
            self.assertIsNotNone(model)
