# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import unittest

import numpy as np
import paddle
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker

from paddlemix.models.eva02.modeling_finetune import (
    EVA02VisionTransformer,
    EVA02VisionTransformerConfig,
)
from paddlemix.models.eva02.modeling_pretrain import (
    EVA02ForPretrain,
    EVA02ForPretrainConfig,
)
from tests.models.test_configuration_common import ConfigTester
from tests.models.test_modeling_common import ModelTesterMixin, floats_tensor
from tests.testing_utils import slow

EVA02_Finetune_MODEL_ARCHIVE_LIST = ["paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_ft_in1k_p14"]
EVA02_Pretrain_Teacher_MODEL_ARCHIVE_LIST = ["paddlemix/EVA/EVA01-CLIP-g-14"]
EVA02_Pretrain_Student_MODEL_ARCHIVE_LIST = ["paddlemix/EVA/EVA02/eva02_Ti_pt_in21k_p14"]

tracker = get_rng_state_tracker()
tracker.add("global_seed", 6666)
tracker.add("local_seed", 1025)


class EVA02ForFinetuneModelTester:
    def __init__(
        self,
        parent,
        image_size=336,
        patch_size=14,
        embed_dim=192,
        layers=12,
        num_heads=3,
        mlp_ratio=2.6667,
        in_chans=3,
        num_classes=1000,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_values=None,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        use_mean_pooling=True,
        init_scale=0.001,
        enable_recompute=False,
        stop_grad_conv1=True,  #
        postnorm=False,
        deepnorm=False,  #
        subln=True,
        xattn=False,  #
        swiglu=True,  #
        naiveswiglu=False,
        rope=True,
        pt_hw_seq_len=16,
        intp_freq=True,  #
        fusedLN=True,  #
        batchsize=4,
    ):
        self.parent = parent
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.layers = layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.init_values = init_values
        self.use_abs_pos_emb = use_abs_pos_emb
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_shared_rel_pos_bias = use_shared_rel_pos_bias
        self.use_mean_pooling = use_mean_pooling
        self.init_scale = init_scale
        self.enable_recompute = enable_recompute
        self.stop_grad_conv1 = stop_grad_conv1
        self.deepnorm = deepnorm
        self.postnorm = postnorm
        self.xattn = xattn
        self.intp_freq = intp_freq
        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu
        self.rope = rope
        self.pt_hw_seq_len = pt_hw_seq_len
        self.subln = subln
        self.fusedLN = fusedLN
        self.batch_size = batchsize

    def prepare_config_and_inputs(self):
        image = floats_tensor([self.batch_size, 3, self.image_size, self.image_size])
        config = self.get_config()
        return config, image

    def get_config(self):
        return EVA02VisionTransformerConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            layers=self.layers,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
            init_values=self.init_values,
            use_abs_pos_emb=self.use_abs_pos_emb,
            use_rel_pos_bias=self.use_rel_pos_bias,
            use_shared_rel_pos_bias=self.use_shared_rel_pos_bias,
            use_mean_pooling=self.use_mean_pooling,
            init_scale=self.init_scale,
            enable_recompute=self.enable_recompute,
            stop_grad_conv1=self.stop_grad_conv1,
            postnorm=self.postnorm,
            deepnorm=self.deepnorm,
            subln=self.subln,
            xattn=self.xattn,
            swiglu=self.swiglu,
            naiveswiglu=self.naiveswiglu,
            rope=self.rope,
            pt_hw_seq_len=self.pt_hw_seq_len,
            intp_freq=self.intp_freq,
            fusedLN=self.fusedLN,
        )

    def create_and_check_model(self, config, image):
        model = EVA02VisionTransformer(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(image)
        self.parent.assertEqual(
            result.shape,
            [self.batch_size, self.num_classes],
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, image = config_and_inputs
        inputs_dict = {"image": image}
        return config, inputs_dict


class EVA02ForFinetuneModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (EVA02VisionTransformer,)
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = EVA02ForFinetuneModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=EVA02VisionTransformerConfig,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="EVA02's vision encoder does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="EVA02 has no text encoder")
    def test_model_common_attributes(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["image"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for model_name in EVA02_Finetune_MODEL_ARCHIVE_LIST:
            model = EVA02VisionTransformer.from_pretrained(model_name)
            self.assertIsNotNone(model)


class EVA02ForPretrainModelTester:
    def __init__(
        self,
        parent,
        teacher_config={
            "embed_dim": 1024,
            "vision_cfg": {
                "width": 1408,
                "image_size": 224,
                "layers": 40,
                "head_width": 88,
                "patch_size": 14,
                "mlp_ratio": 4.3637,
                "eva_model_name": "eva-clip-g-14-x",
                "drop_path_rate": 0.0,
                "xattn": True,
                "fusedLN": True,
                "rope": False,
                "subln": False,
                "quick_gelu": False,
                "qkv_bias": True,
                "use_mean_pooling": False,
                "inner_attn_ln": False,
                "output_tokens": True,  #
                "token_feats": True,  #
            },
            "text_cfg": {
                "output_dim": 1024,
                "context_length": 77,
                "vocab_size": 49408,
                "width": 768,
                "heads": 12,
                "layers": 12,
                "xattn": False,
                "fusedLN": True,
            },
        },
        student_config={
            "image_size": 224,
            "depth": 12,
            "embed_dim": 192,
            "patch_size": 14,
            "num_heads": 3,
            "mlp_ratio": 2.6667,
            "drop_path_rate": 0.0,
            "xattn": False,
            "fusedLN": True,
            "rope": True,
            "swiglu": True,
            "naiveswiglu": False,
            "subln": True,
            "quick_gelu": False,
            "qkv_bias": True,
            "predict_feature_dim": 1024,
        },
        batchsize=4,
    ):
        self.parent = parent
        self.batch_size = batchsize
        self.teacher_config = teacher_config
        self.student_config = student_config

    def prepare_config_and_inputs(self):
        samples = floats_tensor(
            [self.batch_size, 3, self.student_config["image_size"], self.student_config["image_size"]]
        )
        image = floats_tensor(
            [self.batch_size, 3, self.student_config["image_size"], self.student_config["image_size"]]
        )
        bool_masked_pos = floats_tensor([self.batch_size, 256])
        config = self.get_config()
        return config, samples, image, bool_masked_pos

    def get_config(self):
        return EVA02ForPretrainConfig(
            teacher_config=self.teacher_config,
            student_config=self.student_config,
        )

    def create_and_check_model(self, config, samples, image, bool_masked_pos):
        model = EVA02ForPretrain(config=config)
        model.eval()
        with paddle.no_grad():
            result = model(samples, image, bool_masked_pos, get_feats=True)  # get_feats, not loss
        self.parent.assertEqual(result.shape[1], model.teacher.visual.output_dim)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, samples, image, bool_masked_pos = config_and_inputs
        inputs_dict = {"samples": samples, "image": image, "bool_masked_pos": bool_masked_pos}
        return config, inputs_dict


class EVA02ForPretrainModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (EVA02ForPretrain,)
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False
    use_test_model_name_list = False

    def setUp(self):
        self.model_tester = EVA02ForPretrainModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=EVA02ForPretrainConfig,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="EVA02's vision encoder does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="EVA02 has no text encoder")
    def test_model_common_attributes(self):
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
            model = self._make_model_instance(config, model_class)
            model.eval()
            with paddle.no_grad():
                first = model(**self._prepare_for_class(inputs_dict, model_class))

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, save_function=paddle.save)
                model = model_class.from_pretrained(tmpdirname)
                model.eval()
                with paddle.no_grad():
                    second = model(**self._prepare_for_class(inputs_dict, model_class))

            # support tuple of tensor
            if isinstance(first, tuple) and isinstance(second, tuple):
                for tensor1, tensor2 in zip(first, second):
                    check_save_load(tensor1, tensor2)
            else:
                check_save_load(first, second)

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
                first = model(**self._prepare_for_class(inputs_dict, model_class))
                second = model(**self._prepare_for_class(inputs_dict, model_class))

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

            expected_arg_names = ["samples", "image", "bool_masked_pos"]
            self.assertListEqual(arg_names[:3], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @slow
    def test_model_from_pretrained(self):
        for tea_name, stu_name in zip(
            EVA02_Pretrain_Teacher_MODEL_ARCHIVE_LIST, EVA02_Pretrain_Student_MODEL_ARCHIVE_LIST
        ):
            model = EVA02ForPretrain.from_pretrained(
                pretrained_model_name_or_path=None,
                pretrained_teacher_name_or_path=tea_name,
                pretrained_student_name_or_path=stu_name,
            )
            self.assertIsNotNone(model)
