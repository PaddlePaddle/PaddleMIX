# coding=utf-8
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 HuggingFace Inc.
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

import copy
import os
import tempfile
import unittest

import numpy as np
import paddle

from ppdiffusers import MotionAdapter, UNet2DConditionModel, UNetMotionModel
from ppdiffusers.utils import logging
from ppdiffusers.utils.testing_utils import enable_full_determinism, floats_tensor

from .test_modeling_common import ModelTesterMixin, UNetTesterMixin

logger = logging.get_logger(__name__)

enable_full_determinism()


class UNetMotionModelTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = UNetMotionModel
    main_input_name = "sample"

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        num_frames = 8
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels, num_frames) + sizes)
        time_step = paddle.to_tensor([10])
        encoder_hidden_states = floats_tensor((batch_size, 4, 32))

        return {"sample": noise, "timestep": time_step, "encoder_hidden_states": encoder_hidden_states}

    @property
    def input_shape(self):
        return (4, 8, 32, 32)

    @property
    def output_shape(self):
        return (4, 8, 32, 32)

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": (32, 64),
            "down_block_types": ("CrossAttnDownBlockMotion", "DownBlockMotion"),
            "up_block_types": ("UpBlockMotion", "CrossAttnUpBlockMotion"),
            "cross_attention_dim": 32,
            "num_attention_heads": 4,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 1,
            "sample_size": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_from_unet2d(self):
        paddle.seed(0)
        unet2d = UNet2DConditionModel()

        paddle.seed(1)
        model = self.model_class.from_unet2d(unet2d)
        model_state_dict = model.state_dict()

        for param_name, param_value in unet2d.named_parameters():
            self.assertTrue(paddle.equal_all(model_state_dict[param_name], param_value))

    def test_freeze_unet2d(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.freeze_unet2d_params()

        for param_name, param_value in model.named_parameters():
            if "motion_modules" not in param_name:
                self.assertFalse(not param_value.stop_gradient)

            else:
                self.assertTrue(not param_value.stop_gradient)

    def test_loading_motion_adapter(self):
        model = self.model_class()
        adapter = MotionAdapter()
        model.load_motion_modules(adapter)

        for idx, down_block in enumerate(model.down_blocks):
            adapter_state_dict = adapter.down_blocks[idx].motion_modules.state_dict()
            for param_name, param_value in down_block.motion_modules.named_parameters():
                self.assertTrue(paddle.equal_all(adapter_state_dict[param_name], param_value))

        for idx, up_block in enumerate(model.up_blocks):
            adapter_state_dict = adapter.up_blocks[idx].motion_modules.state_dict()
            for param_name, param_value in up_block.motion_modules.named_parameters():
                self.assertTrue(paddle.equal_all(adapter_state_dict[param_name], param_value))

        mid_block_adapter_state_dict = adapter.mid_block.motion_modules.state_dict()
        for param_name, param_value in model.mid_block.motion_modules.named_parameters():
            self.assertTrue(paddle.equal_all(mid_block_adapter_state_dict[param_name], param_value))

    def test_saving_motion_modules(self):
        paddle.seed(0)
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_motion_modules(tmpdirname)
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "diffusion_paddle_model.safetensors")))

            adapter_loaded = MotionAdapter.from_pretrained(tmpdirname)

            paddle.seed(0)
            model_loaded = self.model_class(**init_dict)
            model_loaded.load_motion_modules(adapter_loaded)

        with paddle.no_grad():
            output = model(**inputs_dict)[0]
            output_loaded = model_loaded(**inputs_dict)[0]

        max_diff = (output - output_loaded).abs().max().item()
        self.assertLessEqual(max_diff, 1e-4, "Models give different forward passes")

    # @unittest.skipIf(
    #     paddle_device != "cuda" or not is_ppxformers_available(),
    #     reason="XFormers attention is only available with CUDA and `xformers` installed",
    # )
    # def test_xformers_enable_works(self):
    #     init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
    #     model = self.model_class(**init_dict)

    #     model.enable_xformers_memory_efficient_attention()

    #     assert (
    #         model.mid_block.attentions[0].transformer_blocks[0].attn1.processor.__class__.__name__
    #         == "XFormersAttnProcessor"
    #     ), "xformers is not enabled"

    def test_gradient_checkpointing_is_applied(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model_class_copy = copy.copy(self.model_class)

        modules_with_gc_enabled = {}

        # now monkey patch the following function:
        #     def _set_gradient_checkpointing(self, module, value=False):
        #         if hasattr(module, "gradient_checkpointing"):
        #             module.gradient_checkpointing = value

        def _set_gradient_checkpointing_new(self, module, value=False):
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = value
                modules_with_gc_enabled[module.__class__.__name__] = True

        model_class_copy._set_gradient_checkpointing = _set_gradient_checkpointing_new

        model = model_class_copy(**init_dict)
        model.enable_gradient_checkpointing()

        EXPECTED_SET = {
            "CrossAttnUpBlockMotion",
            "CrossAttnDownBlockMotion",
            "UNetMidBlockCrossAttnMotion",
            "UpBlockMotion",
            "Transformer2DModel",
            "DownBlockMotion",
        }

        assert set(modules_with_gc_enabled.keys()) == EXPECTED_SET
        assert all(modules_with_gc_enabled.values()), "All modules should be enabled"

    def test_feed_forward_chunking(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        init_dict["norm_num_groups"] = 32

        model = self.model_class(**init_dict)
        model.eval()

        with paddle.no_grad():
            output = model(**inputs_dict)[0]

        model.enable_forward_chunking()
        with paddle.no_grad():
            output_2 = model(**inputs_dict)[0]

        self.assertEqual(output.shape, output_2.shape, "Shape doesn't match")
        assert np.abs(output.cpu().numpy() - output_2.cpu().numpy()).max() < 1e-2

    def test_pickle(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)

        with paddle.no_grad():
            sample = model(**inputs_dict).sample

        sample_copy = paddle.clone(sample)

        assert (sample - sample_copy).abs().max() < 1e-4

    def test_from_save_pretrained(self, expected_max_diff=5e-5):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        paddle.seed(0)
        model = self.model_class(**init_dict)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, safe_serialization=False)
            paddle.seed(0)
            new_model = self.model_class.from_pretrained(tmpdirname)

        with paddle.no_grad():
            image = model(**inputs_dict)
            if isinstance(image, dict):
                image = image.to_tuple()[0]

            new_image = new_model(**inputs_dict)

            if isinstance(new_image, dict):
                new_image = new_image.to_tuple()[0]

        max_diff = (image - new_image).abs().max().item()
        self.assertLessEqual(max_diff, expected_max_diff, "Models give different forward passes")

    def test_from_save_pretrained_variant(self, expected_max_diff=5e-5):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        paddle.seed(0)
        model = self.model_class(**init_dict)
        model.eval()
        if hasattr(model, "set_default_attn_processor"):
            model.set_default_attn_processor()

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, variant="fp16", safe_serialization=False)

            paddle.seed(0)
            new_model = self.model_class.from_pretrained(tmpdirname, variant="fp16")
            if hasattr(new_model, "set_default_attn_processor"):
                new_model.set_default_attn_processor()

            # TOFXI
            # non-variant cannot be loaded
            # with self.assertRaises(OSError) as error_context:
            #     self.model_class.from_pretrained(tmpdirname)
            # make sure that error message states what keys are missing
            # assert "Error no file named model_state.pdparams found in directory" in str(error_context.exception)
            # model = self.model_class.from_pretrained(tmpdirname)
            # self.assertIsNotNone(model)

        with paddle.no_grad():
            image = model(**inputs_dict)
            if isinstance(image, dict):
                image = image.to_tuple()[0]

            new_image = new_model(**inputs_dict)

            if isinstance(new_image, dict):
                new_image = new_image.to_tuple()[0]

        max_diff = (image - new_image).abs().max().item()
        self.assertLessEqual(max_diff, expected_max_diff, "Models give different forward passes")

    def test_forward_with_norm_groups(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["norm_num_groups"] = 16
        init_dict["block_out_channels"] = (16, 32)

        model = self.model_class(**init_dict)
        model.eval()

        with paddle.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.to_tuple()[0]

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")
