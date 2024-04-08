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

import inspect
import os
import tempfile
import unittest
import unittest.mock as mock
from typing import Dict, List, Tuple

import numpy as np
import paddle
from requests.exceptions import HTTPError

from ppdiffusers.models import UNet2DConditionModel
from ppdiffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_5,
    XFormersAttnProcessor,
)
from ppdiffusers.training_utils import EMAModel
from ppdiffusers.utils import logging
from ppdiffusers.utils.testing_utils import (
    CaptureLogger,
    paddle_device,
    require_paddle_gpu,
)


class ModelUtilsTest(unittest.TestCase):
    def tearDown(self):
        super().tearDown()

    # def test_accelerate_loading_error_message(self):
    #     with self.assertRaises(ValueError) as error_context:
    #         UNet2DConditionModel.from_pretrained("hf-internal-testing/stable-diffusion-broken", subfolder="unet")

    #     # make sure that error message states what keys are missing
    #     assert "conv_out.bias" in str(error_context.exception)

    def test_cached_files_are_used_when_no_internet(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}

        # Download this model to make sure it's in the cache.
        orig_model = UNet2DConditionModel.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="unet"
        )

        # Under the mock environment we get a 500 error when trying to reach the model.
        with mock.patch("requests.request", return_value=response_mock):
            # Download this model to make sure it's in the cache.
            model = UNet2DConditionModel.from_pretrained(
                "hf-internal-testing/tiny-stable-diffusion-torch", subfolder="unet", local_files_only=True
            )

        for p1, p2 in zip(orig_model.parameters(), model.parameters()):
            if (p1 != p2).cast("int64").sum() > 0:
                assert False, "Parameters not the same!"


class UNetTesterMixin:
    def test_forward_signature(self):
        init_dict, _ = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        signature = inspect.signature(model.forward)
        # signature.parameters is an OrderedDict => so arg_names order is deterministic
        arg_names = [*signature.parameters.keys()]

        expected_arg_names = ["sample", "timestep"]
        self.assertListEqual(arg_names[:2], expected_arg_names)

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


class ModelTesterMixin:
    main_input_name = None  # overwrite in model specific tester class
    base_precision = 1e-3
    forward_requires_fresh_args = False

    def test_from_save_pretrained(self, expected_max_diff=1e-1):
        if self.forward_requires_fresh_args:
            model = self.model_class(**self.init_dict)
        else:
            init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            model = self.model_class(**init_dict)

        if hasattr(model, "set_default_attn_processor"):
            model.set_default_attn_processor()
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, safe_serialization=False)
            new_model = self.model_class.from_pretrained(tmpdirname)
            if hasattr(new_model, "set_default_attn_processor"):
                new_model.set_default_attn_processor()

        with paddle.no_grad():
            if self.forward_requires_fresh_args:
                image = model(**self.inputs_dict(0))
            else:
                image = model(**inputs_dict)

            if isinstance(image, dict):
                image = image.to_tuple()[0]

            if self.forward_requires_fresh_args:
                new_image = new_model(**self.inputs_dict(0))
            else:
                new_image = new_model(**inputs_dict)

            if isinstance(new_image, dict):
                new_image = new_image.to_tuple()[0]

        max_diff = (image - new_image).abs().max().item()
        self.assertLessEqual(max_diff, expected_max_diff, "Models give different forward passes")

    def test_getattr_is_correct(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)

        # save some things to test
        model.dummy_attribute = 5
        model.register_to_config(test_attribute=5)

        logger = logging.get_logger("ppdiffusers.models.modeling_utils")
        # 30 for warning
        logger.setLevel(30)
        with CaptureLogger(logger) as cap_logger:
            assert hasattr(model, "dummy_attribute")
            assert getattr(model, "dummy_attribute") == 5
            assert model.dummy_attribute == 5

        # no warning should be thrown
        assert cap_logger.out == ""

        logger = logging.get_logger("ppdiffusers.models.modeling_utils")
        # 30 for warning
        logger.setLevel(30)
        with CaptureLogger(logger) as cap_logger:
            assert hasattr(model, "save_pretrained")
            fn = model.save_pretrained
            fn_1 = getattr(model, "save_pretrained")

            assert fn == fn_1
        # no warning should be thrown
        assert cap_logger.out == ""

        # warning should be thrown
        with self.assertWarns(FutureWarning):
            assert model.test_attribute == 5

        with self.assertWarns(FutureWarning):
            assert getattr(model, "test_attribute") == 5

        with self.assertRaises(AttributeError) as error:
            model.does_not_exist

        assert str(error.exception) == f"'{type(model).__name__}' object has no attribute 'does_not_exist'"

    @require_paddle_gpu
    def test_set_attn_processor_for_determinism(self):
        os.environ["FLAGS_cudnn_deterministic"] = "False"
        os.environ["FLAGS_cpu_deterministic"] = "False"
        if self.forward_requires_fresh_args:
            model = self.model_class(**self.init_dict)
        else:
            init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            model = self.model_class(**init_dict)

        if not hasattr(model, "set_attn_processor"):
            # If not has `set_attn_processor`, skip test
            os.environ["FLAGS_cudnn_deterministic"] = "True"
            os.environ["FLAGS_cpu_deterministic"] = "True"
            return

        assert all(type(proc) == AttnProcessor2_5 for proc in model.attn_processors.values())
        with paddle.no_grad():
            if self.forward_requires_fresh_args:
                output_1 = model(**self.inputs_dict(0))[0]
            else:
                output_1 = model(**inputs_dict)[0]

        model.set_default_attn_processor()
        assert all(type(proc) == AttnProcessor for proc in model.attn_processors.values())
        with paddle.no_grad():
            if self.forward_requires_fresh_args:
                output_2 = model(**self.inputs_dict(0))[0]
            else:
                output_2 = model(**inputs_dict)[0]

        model.set_attn_processor(AttnProcessor2_5())
        assert all(type(proc) == AttnProcessor2_5 for proc in model.attn_processors.values())
        with paddle.no_grad():
            if self.forward_requires_fresh_args:
                output_4 = model(**self.inputs_dict(0))[0]
            else:
                output_4 = model(**inputs_dict)[0]

        model.set_attn_processor(AttnProcessor())
        assert all(type(proc) == AttnProcessor for proc in model.attn_processors.values())
        with paddle.no_grad():
            if self.forward_requires_fresh_args:
                output_5 = model(**self.inputs_dict(0))[0]
            else:
                output_5 = model(**inputs_dict)[0]
            model.set_attn_processor(XFormersAttnProcessor())
            assert all(type(proc) == XFormersAttnProcessor for proc in model.attn_processors.values())
            with paddle.no_grad():
                if self.forward_requires_fresh_args:
                    output_6 = model(**self.inputs_dict(0))[0]
                else:
                    output_6 = model(**inputs_dict)[0]
            os.environ["FLAGS_cudnn_deterministic"] = "True"
            os.environ["FLAGS_cpu_deterministic"] = "True"

        # make sure that outputs match
        assert paddle.allclose(output_2, output_1, atol=self.base_precision).item()
        assert paddle.allclose(output_2, output_4, atol=self.base_precision).item()
        assert paddle.allclose(output_2, output_5, atol=self.base_precision).item()
        assert paddle.allclose(x=output_2, y=output_6, atol=self.base_precision).item()

    def test_from_save_pretrained_variant(self, expected_max_diff=5e-5):
        if self.forward_requires_fresh_args:
            model = self.model_class(**self.init_dict)
        else:
            init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            model = self.model_class(**init_dict)

        if hasattr(model, "set_default_attn_processor"):
            model.set_default_attn_processor()

        model.eval()

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, variant="fp16", safe_serialization=False)
            new_model = self.model_class.from_pretrained(tmpdirname, variant="fp16")
            if hasattr(new_model, "set_default_attn_processor"):
                new_model.set_default_attn_processor()

            # non-variant cannot be loaded
            # model = self.model_class.from_pretrained(tmpdirname)

            # self.assertIsNotNone(model)

        with paddle.no_grad():
            if self.forward_requires_fresh_args:
                image = model(**self.inputs_dict(0))
            else:
                image = model(**inputs_dict)
            if isinstance(image, dict):
                image = image.to_tuple()[0]

            if self.forward_requires_fresh_args:
                new_image = new_model(**self.inputs_dict(0))
            else:
                new_image = new_model(**inputs_dict)

            if isinstance(new_image, dict):
                new_image = new_image.to_tuple()[0]

        max_diff = (image - new_image).abs().max().item()
        self.assertLessEqual(max_diff, expected_max_diff, "Models give different forward passes")

    def test_from_save_pretrained_dtype(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        model.eval()

        for dtype in [paddle.float32, paddle.float16, paddle.bfloat16]:
            if paddle_device == "mps" and dtype == paddle.bfloat16:
                continue
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.to(dtype=dtype)
                model.save_pretrained(tmpdirname, safe_serialization=False)
                # new_model = self.model_class.from_pretrained(tmpdirname, low_cpu_mem_usage=True, paddle_dtype=dtype)
                # assert new_model.dtype == dtype
                new_model = self.model_class.from_pretrained(tmpdirname, low_cpu_mem_usage=False, paddle_dtype=dtype)
                assert new_model.dtype == dtype

    def test_determinism(self, expected_max_diff=5e-4):
        if self.forward_requires_fresh_args:
            model = self.model_class(**self.init_dict)
        else:
            init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            model = self.model_class(**init_dict)
        model.eval()

        with paddle.no_grad():
            if self.forward_requires_fresh_args:
                first = model(**self.inputs_dict(0))
            else:
                first = model(**inputs_dict)
            if isinstance(first, dict):
                first = first.to_tuple()[0]

            if self.forward_requires_fresh_args:
                second = model(**self.inputs_dict(0))
            else:
                second = model(**inputs_dict)
            if isinstance(second, dict):
                second = second.to_tuple()[0]

        out_1 = first.cpu().numpy()
        out_2 = second.cpu().numpy()
        out_1 = out_1[~np.isnan(out_1)]
        out_2 = out_2[~np.isnan(out_2)]
        max_diff = np.amax(np.abs(out_1 - out_2))
        self.assertLessEqual(max_diff, expected_max_diff)

    def test_output(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**init_dict)
        model.eval()

        with paddle.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.to_tuple()[0]

        self.assertIsNotNone(output)

        # input & output have to have the same shape
        input_tensor = inputs_dict[self.main_input_name]
        expected_shape = input_tensor.shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_model_from_pretrained(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        model.eval()

        # test if the model can be loaded from the config
        # and has all the expected shape
        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname, safe_serialization=False)
            new_model = self.model_class.from_pretrained(tmpdirname)
            new_model.eval()

        # check if all parameters shape are the same
        for param_name in model.state_dict().keys():
            param_1 = model.state_dict()[param_name]
            param_2 = new_model.state_dict()[param_name]
            self.assertEqual(param_1.shape, param_2.shape)

        with paddle.no_grad():
            output_1 = model(**inputs_dict)

            if isinstance(output_1, dict):
                output_1 = output_1.to_tuple()[0]

            output_2 = new_model(**inputs_dict)

            if isinstance(output_2, dict):
                output_2 = output_2.to_tuple()[0]

        self.assertEqual(output_1.shape, output_2.shape)

    def test_training(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        model.train()
        output = model(**inputs_dict)

        if isinstance(output, dict):
            output = output.to_tuple()[0]

        input_tensor = inputs_dict[self.main_input_name]
        noise = paddle.randn((input_tensor.shape[0],) + self.output_shape)
        loss = paddle.nn.functional.mse_loss(output, noise)
        loss.backward()

    def test_ema_training(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**init_dict)
        model.train()
        ema_model = EMAModel(model.parameters())

        output = model(**inputs_dict)

        if isinstance(output, dict):
            output = output.to_tuple()[0]

        input_tensor = inputs_dict[self.main_input_name]
        noise = paddle.randn((input_tensor.shape[0],) + self.output_shape)
        loss = paddle.nn.functional.mse_loss(output, noise)
        loss.backward()
        ema_model.step(model.parameters())

    def test_outputs_equivalence(self):
        def set_nan_tensor_to_zero(t):
            # Temporary fallback until `aten::_index_put_impl_` is implemented in mps
            # Track progress in https://github.com/pypaddle/pypaddle/issues/77764
            # t[t != t] = 0
            # return t.to(device)
            return t

        def recursive_check(tuple_object, dict_object):
            if isinstance(tuple_object, (List, Tuple)):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif isinstance(tuple_object, Dict):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object.values(), dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif tuple_object is None:
                return
            else:
                self.assertTrue(
                    paddle.allclose(
                        set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=5e-5
                    ),
                    msg=(
                        "Tuple and dict output are not equal. Difference:"
                        f" {paddle.max(paddle.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                        f" {paddle.isnan(tuple_object).any()} and `inf`: {paddle.isinf(tuple_object)}. Dict has"
                        f" `nan`: {paddle.isnan(dict_object).any()} and `inf`: {paddle.isinf(dict_object)}."
                    ),
                )

        if self.forward_requires_fresh_args:
            model = self.model_class(**self.init_dict)
        else:
            init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            model = self.model_class(**init_dict)

        model.eval()

        with paddle.no_grad():
            if self.forward_requires_fresh_args:
                outputs_dict = model(**self.inputs_dict(0))
                outputs_tuple = model(**self.inputs_dict(0), return_dict=False)
            else:
                outputs_dict = model(**inputs_dict)
                outputs_tuple = model(**inputs_dict, return_dict=False)

        recursive_check(outputs_tuple, outputs_dict)

    def test_enable_disable_gradient_checkpointing(self):
        if not self.model_class._supports_gradient_checkpointing:
            return  # Skip test if model does not support gradient checkpointing

        init_dict, _ = self.prepare_init_args_and_inputs_for_common()

        # at init model should have gradient checkpointing disabled
        model = self.model_class(**init_dict)
        self.assertFalse(model.is_gradient_checkpointing)

        # check enable works
        model.enable_gradient_checkpointing()
        self.assertTrue(model.is_gradient_checkpointing)

        # check disable works
        model.disable_gradient_checkpointing()
        self.assertFalse(model.is_gradient_checkpointing)

    def test_deprecated_kwargs(self):
        has_kwarg_in_model_class = "kwargs" in inspect.signature(self.model_class.__init__).parameters
        has_deprecated_kwarg = len(self.model_class._deprecated_kwargs) > 0

        if has_kwarg_in_model_class and not has_deprecated_kwarg:
            raise ValueError(
                f"{self.model_class} has `**kwargs` in its __init__ method but has not defined any deprecated kwargs"
                " under the `_deprecated_kwargs` class attribute. Make sure to either remove `**kwargs` if there are"
                " no deprecated arguments or add the deprecated argument with `_deprecated_kwargs ="
                " [<deprecated_argument>]`"
            )

        if not has_kwarg_in_model_class and has_deprecated_kwarg:
            raise ValueError(
                f"{self.model_class} doesn't have `**kwargs` in its __init__ method but has defined deprecated kwargs"
                " under the `_deprecated_kwargs` class attribute. Make sure to either add the `**kwargs` argument to"
                f" {self.model_class}.__init__ if there are deprecated arguments or remove the deprecated argument"
                " from `_deprecated_kwargs = [<deprecated_argument>]`"
            )
