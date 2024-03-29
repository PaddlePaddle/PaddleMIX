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

import gc
import math
import unittest

import paddle

from ppdiffusers import UNet2DModel
from ppdiffusers.utils import logging
from ppdiffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    paddle_all_close,
    paddle_device,
    slow,
)

from .test_modeling_common import ModelTesterMixin, UNetTesterMixin

logger = logging.get_logger(__name__)

enable_full_determinism()


class Unet2DModelTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = UNet2DModel
    main_input_name = "sample"

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = 32, 32

        noise = floats_tensor((batch_size, num_channels) + sizes)
        time_step = paddle.to_tensor([10])

        return {"sample": noise, "timestep": time_step}

    @property
    def input_shape(self):
        return 3, 32, 32

    @property
    def output_shape(self):
        return 3, 32, 32

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": (32, 64),
            "down_block_types": ("DownBlock2D", "AttnDownBlock2D"),
            "up_block_types": ("AttnUpBlock2D", "UpBlock2D"),
            "attention_head_dim": 3,
            "out_channels": 3,
            "in_channels": 3,
            "layers_per_block": 2,
            "sample_size": 32,
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_mid_block_attn_groups(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["norm_num_groups"] = 16
        init_dict["add_attention"] = True
        init_dict["attn_norm_num_groups"] = 8

        model = self.model_class(**init_dict)
        model.eval()

        self.assertIsNotNone(
            model.mid_block.attentions[0].group_norm, "Mid block Attention group norm should exist but does not."
        )
        self.assertEqual(
            model.mid_block.attentions[0].group_norm._num_groups,
            init_dict["attn_norm_num_groups"],
            "Mid block Attention group norm does not have the expected number of groups.",
        )

        with paddle.no_grad():
            output = model(**inputs_dict)

            if isinstance(output, dict):
                output = output.to_tuple()[0]

        self.assertIsNotNone(output)
        expected_shape = inputs_dict["sample"].shape
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")


class UNetLDMModelTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = UNet2DModel
    main_input_name = "sample"

    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 4
        sizes = 32, 32

        noise = floats_tensor((batch_size, num_channels) + sizes)
        time_step = paddle.to_tensor([10])

        return {"sample": noise, "timestep": time_step}

    @property
    def input_shape(self):
        return 4, 32, 32

    @property
    def output_shape(self):
        return 4, 32, 32

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "sample_size": 32,
            "in_channels": 4,
            "out_channels": 4,
            "layers_per_block": 2,
            "block_out_channels": (32, 64),
            "attention_head_dim": 32,
            "down_block_types": ("DownBlock2D", "DownBlock2D"),
            "up_block_types": ("UpBlock2D", "UpBlock2D"),
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    def test_from_pretrained_hub(self):
        model, loading_info = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update", output_loading_info=True)

        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        image = model(**self.dummy_input).sample

        assert image is not None, "Make sure output is not None"

    @unittest.skipIf(paddle_device != "cuda", "This test is supposed to run on GPU")
    def test_from_pretrained_accelerate(self):
        model, _ = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update", output_loading_info=True)
        image = model(**self.dummy_input).sample

        assert image is not None, "Make sure output is not None"

    @unittest.skipIf(paddle_device != "cuda", "This test is supposed to run on GPU")
    def test_from_pretrained_accelerate_wont_change_results(self):
        # by defautl model loading will use accelerate as `low_cpu_mem_usage=True`
        model_accelerate, _ = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update", output_loading_info=True)
        # model_accelerate
        model_accelerate.eval()

        noise = paddle.randn(
            [
                1,
                model_accelerate.config.in_channels,
                model_accelerate.config.sample_size,
                model_accelerate.config.sample_size,
            ],
            generator=paddle.Generator().manual_seed(0),
        )
        time_step = paddle.to_tensor([10] * noise.shape[0])

        arr_accelerate = model_accelerate(noise, time_step)["sample"]

        # two models don't need to stay in the device at the same time
        del model_accelerate
        paddle.device.cuda.empty_cache()
        gc.collect()

        model_normal_load, _ = UNet2DModel.from_pretrained(
            "fusing/unet-ldm-dummy-update", output_loading_info=True, low_cpu_mem_usage=False
        )
        model_normal_load.eval()
        arr_normal_load = model_normal_load(noise, time_step)["sample"]

        assert paddle_all_close(arr_accelerate, arr_normal_load, rtol=1e-3)

    def test_output_pretrained(self):
        model = UNet2DModel.from_pretrained("fusing/unet-ldm-dummy-update")
        model.eval()

        noise = paddle.randn(
            shape=[1, model.config.in_channels, model.config.sample_size, model.config.sample_size],
            generator=paddle.Generator().manual_seed(0),
        )
        time_step = paddle.to_tensor([10] * noise.shape[0])

        with paddle.no_grad():
            output = model(noise, time_step).sample

        output_slice = output[0, -1, -3:, -3:].flatten().cpu()
        # fmt: off
        expected_output_slice = paddle.to_tensor(
            [
                0.43855608,
                -10.29346752,
                -9.60953522,
                -8.39902020,
                -16.29206276,
                -13.07511997,
                -9.30383205,
                -13.69859409,
                -10.52999401,
            ]
        )
        # fmt: on

        self.assertTrue(paddle_all_close(output_slice, expected_output_slice, rtol=1e-3))


class NCSNppModelTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = UNet2DModel
    main_input_name = "sample"

    @property
    def dummy_input(self, sizes=(32, 32)):
        batch_size = 4
        num_channels = 3

        noise = floats_tensor((batch_size, num_channels) + sizes)
        time_step = paddle.to_tensor(batch_size * [10]).cast("int32")

        return {"sample": noise, "timestep": time_step}

    @property
    def input_shape(self):
        return 3, 32, 32

    @property
    def output_shape(self):
        return 3, 32, 32

    def prepare_init_args_and_inputs_for_common(self):
        init_dict = {
            "block_out_channels": [32, 64, 64, 64],
            "in_channels": 3,
            "layers_per_block": 1,
            "out_channels": 3,
            "time_embedding_type": "fourier",
            "norm_eps": 1e-6,
            "mid_block_scale_factor": math.sqrt(2.0),
            "norm_num_groups": None,
            "down_block_types": [
                "SkipDownBlock2D",
                "AttnSkipDownBlock2D",
                "SkipDownBlock2D",
                "SkipDownBlock2D",
            ],
            "up_block_types": [
                "SkipUpBlock2D",
                "SkipUpBlock2D",
                "AttnSkipUpBlock2D",
                "SkipUpBlock2D",
            ],
        }
        inputs_dict = self.dummy_input
        return init_dict, inputs_dict

    @slow
    def test_from_pretrained_hub(self):
        model, loading_info = UNet2DModel.from_pretrained("google/ncsnpp-celebahq-256", output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertEqual(len(loading_info["missing_keys"]), 0)

        inputs = self.dummy_input
        noise = floats_tensor((4, 3) + (256, 256))
        inputs["sample"] = noise
        image = model(**inputs)

        assert image is not None, "Make sure output is not None"

    @slow
    def test_output_pretrained_ve_mid(self):
        model = UNet2DModel.from_pretrained("google/ncsnpp-celebahq-256")

        batch_size = 4
        num_channels = 3
        sizes = 256, 256

        noise = paddle.ones((batch_size, num_channels, *sizes))
        time_step = paddle.to_tensor(batch_size * [1e-4])

        with paddle.no_grad():
            output = model(noise, time_step).sample

        output_slice = output[0, -3:, -3:, -1].flatten().cpu()
        # fmt: off
        expected_output_slice = paddle.to_tensor([-4836.2178, -6487.1470, -3816.8196, -7964.9302, -10966.3037, -20043.5957, 8137.0513, 2340.3328, 544.6056])
        # fmt: on

        self.assertTrue(paddle_all_close(output_slice, expected_output_slice, rtol=1e-2))

    def test_output_pretrained_ve_large(self):
        model = UNet2DModel.from_pretrained("fusing/ncsnpp-ffhq-ve-dummy-update")

        batch_size = 4
        num_channels = 3
        sizes = 32, 32

        noise = paddle.ones((batch_size, num_channels, *sizes))
        time_step = paddle.to_tensor(batch_size * [1e-4])

        with paddle.no_grad():
            output = model(noise, time_step).sample

        output_slice = output[0, -3:, -3:, -1].flatten().cpu()
        # fmt: off
        expected_output_slice = paddle.to_tensor([-0.0325, -0.0900, -0.0869, -0.0332, -0.0725, -0.0270, -0.0101, 0.0227, 0.0256])
        # fmt: on

        self.assertTrue(paddle_all_close(output_slice, expected_output_slice, rtol=1e-2))

    def test_forward_with_norm_groups(self):
        # not required for this model
        pass
