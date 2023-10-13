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

import gc
import unittest

import numpy as np
import paddle

from ppdiffusers import (
    CMStochasticIterativeScheduler,
    ConsistencyModelPipeline,
    UNet2DModel,
)
from ppdiffusers.utils import randn_tensor, slow
from ppdiffusers.utils.testing_utils import enable_full_determinism, require_paddle_gpu

from ..pipeline_params import (
    UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS,
    UNCONDITIONAL_IMAGE_GENERATION_PARAMS,
)
from ..test_pipelines_common import PipelineTesterMixin

enable_full_determinism()


class ConsistencyModelPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = ConsistencyModelPipeline
    params = UNCONDITIONAL_IMAGE_GENERATION_PARAMS
    batch_params = UNCONDITIONAL_IMAGE_GENERATION_BATCH_PARAMS
    required_optional_params = frozenset(
        ["num_inference_steps", "generator", "latents", "output_type", "return_dict", "callback", "callback_steps"]
    )

    @property
    def dummy_uncond_unet(self):
        unet = UNet2DModel.from_pretrained("diffusers/consistency-models-test", subfolder="test_unet")
        return unet

    @property
    def dummy_cond_unet(self):
        unet = UNet2DModel.from_pretrained("diffusers/consistency-models-test", subfolder="test_unet_class_cond")
        return unet

    def get_dummy_components(self, class_cond=False):
        if class_cond:
            unet = self.dummy_cond_unet
        else:
            unet = self.dummy_uncond_unet
        scheduler = CMStochasticIterativeScheduler(num_train_timesteps=40, sigma_min=0.002, sigma_max=80.0)
        components = {"unet": unet, "scheduler": scheduler}
        return components

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        inputs = {
            "batch_size": 1,
            "num_inference_steps": None,
            "timesteps": [22, 0],
            "generator": generator,
            "output_type": "np",
        }
        return inputs

    def test_consistency_model_pipeline_multistep(self):
        components = self.get_dummy_components()
        pipe = ConsistencyModelPipeline(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_slice = np.array(
            [0.50192136, 0.53004247, 0.45772523, 0.44602677, 0.6048462, 0.4337338, 0.6019111, 0.43378073, 0.4980215]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

    def test_consistency_model_pipeline_multistep_class_cond(self):
        components = self.get_dummy_components(class_cond=True)
        pipe = ConsistencyModelPipeline(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        inputs["class_labels"] = 0
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_slice = np.array(
            [0.50192136, 0.53004247, 0.45772523, 0.44602677, 0.6048462, 0.4337338, 0.6019111, 0.43378073, 0.4980215]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

    def test_consistency_model_pipeline_onestep(self):
        components = self.get_dummy_components()
        pipe = ConsistencyModelPipeline(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 1
        inputs["timesteps"] = None
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_slice = np.array(
            [0.49557337, 0.50040334, 0.50025475, 0.49875367, 0.4996505, 0.49925715, 0.49896982, 0.50362164, 0.5007533]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

    def test_consistency_model_pipeline_onestep_class_cond(self):
        components = self.get_dummy_components(class_cond=True)
        pipe = ConsistencyModelPipeline(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 1
        inputs["timesteps"] = None
        inputs["class_labels"] = 0
        image = pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_slice = np.array(
            [0.49557337, 0.50040334, 0.50025475, 0.49875367, 0.4996505, 0.49925715, 0.49896982, 0.50362164, 0.5007533]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001


@slow
@require_paddle_gpu
class ConsistencyModelPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, seed=0, get_fixed_latents=False, dtype="float32", shape=(1, 3, 64, 64)):
        generator = paddle.Generator().manual_seed(seed)
        inputs = {
            "num_inference_steps": None,
            "timesteps": [22, 0],
            "class_labels": 0,
            "generator": generator,
            "output_type": "np",
        }
        if get_fixed_latents:
            latents = self.get_fixed_latents(seed=seed, dtype=dtype, shape=shape)
            inputs["latents"] = latents
        return inputs

    def get_fixed_latents(self, seed=0, dtype="float32", shape=(1, 3, 64, 64)):
        generator = paddle.Generator().manual_seed(seed)
        latents = randn_tensor(shape, generator=generator, dtype=dtype)
        return latents

    def test_consistency_model_cd_multistep(self):
        unet = UNet2DModel.from_pretrained("diffusers/consistency_models", subfolder="diffusers_cd_imagenet64_l2")
        scheduler = CMStochasticIterativeScheduler(num_train_timesteps=40, sigma_min=0.002, sigma_max=80.0)
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        assert image.shape == (1, 64, 64, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_slice = np.array([0.2346, 0.2462, 0.2377, 0.2618, 0.2569, 0.2520, 0.2959, 0.2978, 0.2866])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.02

    def test_consistency_model_cd_onestep(self):
        unet = UNet2DModel.from_pretrained("diffusers/consistency_models", subfolder="diffusers_cd_imagenet64_l2")
        scheduler = CMStochasticIterativeScheduler(num_train_timesteps=40, sigma_min=0.002, sigma_max=80.0)
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 1
        inputs["timesteps"] = None
        image = pipe(**inputs).images
        assert image.shape == (1, 64, 64, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_slice = np.array([0.1659, 0.1947, 0.2273, 0.1677, 0.1203, 0.1242, 0.1858, 0.1336, 0.2093])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.02

    def test_consistency_model_cd_multistep_flash_attn(self):
        unet = UNet2DModel.from_pretrained("diffusers/consistency_models", subfolder="diffusers_cd_imagenet64_l2")
        scheduler = CMStochasticIterativeScheduler(num_train_timesteps=40, sigma_min=0.002, sigma_max=80.0)
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
        pipe.to(paddle_dtype="float16")
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs(get_fixed_latents=True)
        image = pipe(**inputs).images
        assert image.shape == (1, 64, 64, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_slice = np.array([0.0959, 0.2263, 0.3322, 0.0972, 0.0331, 0.0470, 0.1296, 0.0430, 0.2202])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

    def test_consistency_model_cd_onestep_flash_attn(self):
        unet = UNet2DModel.from_pretrained("diffusers/consistency_models", subfolder="diffusers_cd_imagenet64_l2")
        scheduler = CMStochasticIterativeScheduler(num_train_timesteps=40, sigma_min=0.002, sigma_max=80.0)
        pipe = ConsistencyModelPipeline(unet=unet, scheduler=scheduler)
        pipe.to(paddle_dtype="float16")
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs(get_fixed_latents=True)
        inputs["num_inference_steps"] = 1
        inputs["timesteps"] = None
        image = pipe(**inputs).images
        assert image.shape == (1, 64, 64, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_slice = np.array([0.1663, 0.1948, 0.2275, 0.168, 0.1204, 0.1245, 0.1858, 0.1338, 0.2095])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001
