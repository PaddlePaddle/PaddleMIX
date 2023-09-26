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
from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionModelEditingPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import slow
from ppdiffusers.utils.testing_utils import (
    enable_full_determinism,
    paddle_device,
    require_paddle_gpu,
)

from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..test_pipelines_common import (
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)

enable_full_determinism()


class StableDiffusionModelEditingPipelineFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionModelEditingPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    def get_dummy_components(self):
        paddle.seed(seed=0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = DDIMScheduler()
        paddle.seed(seed=0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        paddle.seed(seed=0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        inputs = {
            "prompt": "A field of roses",
            "generator": generator,
            "height": None,
            "width": None,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_model_editing_default_case(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionModelEditingPipeline(**components)
        sd_pipe = sd_pipe.to(paddle_device)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.2471, 0.2747, 0.5817, 0.3581, 0.19, 0.4423, 0.3357, 0.1975, 0.4318])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_model_editing_negative_prompt(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionModelEditingPipeline(**components)
        sd_pipe = sd_pipe.to(paddle_device)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        negative_prompt = "french fries"
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.2218, 0.2555, 0.5607, 0.2025, 0.2799, 0.5633, 0.2034, 0.2858, 0.4895])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_model_editing_euler(self):
        components = self.get_dummy_components()
        components["scheduler"] = EulerAncestralDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"
        )
        sd_pipe = StableDiffusionModelEditingPipeline(**components)
        sd_pipe = sd_pipe.to(paddle_device)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.2786, 0.1732, 0.3241, 0.2671, 0.2245, 0.5156, 0.2579, 0.2105, 0.4317])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_model_editing_pndm(self):
        components = self.get_dummy_components()
        components["scheduler"] = PNDMScheduler()
        sd_pipe = StableDiffusionModelEditingPipeline(**components)
        sd_pipe = sd_pipe.to(paddle_device)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        with self.assertRaises(ValueError):
            _ = sd_pipe(**inputs).images

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical()

    def test_attention_slicing_forward_pass(self):
        super().test_attention_slicing_forward_pass()


@slow
@require_paddle_gpu
class StableDiffusionModelEditingSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        inputs = {
            "prompt": "A field of roses",
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_model_editing_default(self):
        model_ckpt = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionModelEditingPipeline.from_pretrained(model_ckpt, safety_checker=None)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[(0), -3:, -3:, (-1)].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([1.0, 1.0, 1.0, 0.9911, 1.0, 0.9674, 0.9825, 0.9558, 0.8864])
        assert np.abs(expected_slice - image_slice).max() < 0.01
        pipe.edit_model("A pack of roses", "A pack of blue roses")
        image = pipe(**inputs).images
        image_slice = image[(0), -3:, -3:, (-1)].flatten()
        assert image.shape == (1, 512, 512, 3)
        assert np.abs(expected_slice - image_slice).max() > 0.1

    # def test_stable_diffusion_model_editing_pipeline_with_sequential_cpu_offloading(self):
    #     paddle.device.cuda.empty_cache()
    #     model_ckpt = "CompVis/stable-diffusion-v1-4"
    #     scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
    #     pipe = StableDiffusionModelEditingPipeline.from_pretrained(
    #         model_ckpt, scheduler=scheduler, safety_checker=None
    #     )
    #     pipe.set_progress_bar_config(disable=None)
    #     pipe.enable_attention_slicing(1)
    #     pipe.enable_sequential_cpu_offload()
    #     inputs = self.get_inputs()
    #     _ = pipe(**inputs)
    #     mem_bytes = paddle.device.cuda.max_memory_allocated()
    #     assert mem_bytes < 4.4 * 10**9
