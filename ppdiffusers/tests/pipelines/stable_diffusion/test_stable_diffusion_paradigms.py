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
    DDIMParallelScheduler,
    DDPMParallelScheduler,
    StableDiffusionParadigmsPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import slow
from ppdiffusers.utils.testing_utils import enable_full_determinism, require_paddle_gpu

from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..test_pipelines_common import PipelineLatentTesterMixin, PipelineTesterMixin

enable_full_determinism()


class StableDiffusionParadigmsPipelineFastTests(PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionParadigmsPipeline
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
            attention_head_dim=(2, 4),
            use_linear_projection=True,
        )
        scheduler = DDIMParallelScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        paddle.seed(seed=0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
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
            hidden_act="gelu",
            projection_dim=512,
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
            "prompt": "a photograph of an astronaut riding a horse",
            "generator": generator,
            "num_inference_steps": 10,
            "guidance_scale": 6.0,
            "output_type": "np",
            "parallel": 3,
            "debug": True,
        }
        return inputs

    def test_stable_diffusion_paradigms_default_case(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionParadigmsPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.2891, 0.3575, 0.3609, 0.1002, 0.2995, 0.408, 0.1136, 0.1684, 0.3375])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_paradigms_default_case_ddpm(self):
        components = self.get_dummy_components()
        paddle.seed(seed=0)
        components["scheduler"] = DDPMParallelScheduler()
        paddle.seed(seed=0)
        sd_pipe = StableDiffusionParadigmsPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.5261, 0.5763, 0.3414, 0.2257, 0.5636, 0.2942, 0.126, 0.2141, 0.2849])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_inference_batch_consistent(self):
        super().test_inference_batch_consistent(batch_sizes=[1, 2])

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(batch_size=2)

    def test_stable_diffusion_paradigms_negative_prompt(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionParadigmsPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        negative_prompt = "french fries"
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)
        image = output.images
        image_slice = image[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.2559, 0.3297, 0.3684, 0.1491, 0.3172, 0.4713, 0.1574, 0.189, 0.384])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01


@slow
@require_paddle_gpu
class StableDiffusionParadigmsPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "generator": generator,
            "num_inference_steps": 10,
            "guidance_scale": 7.5,
            "output_type": "np",
            "parallel": 3,
            "debug": True,
        }
        return inputs

    def test_stable_diffusion_paradigms_default(self):
        model_ckpt = "stabilityai/stable-diffusion-2-base"
        scheduler = DDIMParallelScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
        pipe = StableDiffusionParadigmsPipeline.from_pretrained(model_ckpt, scheduler=scheduler, safety_checker=None)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[(0), -3:, -3:, (-1)].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.9622, 0.9602, 0.9748, 0.9591, 0.963, 0.9691, 0.9661, 0.9631, 0.9741])
        assert np.abs(expected_slice - image_slice).max() < 0.01
