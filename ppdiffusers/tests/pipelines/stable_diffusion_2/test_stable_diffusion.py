# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    logging,
)
from ppdiffusers.utils import nightly, slow
from ppdiffusers.utils.testing_utils import (
    CaptureLogger,
    enable_full_determinism,
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


class StableDiffusion2PipelineFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS

    def get_dummy_components(self):
        paddle.seed(0)
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
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        paddle.seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            sample_size=128,
        )
        paddle.seed(0)
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
        text_encoder = CLIPTextModel(text_encoder_config).eval()
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
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_ddim(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.3505131, 0.36318004, 0.39201266, 0.12107915, 0.27704653, 0.40363187, 0.09379572, 0.16225743, 0.36048344]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_pndm(self):
        components = self.get_dummy_components()
        components["scheduler"] = PNDMScheduler(skip_prk_steps=True)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.25144678, 0.35438284, 0.3613463, 0.11020249, 0.3101831, 0.42739886, 0.1142821, 0.17371863, 0.35148838]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_k_lms(self):
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.3676631, 0.38155898, 0.4023114, 0.11294425, 0.2891888, 0.40432304, 0.08882684, 0.1466648, 0.33633134]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_k_euler_ancestral(self):
        components = self.get_dummy_components()
        components["scheduler"] = EulerAncestralDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.36797395, 0.38137895, 0.40199342, 0.11330777, 0.2886864, 0.40422022, 0.08929691, 0.14658183, 0.3363046]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_k_euler(self):
        components = self.get_dummy_components()
        components["scheduler"] = EulerDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.36766386, 0.3815591, 0.40231153, 0.11294428, 0.28918856, 0.40432304, 0.08882678, 0.14666462, 0.3363313]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_unflawed(self):

        components = self.get_dummy_components()
        components["scheduler"] = DDIMScheduler.from_config(
            components["scheduler"].config, timestep_spacing="trailing"
        )
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        inputs["guidance_rescale"] = 0.7
        inputs["num_inference_steps"] = 10
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.2907, 0.3519, 0.3543, 0.1222, 0.3108, 0.4291, 0.1256, 0.1755, 0.3492])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_long_prompt(self):
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        do_classifier_free_guidance = True
        negative_prompt = None
        num_images_per_prompt = 1
        logger = logging.get_logger("ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion")
        prompt = 25 * "@"
        with CaptureLogger(logger) as cap_logger_3:
            text_embeddings_3 = sd_pipe._encode_prompt(
                prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
        prompt = 100 * "@"
        with CaptureLogger(logger) as cap_logger:
            text_embeddings = sd_pipe._encode_prompt(
                prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
        negative_prompt = "Hello"
        with CaptureLogger(logger) as cap_logger_2:
            text_embeddings_2 = sd_pipe._encode_prompt(
                prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
        assert text_embeddings_3.shape == text_embeddings_2.shape == text_embeddings.shape
        assert text_embeddings.shape[1] == 77
        assert cap_logger.out == cap_logger_2.out
        assert cap_logger.out.count("@") == 25
        assert cap_logger_3.out == ""

    def test_attention_slicing_forward_pass(self):
        super().test_attention_slicing_forward_pass()

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical()


@slow
@require_paddle_gpu
class StableDiffusion2PipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype="float32", seed=0):
        generator = paddle.Generator().manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = paddle.to_tensor(latents).cast(dtype)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_default_ddim(self):
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.95444, 0.94495, 0.95216, 0.9553, 0.96735, 0.96266, 0.93611, 0.93607, 0.92361])
        assert np.abs(image_slice - expected_slice).max() < 7e-3

    def test_stable_diffusion_pndm(self):
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.95444, 0.94495, 0.95216, 0.9553, 0.96735, 0.96266, 0.93611, 0.93606, 0.92361])
        assert np.abs(image_slice - expected_slice).max() < 7e-3

    def test_stable_diffusion_k_lms(self):
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.97127, 0.96023, 0.97548, 0.96638, 0.96024, 0.97627, 0.97062, 0.96507, 0.96851])
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    # def test_stable_diffusion_attention_slicing(self):
    #     pipe = StableDiffusionPipeline.from_pretrained(
    #         "stabilityai/stable-diffusion-2-base", paddle_dtype=paddle.float16
    #     )
    #     pipe.set_progress_bar_config(disable=None)

    #     pipe.enable_attention_slicing()
    #     inputs = self.get_inputs(dtype="float16")
    #     image_sliced = pipe(**inputs).images

    #     mem_bytes = paddle.device.cuda.memory_allocated()
    #     assert mem_bytes < 3.3 * 10**9

    #     pipe.disable_attention_slicing()
    #     inputs = self.get_inputs(dtype="float16")
    #     image = pipe(**inputs).images

    #     mem_bytes = paddle.device.cuda.memory_allocated()
    #     assert mem_bytes > 3.3 * 10**9
    #     assert np.abs(image_sliced - image).max() < 0.001

    def test_stable_diffusion_text2img_intermediate_state(self):
        number_of_steps = 0

        def callback_fn(step: int, timestep: int, latents: paddle.Tensor) -> None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([1.7967, 1.4192, 0.3793, 0.9932, -1.9344, 0.9437, -0.5885, -0.8557, -2.4879])
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                expected_slice = np.array([0.8416, 1.6526, 0.4555, 0.4148, -1.4008, 0.4431, -0.1944, -1.1958, -0.7446])
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05

        callback_fn.has_been_called = False
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-base", paddle_dtype=paddle.float16
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(dtype="float16")
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == inputs["num_inference_steps"]


@nightly
@require_paddle_gpu
class StableDiffusion2PipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, dtype="float32", seed=0):
        generator = paddle.Generator().manual_seed(seed)
        latents = np.random.RandomState(seed).standard_normal((1, 4, 64, 64))
        latents = paddle.to_tensor(latents).cast(dtype)
        inputs = {
            "prompt": "a photograph of an astronaut riding a horse",
            "latents": latents,
            "generator": generator,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_2_0_default_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base")
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        # expected_image = load_numpy(
        #     "https://bj.bcebos.com/v1/paddlenlp/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_2_text2img/stable_diffusion_2_0_base_ddim.npy"
        # )
        expected_image = np.array([0.9921, 0.9876, 0.9964, 0.9858, 0.9905, 0.9936, 0.9867, 0.9856, 1.0])
        image = image[-3:, -3:, -1].flatten()
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_stable_diffusion_2_1_default_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        # expected_image = load_numpy(
        #     "https://bj.bcebos.com/v1/paddlenlp/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_2_text2img/stable_diffusion_2_1_base_pndm.npy"
        # )
        expected_image = np.array([0.9154, 0.9076, 0.9186, 0.9141, 0.9097, 0.9284, 0.9232, 0.919, 0.9332])
        image = image[-3:, -3:, -1].flatten()
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.1

    def test_stable_diffusion_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        # expected_image = load_numpy(
        #     "https://bj.bcebos.com/v1/paddlenlp/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_2_text2img/stable_diffusion_2_1_base_ddim.npy"
        # )
        expected_image = np.array([0.9154, 0.9076, 0.9186, 0.9141, 0.9098, 0.9284, 0.9232, 0.919, 0.9332])
        image = image[-3:, -3:, -1].flatten()
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_stable_diffusion_lms(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        # expected_image = load_numpy(
        #     "https://bj.bcebos.com/v1/paddlenlp/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_2_text2img/stable_diffusion_2_1_base_lms.npy"
        # )
        expected_image = np.array([0.9843, 0.9761, 0.9742, 0.9726, 0.9763, 0.9807, 0.9681, 0.97, 0.9801])
        image = image[-3:, -3:, -1].flatten()
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_stable_diffusion_euler(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        # expected_image = load_numpy(
        #     "https://bj.bcebos.com/v1/paddlenlp/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_2_text2img/stable_diffusion_2_1_base_euler.npy"
        # )
        expected_image = np.array([0.9684, 0.9616, 0.9651, 0.9608, 0.9616, 0.9691, 0.962, 0.9574, 0.9681])
        image = image[-3:, -3:, -1].flatten()
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_stable_diffusion_dpm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 25
        image = sd_pipe(**inputs).images[0]
        # expected_image = load_numpy(
        #     "https://bj.bcebos.com/v1/paddlenlp/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_2_text2img/stable_diffusion_2_1_base_dpm_multi.npy"
        # )
        expected_image = np.array([0.9412, 0.9249, 0.9403, 0.9412, 0.932, 0.9462, 0.9422, 0.9346, 0.9368])
        image = image[-3:, -3:, -1].flatten()
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001
