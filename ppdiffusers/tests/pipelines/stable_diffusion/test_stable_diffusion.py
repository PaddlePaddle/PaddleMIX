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

# import tempfile
# import time
import traceback
import unittest

import numpy as np
import paddle

from ppdiffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    logging,
)

# from ppdiffusers.models.attention_processor import AttnProcessor
from ppdiffusers.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers.utils.testing_utils import (  # require_torch_2,; run_test_in_subprocess,; paddle_device; load_image,; load_numpy,; numpy_cosine_similarity_distance,; require_python39_or_higher,
    CaptureLogger,
    enable_full_determinism,
    nightly,
    require_paddle_gpu,
    slow,
)

from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..test_pipelines_common import (
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)

enable_full_determinism()


# Will be run via run_test_in_subprocess
def _test_stable_diffusion_compile(in_queue, out_queue, timeout):
    error = None
    try:
        inputs = in_queue.get(timeout=timeout)
        paddle_device = inputs.pop("paddle_device")
        seed = inputs.pop("seed")
        inputs["generator"] = paddle.Generator(device=paddle_device).manual_seed(seed)

        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)

        # sd_pipe.unet.to(memory_format=torch.channels_last)
        # sd_pipe.unet = torch.compile(sd_pipe.unet, mode="reduce-overhead", fullgraph=True)

        sd_pipe.set_progress_bar_config(disable=None)

        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.38019, 0.28647, 0.27321, 0.40377, 0.38290, 0.35446, 0.39218, 0.38165, 0.42239])

        assert np.abs(image_slice - expected_slice).max() < 5e-3
    except Exception:
        error = f"{traceback.format_exc()}"

    results = {"error": error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()


class StableDiffusionPipelineFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = TEXT_TO_IMAGE_IMAGE_PARAMS
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS

    def get_dummy_components(self, time_cond_proj_dim=None):
        paddle.seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(4, 8),
            layers_per_block=1,
            sample_size=32,
            time_cond_proj_dim=time_cond_proj_dim,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
            norm_num_groups=2,
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
            block_out_channels=[4, 8],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
            norm_num_groups=2,
        )
        paddle.seed(0)
        text_encoder_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=64,
            layer_norm_eps=1e-05,
            num_attention_heads=8,
            num_hidden_layers=3,
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
            "image_encoder": None,
        }
        return components

    def get_dummy_inputs(self, device="gpu", seed=0):
        generator = paddle.Generator(device=device).manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_ddim(self):
        device = "gpu"  # ensure determinism for the device-dependent paddle.Generator

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        # expected_slice = np.array([0.3203, 0.4555, 0.4711, 0.3505, 0.3973, 0.4650, 0.5137, 0.3392, 0.4045])
        expected_slice = np.array(
            [
                0.28519553,
                0.23807192,
                0.38150552,
                0.21930423,
                0.26092762,
                0.51721215,
                0.25639117,
                0.25039536,
                0.47978917,
            ]
        )
        expected_slice = np.array(
            [1.0, 0.65051216, 0.4743263, 1.0, 0.5698221, 0.47715974, 0.6558254, 0.44712216, 0.40034303]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2

    def test_stable_diffusion_lcm(self):
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator

        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.3454, 0.5349, 0.5185, 0.2808, 0.4509, 0.4612, 0.4655, 0.3601, 0.4315])
        expected_slice = np.array(
            [
                [0.22372422, 0.2870139, 0.5641626],
                [0.5121601, 0.47953755, 0.52897346],
                [0.37784696, 0.58912075, 0.698726],
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice.flatten()).max() < 1e-2

    def test_stable_diffusion_lcm_custom_timesteps(self):
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator

        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        output = sd_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        # expected_slice = np.array([0.3454, 0.5349, 0.5185, 0.2808, 0.4509, 0.4612, 0.4655, 0.3601, 0.4315])
        expected_slice = np.array(
            [
                [0.22372428, 0.2870139, 0.5641626],
                [0.5121601, 0.47953752, 0.5289734],
                [0.37784693, 0.58912075, 0.69872594],
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice.flatten()).max() < 1e-2

    def test_stable_diffusion_prompt_embeds(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        inputs["prompt"] = 3 * [inputs["prompt"]]

        # forward
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        inputs = self.get_dummy_inputs()
        prompt = 3 * [inputs.pop("prompt")]

        text_inputs = sd_pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=sd_pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pd",
        )
        text_inputs = text_inputs["input_ids"]

        prompt_embeds = sd_pipe.text_encoder(text_inputs)[0]

        inputs["prompt_embeds"] = prompt_embeds

        # forward
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]

        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

    def test_stable_diffusion_negative_prompt_embeds(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        negative_prompt = 3 * ["this is a negative prompt"]
        inputs["negative_prompt"] = negative_prompt
        inputs["prompt"] = 3 * [inputs["prompt"]]

        # forward
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        inputs = self.get_dummy_inputs()
        prompt = 3 * [inputs.pop("prompt")]

        embeds = []
        for p in [prompt, negative_prompt]:
            text_inputs = sd_pipe.tokenizer(
                p,
                padding="max_length",
                max_length=sd_pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pd",
            )
            text_inputs = text_inputs["input_ids"]

            embeds.append(sd_pipe.text_encoder(text_inputs)[0])

        inputs["prompt_embeds"], inputs["negative_prompt_embeds"] = embeds

        # forward
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]

        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

    def test_stable_diffusion_prompt_embeds_with_plain_negative_prompt_list(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        negative_prompt = 3 * ["this is a negative prompt"]
        inputs["negative_prompt"] = negative_prompt
        inputs["prompt"] = 3 * [inputs["prompt"]]

        # forward
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[0, -3:, -3:, -1]

        inputs = self.get_dummy_inputs()
        inputs["negative_prompt"] = negative_prompt
        prompt = 3 * [inputs.pop("prompt")]

        text_inputs = sd_pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=sd_pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pd",
        )
        text_inputs = text_inputs["input_ids"]

        prompt_embeds = sd_pipe.text_encoder(text_inputs)[0]

        inputs["prompt_embeds"] = prompt_embeds

        # forward
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[0, -3:, -3:, -1]

        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 1e-4

    def test_stable_diffusion_ddim_factor_8(self):
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs, height=136, width=136)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 136, 136, 3)
        # expected_slice = np.array([0.4346, 0.5621, 0.5016, 0.3926, 0.4533, 0.4134, 0.5625, 0.5632, 0.5265])
        expected_slice = np.array(
            [0.39545745, 0.94682777, 0.6828775, 0.42496994, 0.49475053, 0.48353004, 0.27300328, 0.30724254, 0.50566095]
        )
        expected_slice = np.array(
            [[1.0, 0.63342917, 0.4865236], [0.9100755, 0.5501328, 0.49420765], [0.70722765, 0.51427716, 0.4552924]]
        )
        assert np.abs(image_slice.flatten() - expected_slice.flatten()).max() < 1e-2

    def test_stable_diffusion_pndm(self):
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.scheduler = PNDMScheduler(skip_prk_steps=True)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        # expected_slice = np.array([0.3411, 0.5032, 0.4704, 0.3135, 0.4323, 0.4740, 0.5150, 0.3498, 0.4022])
        expected_slice = np.array(
            [0.18620703, 0.24143961, 0.3609084, 0.21810293, 0.27230006, 0.51992655, 0.22248471, 0.2213102, 0.44538254]
        )
        expected_slice = np.array(
            [
                [0.9393308, 0.57120854, 0.4518213],
                [0.7595422, 0.51620936, 0.4211468],
                [0.5867668, 0.49642342, 0.40899342],
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice.flatten()).max() < 1e-2

    def test_stable_diffusion_no_safety_checker(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-lms-pipe", safety_checker=None
        )
        assert isinstance(pipe, StableDiffusionPipeline)
        assert isinstance(pipe.scheduler, LMSDiscreteScheduler)
        assert pipe.safety_checker is None

        image = pipe("example prompt", num_inference_steps=2).images[0]
        assert image is not None

        # # check that there's no error when saving a pipeline with one of the models being None
        # with tempfile.TemporaryDirectory() as tmpdirname:
        #     pipe.save_pretrained(tmpdirname)
        #     pipe = StableDiffusionPipeline.from_pretrained(tmpdirname)

        # # sanity check that the pipeline still works
        # assert pipe.safety_checker is None
        # image = pipe("example prompt", num_inference_steps=2).images[0]
        # assert image is not None

    def test_stable_diffusion_k_lms(self):
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        # expected_slice = np.array([0.3149, 0.5246, 0.4796, 0.3218, 0.4469, 0.4729, 0.5151, 0.3597, 0.3954])
        expected_slice = np.array(
            [
                0.29910105,
                0.22905633,
                0.37701294,
                0.21332851,
                0.26000416,
                0.52840894,
                0.25865072,
                0.25947532,
                0.47509664,
            ]
        )
        expected_slice = np.array(
            [[1.0, 0.65853244, 0.46585035], [1.0, 0.56990314, 0.47889802], [0.67565185, 0.44096634, 0.38964093]]
        )
        assert np.abs(image_slice.flatten() - expected_slice.flatten()).max() < 1e-2

    def test_stable_diffusion_k_euler_ancestral(self):
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(sd_pipe.scheduler.config)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        # expected_slice = np.array([0.3151, 0.5243, 0.4794, 0.3217, 0.4468, 0.4728, 0.5152, 0.3598, 0.3954])
        expected_slice = np.array(
            [0.29917336, 0.22854236, 0.37669897, 0.2137424, 0.25940597, 0.528258, 0.25919583, 0.2594489, 0.47522712]
        )
        expected_slice = np.array(
            [[1.0, 0.6584619, 0.4658452], [1.0, 0.5696779, 0.47879985], [0.6763958, 0.4408609, 0.38938487]]
        )
        assert np.abs(image_slice.flatten() - expected_slice.flatten()).max() < 1e-2

    def test_stable_diffusion_k_euler(self):
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator

        components = self.get_dummy_components()
        sd_pipe = StableDiffusionPipeline(**components)
        sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        # expected_slice = np.array([0.3149, 0.5246, 0.4796, 0.3218, 0.4469, 0.4729, 0.5151, 0.3597, 0.3954])
        expected_slice = np.array(
            [0.29910135, 0.22905621, 0.3770129, 0.21332836, 0.26000386, 0.52840906, 0.2586509, 0.2594754, 0.47509673]
        )
        expected_slice = np.array(
            [[1.0, 0.6585326, 0.46585047], [1.0, 0.56990314, 0.47889805], [0.675652, 0.4409661, 0.3896407]]
        )
        assert np.abs(image_slice.flatten() - expected_slice.flatten()).max() < 1e-2

    def test_stable_diffusion_vae_slicing(self):
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = StableDiffusionPipeline(**components)

        sd_pipe.set_progress_bar_config(disable=None)

        image_count = 4

        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * image_count
        output_1 = sd_pipe(**inputs)

        # make sure sliced vae decode yields the same result
        sd_pipe.enable_vae_slicing()
        inputs = self.get_dummy_inputs(device)
        inputs["prompt"] = [inputs["prompt"]] * image_count
        output_2 = sd_pipe(**inputs)

        # there is a small discrepancy at image borders vs. full batch decode
        assert np.abs(output_2.images.flatten() - output_1.images.flatten()).max() < 3e-3

    def test_stable_diffusion_vae_tiling(self):
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator
        components = self.get_dummy_components()

        # make sure here that pndm scheduler skips prk
        components["safety_checker"] = None
        sd_pipe = StableDiffusionPipeline(**components)

        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger"

        # Test that tiled decode at 512x512 yields the same result as the non-tiled decode
        generator = paddle.Generator(device=device).manual_seed(0)
        output_1 = sd_pipe([prompt], generator=generator, guidance_scale=6.0, num_inference_steps=2, output_type="np")

        # make sure tiled vae decode yields the same result
        sd_pipe.enable_vae_tiling()
        generator = paddle.Generator(device=device).manual_seed(0)
        output_2 = sd_pipe([prompt], generator=generator, guidance_scale=6.0, num_inference_steps=2, output_type="np")

        assert np.abs(output_2.images.flatten() - output_1.images.flatten()).max() < 6e-1

        # test that tiled decode works with various shapes
        shapes = [(1, 4, 73, 97), (1, 4, 97, 73), (1, 4, 49, 65), (1, 4, 65, 49)]
        for shape in shapes:
            zeros = paddle.zeros(shape).to(device)
            sd_pipe.vae.decode(zeros)

    def test_stable_diffusion_negative_prompt(self):
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator
        components = self.get_dummy_components()
        components["scheduler"] = PNDMScheduler(skip_prk_steps=True)
        sd_pipe = StableDiffusionPipeline(**components)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        negative_prompt = "french fries"
        output = sd_pipe(**inputs, negative_prompt=negative_prompt)

        image = output.images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        # expected_slice = np.array([0.3458, 0.5120, 0.4800, 0.3116, 0.4348, 0.4802, 0.5237, 0.3467, 0.3991])
        expected_slice = np.array(
            [0.16709289, 0.26912582, 0.35834038, 0.23045751, 0.30960953, 0.5324909, 0.20372942, 0.2368694, 0.43633103]
        )
        expected_slice = np.array(
            [
                [0.6781773, 0.59057236, 0.5428893],
                [0.6394354, 0.49193096, 0.4580638],
                [0.45883033, 0.43446106, 0.43641293],
            ]
        )
        assert np.abs(image_slice.flatten() - expected_slice.flatten()).max() < 1e-2

    def test_stable_diffusion_long_prompt(self):
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = StableDiffusionPipeline(**components)

        sd_pipe.set_progress_bar_config(disable=None)

        do_classifier_free_guidance = True
        negative_prompt = None
        num_images_per_prompt = 1
        logger = logging.get_logger("ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion")
        logger.setLevel(logging.WARNING)

        prompt = 100 * "@"
        with CaptureLogger(logger) as cap_logger:
            negative_text_embeddings, text_embeddings = sd_pipe.encode_prompt(
                prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
            if negative_text_embeddings is not None:
                text_embeddings = paddle.concat([negative_text_embeddings, text_embeddings])

        # 100 - 77 + 1 (BOS token) + 1 (EOS token) = 25
        assert cap_logger.out.count("@") == 25

        negative_prompt = "Hello"
        with CaptureLogger(logger) as cap_logger_2:
            negative_text_embeddings_2, text_embeddings_2 = sd_pipe.encode_prompt(
                prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
            if negative_text_embeddings_2 is not None:
                text_embeddings_2 = paddle.concat([negative_text_embeddings_2, text_embeddings_2])

        assert cap_logger.out == cap_logger_2.out

        prompt = 25 * "@"
        with CaptureLogger(logger) as cap_logger_3:
            negative_text_embeddings_3, text_embeddings_3 = sd_pipe.encode_prompt(
                prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )
            if negative_text_embeddings_3 is not None:
                text_embeddings_3 = paddle.concat([negative_text_embeddings_3, text_embeddings_3])

        assert text_embeddings_3.shape == text_embeddings_2.shape == text_embeddings.shape
        assert text_embeddings.shape[1] == 77
        assert cap_logger_3.out == ""

    def test_stable_diffusion_height_width_opt(self):
        components = self.get_dummy_components()
        components["scheduler"] = LMSDiscreteScheduler.from_config(components["scheduler"].config)
        sd_pipe = StableDiffusionPipeline(**components)

        sd_pipe.set_progress_bar_config(disable=None)

        prompt = "hey"

        output = sd_pipe(prompt, num_inference_steps=1, output_type="np")
        image_shape = output.images[0].shape[:2]
        assert image_shape == (64, 64)

        output = sd_pipe(prompt, num_inference_steps=1, height=96, width=96, output_type="np")
        image_shape = output.images[0].shape[:2]
        assert image_shape == (96, 96)

        config = dict(sd_pipe.unet.config)
        config["sample_size"] = 96
        sd_pipe.unet = UNet2DConditionModel.from_config(config)
        output = sd_pipe(prompt, num_inference_steps=1, output_type="np")
        image_shape = output.images[0].shape[:2]
        assert image_shape == (192, 192)

    def test_attention_slicing_forward_pass(self):
        super().test_attention_slicing_forward_pass(expected_max_diff=3e-3)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    # def test_freeu_enabled(self):
    #     components = self.get_dummy_components()
    #     sd_pipe = StableDiffusionPipeline(**components)

    #     sd_pipe.set_progress_bar_config(disable=None)

    #     prompt = "hey"
    #     output = sd_pipe(prompt, num_inference_steps=1, output_type="np", generator=paddle.seed(0)).images

    #     sd_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
    #     output_freeu = sd_pipe(prompt, num_inference_steps=1, output_type="np", generator=paddle.seed(0)).images

    #     assert not np.allclose(
    #         output[0, -3:, -3:, -1], output_freeu[0, -3:, -3:, -1]
    #     ), "Enabling of FreeU should lead to different results."

    # def test_freeu_disabled(self):
    #     components = self.get_dummy_components()
    #     sd_pipe = StableDiffusionPipeline(**components)

    #     sd_pipe.set_progress_bar_config(disable=None)

    #     prompt = "hey"
    #     output = sd_pipe(prompt, num_inference_steps=1, output_type="np", generator=paddle.seed(0)).images

    #     sd_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
    #     sd_pipe.disable_freeu()

    #     freeu_keys = {"s1", "s2", "b1", "b2"}
    #     for upsample_block in sd_pipe.unet.up_blocks:
    #         for key in freeu_keys:
    #             assert getattr(upsample_block, key) is None, f"Disabling of FreeU should have set {key} to None."

    #     output_no_freeu = sd_pipe(
    #         prompt, num_inference_steps=1, output_type="np", generator=paddle.seed(0)
    #     ).images

    #     assert np.allclose(
    #         output[0, -3:, -3:, -1], output_no_freeu[0, -3:, -3:, -1]
    #     ), "Disabling of FreeU should lead to results similar to the default pipeline results."


@slow
@require_paddle_gpu
class StableDiffusionPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, device="gpu", generator_device="cpu", dtype=paddle.float32, seed=0):
        generator = paddle.Generator(device=generator_device).manual_seed(seed)
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

    def test_stable_diffusion_1_1_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-1")

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        # expected_slice = np.array([0.4363, 0.4355, 0.3667, 0.4066, 0.3970, 0.3866, 0.4394, 0.4356, 0.4059])
        # expected_slice = np.array([0.813, 0.8131, 0.7874, 0.8392, 0.8151, 0.8054, 0.8292, 0.8232, 0.7889])
        expected_slice = np.array(
            [0.43625844, 0.43554005, 0.36670125, 0.40660906, 0.39703777, 0.38658676, 0.43935978, 0.4355765, 0.40592635]
        )
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    #    def test_stable_diffusion_v1_4_with_freeu(self):
    #        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    #        sd_pipe.set_progress_bar_config(disable=None)
    #
    #        inputs = self.get_inputs()
    #        inputs["num_inference_steps"] = 25
    #
    #        sd_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
    #        image = sd_pipe(**inputs).images
    #        image = image[0, -3:, -3:, -1].flatten()
    #        expected_image = [0.0721, 0.0588, 0.0268, 0.0384, 0.0636, 0.0, 0.0429, 0.0344, 0.0309]
    #        max_diff = np.abs(expected_image - image).max()
    #        assert max_diff < 1e-3

    def test_stable_diffusion_1_4_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        # expected_slice = np.array([0.5740, 0.4784, 0.3162, 0.6358, 0.5831, 0.5505, 0.5082, 0.5631, 0.5575])
        # expected_slice = np.array([0.8353, 0.821, 0.7806, 0.8376, 0.8179, 0.777, 0.7996, 0.7949, 0.7594])
        expected_slice = np.array(
            [0.57399195, 0.4783988, 0.31624407, 0.6358264, 0.58305466, 0.5505526, 0.5082418, 0.5630579, 0.5574736]
        )
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_stable_diffusion_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        # expected_slice = np.array([0.38019, 0.28647, 0.27321, 0.40377, 0.38290, 0.35446, 0.39218, 0.38165, 0.42239])
        # expected_slice = np.array([0.6155, 0.6341, 0.6258, 0.6384, 0.6364, 0.6122, 0.6449, 0.6366, 0.6709])
        expected_slice = np.array(
            [0.380184, 0.28647956, 0.2732175, 0.4037717, 0.38289914, 0.35445356, 0.39217913, 0.38164824, 0.42238885]
        )
        assert np.abs(image_slice - expected_slice).max() < 1e-4

    def test_stable_diffusion_lms(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        # expected_slice = np.array([0.10542, 0.09620, 0.07332, 0.09015, 0.09382, 0.07597, 0.08496, 0.07806, 0.06455])
        # expected_slice = np.array([0.7169, 0.7612, 0.7463, 0.752, 0.7436, 0.7547, 0.7585, 0.7377, 0.7653])
        expected_slice = np.array(
            [0.10543326, 0.09622014, 0.07333425, 0.09016445, 0.0938372, 0.07599375, 0.08498126, 0.07808119, 0.06457829]
        )
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    def test_stable_diffusion_dpm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", safety_checker=None)
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        # expected_slice = np.array([0.03503, 0.03494, 0.01087, 0.03128, 0.02552, 0.00803, 0.00742, 0.00372, 0.00000])
        # expected_slice = np.array([0.7243, 0.768, 0.7451, 0.7272, 0.7546, 0.7611, 0.7417, 0.7569, 0.7749])
        expected_slice = np.array(
            [0.03503478, 0.03494555, 0.01087055, 0.03128436, 0.02552631, 0.0080339, 0.00742412, 0.00371644, 0.0]
        )
        assert np.abs(image_slice - expected_slice).max() < 3e-3

    # def test_stable_diffusion_attention_slicing(self):
    #     torch.cuda.reset_peak_memory_stats()
    #     pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=paddle.float16)
    #     pipe.unet.set_default_attn_processor()
    #     pipe = pipe
    #     pipe.set_progress_bar_config(disable=None)

    #     # enable attention slicing
    #     pipe.enable_attention_slicing()
    #     inputs = self.get_inputs(paddle_device, dtype=paddle.float16)
    #     image_sliced = pipe(**inputs).images

    #     mem_bytes = torch.cuda.max_memory_allocated()
    #     torch.cuda.reset_peak_memory_stats()
    #     # make sure that less than 3.75 GB is allocated
    #     assert mem_bytes < 3.75 * 10**9

    #     # disable slicing
    #     pipe.disable_attention_slicing()
    #     pipe.unet.set_default_attn_processor()
    #     inputs = self.get_inputs(paddle_device, dtype=paddle.float16)
    #     image = pipe(**inputs).images

    #     # make sure that more than 3.75 GB is allocated
    #     mem_bytes = torch.cuda.max_memory_allocated()
    #     assert mem_bytes > 3.75 * 10**9
    #     max_diff = numpy_cosine_similarity_distance(image_sliced.flatten(), image.flatten())
    #     assert max_diff < 1e-3

    # def test_stable_diffusion_vae_slicing(self):
    #     torch.cuda.reset_peak_memory_stats()
    #     pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=paddle.float16)
    #     pipe = pipe
    #     pipe.set_progress_bar_config(disable=None)
    #     pipe.enable_attention_slicing()

    #     # enable vae slicing
    #     pipe.enable_vae_slicing()
    #     inputs = self.get_inputs(paddle_device, dtype=paddle.float16)
    #     inputs["prompt"] = [inputs["prompt"]] * 4
    #     inputs["latents"] = paddle.concat([inputs["latents"]] * 4)
    #     image_sliced = pipe(**inputs).images

    #     mem_bytes = torch.cuda.max_memory_allocated()
    #     torch.cuda.reset_peak_memory_stats()
    #     # make sure that less than 4 GB is allocated
    #     assert mem_bytes < 4e9

    #     # disable vae slicing
    #     pipe.disable_vae_slicing()
    #     inputs = self.get_inputs(paddle_device, dtype=paddle.float16)
    #     inputs["prompt"] = [inputs["prompt"]] * 4
    #     inputs["latents"] = paddle.concat([inputs["latents"]] * 4)
    #     image = pipe(**inputs).images

    #     # make sure that more than 4 GB is allocated
    #     mem_bytes = torch.cuda.max_memory_allocated()
    #     assert mem_bytes > 4e9
    #     # There is a small discrepancy at the image borders vs. a fully batched version.
    #     max_diff = numpy_cosine_similarity_distance(image_sliced.flatten(), image.flatten())
    #     assert max_diff < 1e-2

    # def test_stable_diffusion_vae_tiling(self):
    #     torch.cuda.reset_peak_memory_stats()
    #     model_id = "CompVis/stable-diffusion-v1-4"
    #     pipe = StableDiffusionPipeline.from_pretrained(
    #         model_id, revision="fp16", torch_dtype=paddle.float16, safety_checker=None
    #     )
    #     pipe.set_progress_bar_config(disable=None)
    #     pipe.enable_attention_slicing()
    #     pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
    #     pipe.vae = pipe.vae.to(memory_format=torch.channels_last)

    #     prompt = "a photograph of an astronaut riding a horse"

    #     # enable vae tiling
    #     pipe.enable_vae_tiling()
    #     pipe.enable_model_cpu_offload()
    #     generator = paddle.Generator(device="cpu").manual_seed(0)
    #     output_chunked = pipe(
    #         [prompt],
    #         width=1024,
    #         height=1024,
    #         generator=generator,
    #         guidance_scale=7.5,
    #         num_inference_steps=2,
    #         output_type="numpy",
    #     )
    #     image_chunked = output_chunked.images

    #     mem_bytes = torch.cuda.max_memory_allocated()

    #     # disable vae tiling
    #     pipe.disable_vae_tiling()
    #     generator = paddle.Generator(device="cpu").manual_seed(0)
    #     output = pipe(
    #         [prompt],
    #         width=1024,
    #         height=1024,
    #         generator=generator,
    #         guidance_scale=7.5,
    #         num_inference_steps=2,
    #         output_type="numpy",
    #     )
    #     image = output.images

    #     assert mem_bytes < 1e10
    #     max_diff = numpy_cosine_similarity_distance(image_chunked.flatten(), image.flatten())
    #     assert max_diff < 1e-2

    # def test_stable_diffusion_fp16_vs_autocast(self):
    #     # this test makes sure that the original model with autocast
    #     # and the new model with fp16 yield the same result
    #     pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=paddle.float16)
    #     pipe = pipe
    #     pipe.set_progress_bar_config(disable=None)

    #     inputs = self.get_inputs(paddle_device, dtype=paddle.float16)
    #     image_fp16 = pipe(**inputs).images

    #     with torch.autocast(paddle_device):
    #         inputs = self.get_inputs()
    #         image_autocast = pipe(**inputs).images

    #     # Make sure results are close enough
    #     diff = np.abs(image_fp16.flatten() - image_autocast.flatten())
    #     # They ARE different since ops are not run always at the same precision
    #     # however, they should be extremely close.
    #     assert diff.mean() < 2e-2

    def test_stable_diffusion_intermediate_state(self):
        number_of_steps = 0

        def callback_fn(step: int, timestep: int, latents: paddle.Tensor) -> None:
            callback_fn.has_been_called = True
            nonlocal number_of_steps
            number_of_steps += 1
            if step == 1:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                # expected_slice = np.array(
                #     [-0.5693, -0.3018, -0.9746, 0.0518, -0.8770, 0.7559, -1.7402, 0.1022, 1.1582]
                # )

                # expected_slice = np.array([1.8209, 1.5543, 0.2858, 0.9747, -2.2018, 0.8413, -0.2585, -0.8049, -2.3286])
                expected_slice = np.array(
                    [
                        -0.56592464,
                        -0.3014981,
                        -0.9714532,
                        0.0543119,
                        -0.87713623,
                        0.7536549,
                        -1.7345023,
                        0.10148501,
                        1.1519401,
                    ]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2
            elif step == 2:
                latents = latents.detach().cpu().numpy()
                assert latents.shape == (1, 4, 64, 64)
                latents_slice = latents[0, -3:, -3:, -1]
                # expected_slice = np.array(
                #     [-0.1958, -0.2993, -1.0166, -0.5005, -0.4810, 0.6162, -0.9492, 0.6621, 1.4492]
                # )

                # expected_slice = np.array([1.2777, 1.5447, 0.6534, 0.2543, -1.6462, 0.659, -0.4193, -0.8901, -0.7815])
                expected_slice = np.array(
                    [
                        -0.17635298,
                        -0.30507395,
                        -1.0076891,
                        -0.500456,
                        -0.4666257,
                        0.60124385,
                        -0.935606,
                        0.65882003,
                        1.4459747,
                    ]
                )
                assert np.abs(latents_slice.flatten() - expected_slice).max() < 5e-2

        callback_fn.has_been_called = False

        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=paddle.float16)
        pipe = pipe
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs(dtype=paddle.float16)
        pipe(**inputs, callback=callback_fn, callback_steps=1)
        assert callback_fn.has_been_called
        assert number_of_steps == inputs["num_inference_steps"]

    # def test_stable_diffusion_low_cpu_mem_usage(self):
    #     pipeline_id = "CompVis/stable-diffusion-v1-4"

    #     start_time = time.time()
    #     pipeline_low_cpu_mem_usage = StableDiffusionPipeline.from_pretrained(pipeline_id, torch_dtype=paddle.float16)
    #     pipeline_low_cpu_mem_usage
    #     low_cpu_mem_usage_time = time.time() - start_time

    #     start_time = time.time()
    #     _ = StableDiffusionPipeline.from_pretrained(pipeline_id, torch_dtype=paddle.float16, low_cpu_mem_usage=False)
    #     normal_load_time = time.time() - start_time

    #     assert 2 * low_cpu_mem_usage_time < normal_load_time

    # def test_stable_diffusion_pipeline_with_sequential_cpu_offloading(self):
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_max_memory_allocated()
    #     torch.cuda.reset_peak_memory_stats()

    #     pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=paddle.float16)
    #     pipe = pipe
    #     pipe.set_progress_bar_config(disable=None)
    #     pipe.enable_attention_slicing(1)
    #     pipe.enable_sequential_cpu_offload()

    #     inputs = self.get_inputs(paddle_device, dtype=paddle.float16)
    #     _ = pipe(**inputs)

    #     mem_bytes = torch.cuda.max_memory_allocated()
    #     # make sure that less than 2.8 GB is allocated
    #     assert mem_bytes < 2.8 * 10**9

    # def test_stable_diffusion_pipeline_with_model_offloading(self):
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_max_memory_allocated()
    #     torch.cuda.reset_peak_memory_stats()

    #     inputs = self.get_inputs(paddle_device, dtype=paddle.float16)

    #     # Normal inference

    #     pipe = StableDiffusionPipeline.from_pretrained(
    #         "CompVis/stable-diffusion-v1-4",
    #         torch_dtype=paddle.float16,
    #     )
    #     pipe.unet.set_default_attn_processor()
    #     pipe
    #     pipe.set_progress_bar_config(disable=None)
    #     outputs = pipe(**inputs)
    #     mem_bytes = torch.cuda.max_memory_allocated()

    #     # With model offloading

    #     # Reload but don't move to cuda
    #     pipe = StableDiffusionPipeline.from_pretrained(
    #         "CompVis/stable-diffusion-v1-4",
    #         torch_dtype=paddle.float16,
    #     )
    #     pipe.unet.set_default_attn_processor()

    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_max_memory_allocated()
    #     torch.cuda.reset_peak_memory_stats()

    #     pipe.enable_model_cpu_offload()
    #     pipe.set_progress_bar_config(disable=None)
    #     inputs = self.get_inputs(paddle_device, dtype=paddle.float16)

    #     outputs_offloaded = pipe(**inputs)
    #     mem_bytes_offloaded = torch.cuda.max_memory_allocated()

    #     images = outputs.images
    #     offloaded_images = outputs_offloaded.images

    #     max_diff = numpy_cosine_similarity_distance(images.flatten(), offloaded_images.flatten())
    #     assert max_diff < 1e-3
    #     assert mem_bytes_offloaded < mem_bytes
    #     assert mem_bytes_offloaded < 3.5 * 10**9
    #     for module in pipe.text_encoder, pipe.unet, pipe.vae:
    #         assert module.device == torch.device("cpu")

    #     # With attention slicing
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_max_memory_allocated()
    #     torch.cuda.reset_peak_memory_stats()

    #     pipe.enable_attention_slicing()
    #     _ = pipe(**inputs)
    #     mem_bytes_slicing = torch.cuda.max_memory_allocated()

    #     assert mem_bytes_slicing < mem_bytes_offloaded
    #     assert mem_bytes_slicing < 3 * 10**9

    # def test_stable_diffusion_textual_inversion(self):
    #     pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    #     pipe.load_textual_inversion("sd-concepts-library/low-poly-hd-logos-icons")

    #     a111_file = hf_hub_download("hf-internal-testing/text_inv_embedding_a1111_format", "winter_style.pt")
    #     a111_file_neg = hf_hub_download(
    #         "hf-internal-testing/text_inv_embedding_a1111_format", "winter_style_negative.pt"
    #     )
    #     pipe.load_textual_inversion(a111_file)
    #     pipe.load_textual_inversion(a111_file_neg)
    #     pipe.to("cuda")

    #     generator = paddle.Generator(device="cpu").manual_seed(1)

    #     prompt = "An logo of a turtle in strong Style-Winter with <low-poly-hd-logos-icons>"
    #     neg_prompt = "Style-Winter-neg"

    #     image = pipe(prompt=prompt, negative_prompt=neg_prompt, generator=generator, output_type="np").images[0]
    #     expected_image = load_numpy(
    #         "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text_inv/winter_logo_style.npy"
    #     )

    #     max_diff = np.abs(expected_image - image).max()
    #     assert max_diff < 8e-1

    # def test_stable_diffusion_textual_inversion_with_model_cpu_offload(self):
    #     pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    #     pipe.enable_model_cpu_offload()
    #     pipe.load_textual_inversion("sd-concepts-library/low-poly-hd-logos-icons")

    #     a111_file = hf_hub_download("hf-internal-testing/text_inv_embedding_a1111_format", "winter_style.pt")
    #     a111_file_neg = hf_hub_download(
    #         "hf-internal-testing/text_inv_embedding_a1111_format", "winter_style_negative.pt"
    #     )
    #     pipe.load_textual_inversion(a111_file)
    #     pipe.load_textual_inversion(a111_file_neg)

    #     generator = paddle.Generator(device="cpu").manual_seed(1)

    #     prompt = "An logo of a turtle in strong Style-Winter with <low-poly-hd-logos-icons>"
    #     neg_prompt = "Style-Winter-neg"

    #     image = pipe(prompt=prompt, negative_prompt=neg_prompt, generator=generator, output_type="np").images[0]
    #     expected_image = load_numpy(
    #         "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text_inv/winter_logo_style.npy"
    #     )

    #     max_diff = np.abs(expected_image - image).max()
    #     assert max_diff < 8e-1

    # def test_stable_diffusion_textual_inversion_with_sequential_cpu_offload(self):
    #     pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    #     pipe.enable_sequential_cpu_offload()
    #     pipe.load_textual_inversion("sd-concepts-library/low-poly-hd-logos-icons")

    #     a111_file = hf_hub_download("hf-internal-testing/text_inv_embedding_a1111_format", "winter_style.pt")
    #     a111_file_neg = hf_hub_download(
    #         "hf-internal-testing/text_inv_embedding_a1111_format", "winter_style_negative.pt"
    #     )
    #     pipe.load_textual_inversion(a111_file)
    #     pipe.load_textual_inversion(a111_file_neg)

    #     generator = paddle.Generator(device="cpu").manual_seed(1)

    #     prompt = "An logo of a turtle in strong Style-Winter with <low-poly-hd-logos-icons>"
    #     neg_prompt = "Style-Winter-neg"

    #     image = pipe(prompt=prompt, negative_prompt=neg_prompt, generator=generator, output_type="np").images[0]
    #     expected_image = load_numpy(
    #         "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/text_inv/winter_logo_style.npy"
    #     )

    #     max_diff = np.abs(expected_image - image).max()
    #     assert max_diff < 8e-1

    # @require_python39_or_higher
    # @require_torch_2
    # def test_stable_diffusion_compile(self):
    #     seed = 0
    #     inputs = self.get_inputs(paddle_device, seed=seed)
    #     # Can't pickle a Generator object
    #     del inputs["generator"]
    #     inputs["paddle_device"] = paddle_device
    #     inputs["seed"] = seed
    #     run_test_in_subprocess(test_case=self, target_func=_test_stable_diffusion_compile, inputs=inputs)

    # def test_stable_diffusion_lcm(self):
    #     unet = UNet2DConditionModel.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", subfolder="unet")
    #     sd_pipe = StableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-7", unet=unet)
    #     sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
    #     sd_pipe.set_progress_bar_config(disable=None)

    #     inputs = self.get_inputs()
    #     inputs["num_inference_steps"] = 6
    #     inputs["output_type"] = "pil"

    #     image = sd_pipe(**inputs).images[0]

    #     expected_image = load_image(
    #         "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/lcm_full/stable_diffusion_lcm.png"
    #     )

    #     image = sd_pipe.image_processor.pil_to_numpy(image)
    #     expected_image = sd_pipe.image_processor.pil_to_numpy(expected_image)

    #     max_diff = numpy_cosine_similarity_distance(image.flatten(), expected_image.flatten())

    #     assert max_diff < 1e-2


# @slow
# @require_paddle_gpu
# class StableDiffusionPipelineCkptTests(unittest.TestCase):
#     def tearDown(self):
#         super().tearDown()
#         gc.collect()
#         paddle.device.cuda.empty_cache()

#     def test_download_from_hub(self):
#         ckpt_paths = [
#             "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt",
#             "https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix_base.ckpt",
#         ]

#         for ckpt_path in ckpt_paths:
#             pipe = StableDiffusionPipeline.from_single_file(ckpt_path, torch_dtype=paddle.float16)
#             pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
#             pipe.to("cuda")

#         image_out = pipe("test", num_inference_steps=1, output_type="np").images[0]

#         assert image_out.shape == (512, 512, 3)

#     def test_download_local(self):
#         filename = hf_hub_download("runwayml/stable-diffusion-v1-5", filename="v1-5-pruned-emaonly.ckpt")

#         pipe = StableDiffusionPipeline.from_single_file(filename, torch_dtype=paddle.float16)
#         pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
#         pipe.to("cuda")

#         image_out = pipe("test", num_inference_steps=1, output_type="np").images[0]

#         assert image_out.shape == (512, 512, 3)

#     def test_download_ckpt_diff_format_is_same(self):
#         ckpt_path = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt"

#         pipe = StableDiffusionPipeline.from_single_file(ckpt_path)
#         pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
#         pipe.unet.set_attn_processor(AttnProcessor())
#         pipe.to("cuda")

#         generator = paddle.Generator(device="cpu").manual_seed(0)
#         image_ckpt = pipe("a turtle", num_inference_steps=2, generator=generator, output_type="np").images[0]

#         pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
#         pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
#         pipe.unet.set_attn_processor(AttnProcessor())
#         pipe.to("cuda")

#         generator = paddle.Generator(device="cpu").manual_seed(0)
#         image = pipe("a turtle", num_inference_steps=2, generator=generator, output_type="np").images[0]

#         max_diff = numpy_cosine_similarity_distance(image.flatten(), image_ckpt.flatten())

#         assert max_diff < 1e-3


@nightly
@require_paddle_gpu
class StableDiffusionPipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, device="gpu", generator_device="cpu", dtype=paddle.float32, seed=0):
        generator = paddle.Generator(device=generator_device).manual_seed(seed)
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

    def test_stable_diffusion_1_4_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        # expected_image = load_numpy(
        #     "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
        #     "/stable_diffusion_text2img/stable_diffusion_1_4_pndm.npy"
        # )
        expected_image = np.array(
            [
                [0.0, 0.00271818, 0.01079074],
                [0.0, 0.0, 0.0],
                [0.01092488, 0.00601336, 0.00778148],
            ]
        )
        expected_image = np.array(
            [
                [0.82469547, 0.73561275, 0.6322198],
                [0.84452856, 0.7488118, 0.62713593],
                [0.8450513, 0.7500273, 0.655013],
            ]
        )
        max_diff = np.abs(expected_image - image[0][0:3]).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_1_5_pndm(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        # expected_image = load_numpy(
        #     "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
        #     "/stable_diffusion_text2img/stable_diffusion_1_5_pndm.npy"
        # )
        # max_diff = np.abs(expected_image - image).max()
        expected_image = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.00028974, 0.0, 0.0],
            ]
        )
        expected_image = np.array(
            [
                [0.7839483, 0.65648746, 0.48896584],
                [0.7808857, 0.6400481, 0.44772914],
                [0.8145921, 0.678653, 0.51496196],
            ]
        )
        max_diff = np.abs(expected_image - image[0][0:3]).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_ddim(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe.scheduler = DDIMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        # expected_image = load_numpy(
        #     "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
        #     "/stable_diffusion_text2img/stable_diffusion_1_4_ddim.npy"
        # )
        # max_diff = np.abs(expected_image - image).max()
        expected_image = np.array(
            [
                [0.22964239, 0.2353485, 0.22666347],
                [0.19678515, 0.1955398, 0.19239765],
                [0.22642559, 0.21655038, 0.22287127],
            ]
        )
        expected_image = np.array(
            [
                [0.8441943, 0.7534112, 0.6637156],
                [0.8470144, 0.74809825, 0.6476983],
                [0.852551, 0.7539777, 0.67560124],
            ]
        )
        max_diff = np.abs(expected_image - image[0][0:3]).max()
        assert max_diff < 3e-3

    def test_stable_diffusion_lms(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        # expected_image = load_numpy(
        #     "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
        #     "/stable_diffusion_text2img/stable_diffusion_1_4_lms.npy"
        # )
        # max_diff = np.abs(expected_image - image).max()
        # expected_image = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        expected_image = np.array(
            [[0.78707314, 0.6970422, 0.591521], [0.8043251, 0.7026402, 0.5748485], [0.8073704, 0.7054389, 0.6011865]]
        )
        max_diff = np.abs(expected_image - image[0][0:3]).max()
        assert max_diff < 1e-3

    def test_stable_diffusion_euler(self):
        sd_pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]

        # expected_image = load_numpy(
        #     "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
        #     "/stable_diffusion_text2img/stable_diffusion_1_4_euler.npy"
        # )
        # max_diff = np.abs(expected_image - image).max()
        expected_image = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        expected_image = np.array(
            [
                [0.79075336, 0.6989653, 0.59113663],
                [0.7878195, 0.68153524, 0.55696714],
                [0.79491574, 0.69076866, 0.58901614],
            ]
        )
        max_diff = np.abs(expected_image - image[0][0:3]).max()
        assert max_diff < 1e-3
