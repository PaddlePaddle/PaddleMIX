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
import random
import unittest

import numpy as np
import paddle
from paddlenlp.transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    GPT2Tokenizer,
)
from PIL import Image

from ppdiffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    UniDiffuserModel,
    UniDiffuserPipeline,
    UniDiffuserTextDecoder,
)
from ppdiffusers.image_processor import CLIPImageProcessor
from ppdiffusers.utils import floats_tensor, load_image, randn_tensor, slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu

from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
)
from ..test_pipelines_common import PipelineTesterMixin


class UniDiffuserPipelineFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = UniDiffuserPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS

    def get_dummy_components(self):
        unet = UniDiffuserModel.from_pretrained("hf-internal-testing/unidiffuser-diffusers-test", subfolder="unet")
        scheduler = DPMSolverMultistepScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", solver_order=3
        )
        vae = AutoencoderKL.from_pretrained("hf-internal-testing/unidiffuser-diffusers-test", subfolder="vae")
        text_encoder = CLIPTextModel.from_pretrained(
            "hf-internal-testing/unidiffuser-diffusers-test", subfolder="text_encoder"
        )
        clip_tokenizer = CLIPTokenizer.from_pretrained(
            "hf-internal-testing/unidiffuser-diffusers-test", subfolder="clip_tokenizer"
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "hf-internal-testing/unidiffuser-diffusers-test", subfolder="image_encoder"
        )
        image_processor = CLIPImageProcessor(crop_size=32, size=32)
        text_tokenizer = GPT2Tokenizer.from_pretrained(
            "hf-internal-testing/unidiffuser-diffusers-test", subfolder="text_tokenizer"
        )
        text_decoder = UniDiffuserTextDecoder.from_pretrained(
            "hf-internal-testing/unidiffuser-diffusers-test", subfolder="text_decoder"
        )
        components = {
            "vae": vae,
            "text_encoder": text_encoder,
            "image_encoder": image_encoder,
            "image_processor": image_processor,
            "clip_tokenizer": clip_tokenizer,
            "text_decoder": text_decoder,
            "text_tokenizer": text_tokenizer,
            "unet": unet,
            "scheduler": scheduler,
        }
        return components

    def get_dummy_inputs(self, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.cpu().transpose(perm=[0, 2, 3, 1])[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB")
        generator = paddle.Generator().manual_seed(seed)
        inputs = {
            "prompt": "an elephant under the sea",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def get_fixed_latents(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        prompt_latents = randn_tensor((1, 77, 32), generator=generator, dtype="float32")
        vae_latents = randn_tensor((1, 4, 16, 16), generator=generator, dtype="float32")
        clip_latents = randn_tensor((1, 1, 32), generator=generator, dtype="float32")
        latents = {"prompt_latents": prompt_latents, "vae_latents": vae_latents, "clip_latents": clip_latents}
        return latents

    def get_dummy_inputs_with_latents(self, seed=0):
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/unidiffuser/unidiffuser_example_image.jpg"
        )
        image = image.resize((32, 32))
        latents = self.get_fixed_latents(seed=seed)
        generator = paddle.Generator().manual_seed(seed)
        inputs = {
            "prompt": "an elephant under the sea",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
            "prompt_latents": latents.get("prompt_latents"),
            "vae_latents": latents.get("vae_latents"),
            "clip_latents": latents.get("clip_latents"),
        }
        return inputs

    def test_unidiffuser_default_joint_v0(self):
        components = self.get_dummy_components()
        unidiffuser_pipe = UniDiffuserPipeline(**components)
        unidiffuser_pipe.set_progress_bar_config(disable=None)
        unidiffuser_pipe.set_joint_mode()
        assert unidiffuser_pipe.mode == "joint"
        inputs = self.get_dummy_inputs_with_latents()
        del inputs["prompt"]
        del inputs["image"]
        sample = unidiffuser_pipe(**inputs)
        image = sample.images
        text = sample.text
        assert image.shape == (1, 32, 32, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_img_slice = np.array([0.576, 0.627, 0.6571, 0.4965, 0.4638, 0.5663, 0.5254, 0.5068, 0.5716])
        assert np.abs(image_slice.flatten() - expected_img_slice).max() < 0.001
        expected_text_prefix = " no no no "
        assert text[0][:10] == expected_text_prefix

    def test_unidiffuser_default_joint_no_cfg_v0(self):
        components = self.get_dummy_components()
        unidiffuser_pipe = UniDiffuserPipeline(**components)
        unidiffuser_pipe.set_progress_bar_config(disable=None)
        unidiffuser_pipe.set_joint_mode()
        assert unidiffuser_pipe.mode == "joint"
        inputs = self.get_dummy_inputs_with_latents()
        del inputs["prompt"]
        del inputs["image"]
        inputs["guidance_scale"] = 1.0
        sample = unidiffuser_pipe(**inputs)
        image = sample.images
        text = sample.text
        assert image.shape == (1, 32, 32, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_img_slice = np.array([0.576, 0.627, 0.6571, 0.4965, 0.4638, 0.5663, 0.5254, 0.5068, 0.5716])
        assert np.abs(image_slice.flatten() - expected_img_slice).max() < 0.001
        expected_text_prefix = " no no no "
        assert text[0][:10] == expected_text_prefix

    def test_unidiffuser_default_text2img_v0(self):
        components = self.get_dummy_components()
        unidiffuser_pipe = UniDiffuserPipeline(**components)
        unidiffuser_pipe.set_progress_bar_config(disable=None)
        unidiffuser_pipe.set_text_to_image_mode()
        assert unidiffuser_pipe.mode == "text2img"
        inputs = self.get_dummy_inputs_with_latents()
        del inputs["image"]
        image = unidiffuser_pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_slice = np.array([0.5758, 0.6269, 0.657, 0.4967, 0.4639, 0.5664, 0.5257, 0.5067, 0.5715])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

    def test_unidiffuser_default_image_0(self):
        components = self.get_dummy_components()
        unidiffuser_pipe = UniDiffuserPipeline(**components)
        unidiffuser_pipe.set_progress_bar_config(disable=None)
        unidiffuser_pipe.set_image_mode()
        assert unidiffuser_pipe.mode == "img"
        inputs = self.get_dummy_inputs()
        del inputs["prompt"]
        del inputs["image"]
        image = unidiffuser_pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_slice = np.array([0.576, 0.627, 0.6571, 0.4966, 0.4638, 0.5663, 0.5254, 0.5068, 0.5715])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

    def test_unidiffuser_default_text_v0(self):
        components = self.get_dummy_components()
        unidiffuser_pipe = UniDiffuserPipeline(**components)
        unidiffuser_pipe.set_progress_bar_config(disable=None)
        unidiffuser_pipe.set_text_mode()
        assert unidiffuser_pipe.mode == "text"
        inputs = self.get_dummy_inputs()
        del inputs["prompt"]
        del inputs["image"]
        text = unidiffuser_pipe(**inputs).text
        expected_text_prefix = " no no no "
        assert text[0][:10] == expected_text_prefix

    def test_unidiffuser_default_img2text_v0(self):
        components = self.get_dummy_components()
        unidiffuser_pipe = UniDiffuserPipeline(**components)
        unidiffuser_pipe.set_progress_bar_config(disable=None)
        unidiffuser_pipe.set_image_to_text_mode()
        assert unidiffuser_pipe.mode == "img2text"
        inputs = self.get_dummy_inputs_with_latents()
        del inputs["prompt"]
        text = unidiffuser_pipe(**inputs).text
        expected_text_prefix = " no no no "
        assert text[0][:10] == expected_text_prefix

    def test_unidiffuser_default_joint_v1(self):
        unidiffuser_pipe = UniDiffuserPipeline.from_pretrained("hf-internal-testing/unidiffuser-test-v1")
        unidiffuser_pipe.set_progress_bar_config(disable=None)
        unidiffuser_pipe.set_joint_mode()
        assert unidiffuser_pipe.mode == "joint"
        inputs = self.get_dummy_inputs_with_latents()
        del inputs["prompt"]
        del inputs["image"]
        inputs["data_type"] = 1
        sample = unidiffuser_pipe(**inputs)
        image = sample.images
        text = sample.text
        assert image.shape == (1, 32, 32, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_img_slice = np.array([0.576, 0.627, 0.6571, 0.4965, 0.4638, 0.5663, 0.5254, 0.5068, 0.5716])
        assert np.abs(image_slice.flatten() - expected_img_slice).max() < 0.001
        expected_text_prefix = " no no no "
        assert text[0][:10] == expected_text_prefix

    def test_unidiffuser_default_text2img_v1(self):
        unidiffuser_pipe = UniDiffuserPipeline.from_pretrained("hf-internal-testing/unidiffuser-test-v1")
        unidiffuser_pipe.set_progress_bar_config(disable=None)
        unidiffuser_pipe.set_text_to_image_mode()
        assert unidiffuser_pipe.mode == "text2img"
        inputs = self.get_dummy_inputs_with_latents()
        del inputs["image"]
        image = unidiffuser_pipe(**inputs).images
        assert image.shape == (1, 32, 32, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_slice = np.array([0.5758, 0.6269, 0.657, 0.4967, 0.4639, 0.5664, 0.5257, 0.5067, 0.5715])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

    def test_unidiffuser_default_img2text_v1(self):
        unidiffuser_pipe = UniDiffuserPipeline.from_pretrained("hf-internal-testing/unidiffuser-test-v1")
        unidiffuser_pipe.set_progress_bar_config(disable=None)
        unidiffuser_pipe.set_image_to_text_mode()
        assert unidiffuser_pipe.mode == "img2text"
        inputs = self.get_dummy_inputs_with_latents()
        del inputs["prompt"]
        text = unidiffuser_pipe(**inputs).text
        expected_text_prefix = " no no no "
        assert text[0][:10] == expected_text_prefix

    def test_unidiffuser_text2img_multiple_images(self):
        components = self.get_dummy_components()
        unidiffuser_pipe = UniDiffuserPipeline(**components)
        unidiffuser_pipe.set_progress_bar_config(disable=None)
        unidiffuser_pipe.set_text_to_image_mode()
        assert unidiffuser_pipe.mode == "text2img"
        inputs = self.get_dummy_inputs()
        del inputs["image"]
        inputs["num_images_per_prompt"] = 2
        inputs["num_prompts_per_image"] = 3
        image = unidiffuser_pipe(**inputs).images
        assert image.shape == (2, 32, 32, 3)

    def test_unidiffuser_img2text_multiple_prompts(self):
        components = self.get_dummy_components()
        unidiffuser_pipe = UniDiffuserPipeline(**components)
        unidiffuser_pipe.set_progress_bar_config(disable=None)
        unidiffuser_pipe.set_image_to_text_mode()
        assert unidiffuser_pipe.mode == "img2text"
        inputs = self.get_dummy_inputs()
        del inputs["prompt"]
        inputs["num_images_per_prompt"] = 2
        inputs["num_prompts_per_image"] = 3
        text = unidiffuser_pipe(**inputs).text
        assert len(text) == 3

    def test_unidiffuser_text2img_multiple_images_with_latents(self):
        components = self.get_dummy_components()
        unidiffuser_pipe = UniDiffuserPipeline(**components)
        unidiffuser_pipe.set_progress_bar_config(disable=None)
        unidiffuser_pipe.set_text_to_image_mode()
        assert unidiffuser_pipe.mode == "text2img"
        inputs = self.get_dummy_inputs_with_latents()
        del inputs["image"]
        inputs["num_images_per_prompt"] = 2
        inputs["num_prompts_per_image"] = 3
        image = unidiffuser_pipe(**inputs).images
        assert image.shape == (2, 32, 32, 3)

    def test_unidiffuser_img2text_multiple_prompts_with_latents(self):
        components = self.get_dummy_components()
        unidiffuser_pipe = UniDiffuserPipeline(**components)
        unidiffuser_pipe.set_progress_bar_config(disable=None)
        unidiffuser_pipe.set_image_to_text_mode()
        assert unidiffuser_pipe.mode == "img2text"
        inputs = self.get_dummy_inputs_with_latents()
        del inputs["prompt"]
        inputs["num_images_per_prompt"] = 2
        inputs["num_prompts_per_image"] = 3
        text = unidiffuser_pipe(**inputs).text
        assert len(text) == 3

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical()

    @require_paddle_gpu
    def test_unidiffuser_default_joint_v1_cuda_fp16(self):
        unidiffuser_pipe = UniDiffuserPipeline.from_pretrained(
            "hf-internal-testing/unidiffuser-test-v1", paddle_dtype="float16"
        )
        unidiffuser_pipe.set_progress_bar_config(disable=None)
        unidiffuser_pipe.set_joint_mode()
        assert unidiffuser_pipe.mode == "joint"
        inputs = self.get_dummy_inputs_with_latents()
        del inputs["prompt"]
        del inputs["image"]
        inputs["data_type"] = 1
        sample = unidiffuser_pipe(**inputs)
        image = sample.images
        text = sample.text
        assert image.shape == (1, 32, 32, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_img_slice = np.array([0.5049, 0.5498, 0.5854, 0.3052, 0.446, 0.6489, 0.5122, 0.481, 0.6138])
        assert np.abs(image_slice.flatten() - expected_img_slice).max() < 0.001
        expected_text_prefix = '" This This'
        assert text[0][: len(expected_text_prefix)] == expected_text_prefix

    @require_paddle_gpu
    def test_unidiffuser_default_text2img_v1_cuda_fp16(self):
        unidiffuser_pipe = UniDiffuserPipeline.from_pretrained(
            "hf-internal-testing/unidiffuser-test-v1", paddle_dtype="float16"
        )
        unidiffuser_pipe.set_progress_bar_config(disable=None)
        unidiffuser_pipe.set_text_to_image_mode()
        assert unidiffuser_pipe.mode == "text2img"
        inputs = self.get_dummy_inputs_with_latents()
        del inputs["image"]
        inputs["data_type"] = 1
        sample = unidiffuser_pipe(**inputs)
        image = sample.images
        assert image.shape == (1, 32, 32, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_img_slice = np.array([0.5054, 0.5498, 0.5854, 0.3052, 0.4458, 0.6489, 0.5122, 0.481, 0.6138])
        assert np.abs(image_slice.flatten() - expected_img_slice).max() < 0.001

    @require_paddle_gpu
    def test_unidiffuser_default_img2text_v1_cuda_fp16(self):
        unidiffuser_pipe = UniDiffuserPipeline.from_pretrained(
            "hf-internal-testing/unidiffuser-test-v1", paddle_dtype="float16"
        )
        unidiffuser_pipe.set_progress_bar_config(disable=None)
        unidiffuser_pipe.set_image_to_text_mode()
        assert unidiffuser_pipe.mode == "img2text"
        inputs = self.get_dummy_inputs_with_latents()
        del inputs["prompt"]
        inputs["data_type"] = 1
        text = unidiffuser_pipe(**inputs).text
        expected_text_prefix = '" This This'
        assert text[0][: len(expected_text_prefix)] == expected_text_prefix


@slow
@require_paddle_gpu
class UniDiffuserPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, seed=0, generate_latents=False):
        generator = paddle.seed(seed=seed)
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/unidiffuser/unidiffuser_example_image.jpg"
        )
        inputs = {
            "prompt": "an elephant under the sea",
            "image": image,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 8.0,
            "output_type": "numpy",
        }
        if generate_latents:
            latents = self.get_fixed_latents(seed=seed)
            for latent_name, latent_tensor in latents.items():
                inputs[latent_name] = latent_tensor
        return inputs

    def get_fixed_latents(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        prompt_latents = randn_tensor((1, 77, 768), generator=generator, dtype="float32")
        vae_latents = randn_tensor((1, 4, 64, 64), generator=generator, dtype="float32")
        clip_latents = randn_tensor((1, 1, 512), generator=generator, dtype="float32")

        latents = {"prompt_latents": prompt_latents, "vae_latents": vae_latents, "clip_latents": clip_latents}
        return latents

    def test_unidiffuser_default_joint_v1(self):
        pipe = UniDiffuserPipeline.from_pretrained("thu-ml/unidiffuser-v1")
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(generate_latents=True)
        del inputs["prompt"]
        del inputs["image"]
        sample = pipe(**inputs)
        image = sample.images
        text = sample.text
        assert image.shape == (1, 512, 512, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_img_slice = np.array([0.2402, 0.2375, 0.2285, 0.2378, 0.2407, 0.2263, 0.2354, 0.2307, 0.252])
        assert np.abs(image_slice.flatten() - expected_img_slice).max() < 0.1
        expected_text_prefix = "a living room"
        assert text[0][: len(expected_text_prefix)] == expected_text_prefix

    def test_unidiffuser_default_text2img_v1(self):
        pipe = UniDiffuserPipeline.from_pretrained("thu-ml/unidiffuser-v1")
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(generate_latents=True)
        del inputs["image"]
        sample = pipe(**inputs)
        image = sample.images
        assert image.shape == (1, 512, 512, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_slice = np.array([0.0242, 0.0103, 0.0022, 0.0129, 0.0, 0.009, 0.0376, 0.0508, 0.0005])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.1

    def test_unidiffuser_default_img2text_v1(self):
        pipe = UniDiffuserPipeline.from_pretrained("thu-ml/unidiffuser-v1")
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(generate_latents=True)
        del inputs["prompt"]
        sample = pipe(**inputs)
        text = sample.text
        expected_text_prefix = "An astronaut"
        assert text[0][: len(expected_text_prefix)] == expected_text_prefix

    def test_unidiffuser_default_joint_v1_fp16(self):
        pipe = UniDiffuserPipeline.from_pretrained("thu-ml/unidiffuser-v1", paddle_dtype="float16")
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(generate_latents=True)
        del inputs["prompt"]
        del inputs["image"]
        sample = pipe(**inputs)
        image = sample.images
        text = sample.text
        assert image.shape == (1, 512, 512, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_img_slice = np.array([0.2402, 0.2375, 0.2285, 0.2378, 0.2407, 0.2263, 0.2354, 0.2307, 0.252])
        assert np.abs(image_slice.flatten() - expected_img_slice).max() < 0.2
        expected_text_prefix = "a living room"
        assert text[0][: len(expected_text_prefix)] == expected_text_prefix

    def test_unidiffuser_default_text2img_v1_fp16(self):
        pipe = UniDiffuserPipeline.from_pretrained("thu-ml/unidiffuser-v1", paddle_dtype="float16")
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(generate_latents=True)
        del inputs["image"]
        sample = pipe(**inputs)
        image = sample.images
        assert image.shape == (1, 512, 512, 3)
        image_slice = image[(0), -3:, -3:, (-1)]
        expected_slice = np.array([0.0242, 0.0103, 0.0022, 0.0129, 0.0, 0.009, 0.0376, 0.0508, 0.0005])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.1

    def test_unidiffuser_default_img2text_v1_fp16(self):
        pipe = UniDiffuserPipeline.from_pretrained("thu-ml/unidiffuser-v1", paddle_dtype="float16")
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(generate_latents=True)
        del inputs["prompt"]
        sample = pipe(**inputs)
        text = sample.text
        expected_text_prefix = "An astronaut"
        assert text[0][: len(expected_text_prefix)] == expected_text_prefix
