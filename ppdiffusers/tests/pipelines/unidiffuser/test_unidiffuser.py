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
from PIL import Image

from ppdiffusers import UniDiffuserPipeline
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
        pipe = UniDiffuserPipeline.from_pretrained("hf-internal-testing/unidiffuser-ppdiffusers-test")
        components = {
            "vae": pipe.vae,
            "text_encoder": pipe.text_encoder,
            "image_encoder": pipe.image_encoder,
            "image_processor": pipe.image_processor,
            "clip_tokenizer": pipe.clip_tokenizer,
            "text_decoder": pipe.text_decoder,
            "text_tokenizer": pipe.text_tokenizer,
            "unet": pipe.unet,
            "scheduler": pipe.scheduler,
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
            "output_type": "np",
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
            "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/unidiffuser/unidiffuser_example_image.jpg"
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
            "output_type": "np",
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


@slow
@require_paddle_gpu
class UniDiffuserPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, seed=0, generate_latents=False):
        generator = paddle.Generator().manual_seed(seed)
        image = load_image(
            "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/unidiffuser/unidiffuser_example_image.jpg"
        )
        inputs = {
            "prompt": "an elephant under the sea",
            "image": image,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 8.0,
            "output_type": "np",
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
        expected_img_slice = np.array([0.0484, 0.0802, 0.053, 0.0782, 0.0824, 0.0817, 0.0822, 0.074, 0.023])
        assert np.abs(image_slice.flatten() - expected_img_slice).max() < 0.1
        expected_text_prefix = "a yellow gable"
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
        expected_slice = np.array([0.1214, 0.1147, 0.113, 0.1092, 0.1107, 0.0977, 0.1324, 0.1324, 0.0884])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.1

    def test_unidiffuser_default_img2text_v1(self):
        pipe = UniDiffuserPipeline.from_pretrained("thu-ml/unidiffuser-v1")
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs(generate_latents=True)
        del inputs["prompt"]
        sample = pipe(**inputs)
        text = sample.text
        expected_text_prefix = "An image of an astronaut"
        assert text[0][: len(expected_text_prefix)] == expected_text_prefix
