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

from ppdiffusers import UniDiffuserPipeline
from ppdiffusers.utils import load_image, randn_tensor, slow
from ppdiffusers.utils.testing_utils import require_paddle_gpu


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
