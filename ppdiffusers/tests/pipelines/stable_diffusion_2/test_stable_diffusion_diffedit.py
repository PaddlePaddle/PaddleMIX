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
import tempfile
import unittest

import numpy as np
import paddle
from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from PIL import Image

from ppdiffusers import (
    AutoencoderKL,
    DDIMInverseScheduler,
    DDIMScheduler,
    DPMSolverMultistepInverseScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionDiffEditPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import load_image, slow
from ppdiffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    require_paddle_gpu,
)

from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_INPAINTING_PARAMS,
)
from ..test_pipelines_common import PipelineLatentTesterMixin, PipelineTesterMixin

enable_full_determinism()


class StableDiffusionDiffEditPipelineFastTests(PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionDiffEditPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS - {"height", "width", "image"} | {"image_latents"}
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS - {"image"} | {"image_latents"}
    image_params = frozenset([])
    image_latents_params = frozenset([])

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
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        inverse_scheduler = DDIMInverseScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_zero=False,
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
            "inverse_scheduler": inverse_scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, seed=0):
        mask = floats_tensor((1, 16, 16), rng=random.Random(seed))
        latents = floats_tensor((1, 2, 4, 16, 16), rng=random.Random(seed))

        generator = paddle.Generator().manual_seed(seed)

        inputs = {
            "prompt": "a dog and a newt",
            "mask_image": mask,
            "image_latents": latents,
            "generator": generator,
            "num_inference_steps": 2,
            "inpaint_strength": 1.0,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    def get_dummy_mask_inputs(self, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.cpu().transpose(perm=[0, 2, 3, 1])[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB")

        generator = paddle.Generator().manual_seed(seed)
        inputs = {
            "image": image,
            "source_prompt": "a cat and a frog",
            "target_prompt": "a dog and a newt",
            "generator": generator,
            "num_inference_steps": 2,
            "num_maps_per_mask": 2,
            "mask_encode_strength": 1.0,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    def get_dummy_inversion_inputs(self, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.cpu().transpose(perm=[0, 2, 3, 1])[0]
        image = Image.fromarray(np.uint8(image)).convert("RGB")
        generator = paddle.Generator().manual_seed(seed)
        inputs = {
            "image": image,
            "prompt": "a cat and a frog",
            "generator": generator,
            "num_inference_steps": 2,
            "inpaint_strength": 1.0,
            "guidance_scale": 6.0,
            "decode_latents": True,
            "output_type": "np",
        }
        return inputs

    def test_save_load_optional_components(self):
        if not hasattr(self.pipeline_class, "_optional_components"):
            return
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        for optional_component in pipe._optional_components:
            setattr(pipe, optional_component, None)
        pipe.register_modules(**{optional_component: None for optional_component in pipe._optional_components})
        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            pipe.save_pretrained(tmpdir)
            pipe_loaded = self.pipeline_class.from_pretrained(tmpdir)
            pipe_loaded.to()
            pipe_loaded.set_progress_bar_config(disable=None)
        for optional_component in pipe._optional_components:
            self.assertTrue(
                getattr(pipe_loaded, optional_component) is None,
                f"`{optional_component}` did not stay set to None after loading.",
            )
        inputs = self.get_dummy_inputs()
        output_loaded = pipe_loaded(**inputs)[0]
        max_diff = np.abs(output - output_loaded).max()
        self.assertLess(max_diff, 0.0001)

    def test_mask(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_mask_inputs()
        mask = pipe.generate_mask(**inputs)
        mask_slice = mask[(0), -3:, -3:]
        self.assertEqual(mask.shape, (1, 16, 16))
        expected_slice = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0])
        max_diff = np.abs(mask_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 0.001)
        self.assertEqual(mask[0, -3, -4], 0)

    def test_inversion(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inversion_inputs()
        image = pipe.invert(**inputs).images
        image_slice = image[(0), (-1), -3:, -3:]
        self.assertEqual(image.shape, (2, 32, 32, 3))
        expected_slice = np.array([0.4227, 0.4379, 0.4405, 0.5314, 0.5412, 0.523, 0.5163, 0.4944, 0.48])
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 0.001)

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=0.005)

    def test_inversion_dpm(self):
        components = self.get_dummy_components()
        scheduler_args = {"beta_start": 0.00085, "beta_end": 0.012, "beta_schedule": "scaled_linear"}
        components["scheduler"] = DPMSolverMultistepScheduler(**scheduler_args)
        components["inverse_scheduler"] = DPMSolverMultistepInverseScheduler(**scheduler_args)
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inversion_inputs()
        image = pipe.invert(**inputs).images
        image_slice = image[(0), (-1), -3:, -3:]
        self.assertEqual(image.shape, (2, 32, 32, 3))
        expected_slice = np.array([0.7061, 0.9805, 0.6064, 0.7288, 0.8416, 0.6517, 0.5912, 0.5704, 0.5726])
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 0.001)


@require_paddle_gpu
@slow
class StableDiffusionDiffEditPipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    @classmethod
    def setUpClass(cls):
        raw_image = load_image(
            "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/diffedit/fruit.png"
        )
        raw_image = raw_image.convert("RGB").resize((768, 768))
        cls.raw_image = raw_image

    def test_stable_diffusion_diffedit_full(self):
        generator = paddle.Generator().manual_seed(seed=0)
        pipe = StableDiffusionDiffEditPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", safety_checker=None, paddle_dtype="float16"
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        source_prompt = "a bowl of fruit"
        target_prompt = "a bowl of pears"
        mask_image = pipe.generate_mask(
            image=self.raw_image, source_prompt=source_prompt, target_prompt=target_prompt, generator=generator
        )
        inv_latents = pipe.invert(
            prompt=source_prompt, image=self.raw_image, inpaint_strength=0.7, generator=generator
        ).latents
        image = pipe(
            prompt=target_prompt,
            mask_image=mask_image,
            image_latents=inv_latents,
            generator=generator,
            negative_prompt=source_prompt,
            inpaint_strength=0.7,
            output_type="np",
        ).images[0]
        expected_image = (
            np.array(
                load_image(
                    "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/diffedit/pears.png"
                ).resize((768, 768))
            )
            / 255
        )
        assert np.abs((expected_image - image).max()) < 0.75

    def test_stable_diffusion_diffedit_dpm(self):
        generator = paddle.Generator().manual_seed(seed=0)
        pipe = StableDiffusionDiffEditPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", safety_checker=None, torch_dtype="float16"
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.inverse_scheduler = DPMSolverMultistepInverseScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        source_prompt = "a bowl of fruit"
        target_prompt = "a bowl of pears"
        mask_image = pipe.generate_mask(
            image=self.raw_image, source_prompt=source_prompt, target_prompt=target_prompt, generator=generator
        )
        inv_latents = pipe.invert(
            prompt=source_prompt,
            image=self.raw_image,
            inpaint_strength=0.7,
            generator=generator,
            num_inference_steps=25,
        ).latents
        image = pipe(
            prompt=target_prompt,
            mask_image=mask_image,
            image_latents=inv_latents,
            generator=generator,
            negative_prompt=source_prompt,
            inpaint_strength=0.7,
            num_inference_steps=25,
            output_type="np",
        ).images[0]
        expected_image = (
            np.array(
                load_image(
                    "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/diffedit/pears.png"
                ).resize((768, 768))
            )
            / 255
        )
        assert np.abs((expected_image - image).max()) < 0.5
