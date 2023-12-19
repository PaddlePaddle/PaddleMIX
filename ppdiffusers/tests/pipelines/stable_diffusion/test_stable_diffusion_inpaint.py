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
import random
import unittest

import numpy as np
import paddle
from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from PIL import Image

from ppdiffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    prepare_mask_and_masked_image,
)
from ppdiffusers.utils import floats_tensor, load_image, load_numpy, nightly, slow
from ppdiffusers.utils.testing_utils import (
    enable_full_determinism,
    paddle_device,
    require_paddle_gpu,
)

from ...models.test_models_unet_2d_condition import create_lora_layers
from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_INPAINTING_PARAMS,
)
from ..test_pipelines_common import PipelineLatentTesterMixin, PipelineTesterMixin

enable_full_determinism()


class StableDiffusionInpaintPipelineFastTests(PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionInpaintPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = frozenset([])
    image_latents_params = frozenset([])

    def get_dummy_components(self):
        paddle.seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=9,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        scheduler = PNDMScheduler(skip_prk_steps=True)
        paddle.seed(0)
        vae = AutoencoderKL(
            block_out_channels=[32, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
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
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image.cpu().transpose(perm=[0, 2, 3, 1])[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((64, 64))
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((64, 64))
        generator = paddle.Generator().manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_inpaint(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array(
            [0.55786943, 0.628228, 0.49147403, 0.3191774, 0.39249492, 0.46521175, 0.29909956, 0.21160087, 0.42932406]
        )
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_inpaint_image_tensor(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        output = sd_pipe(**inputs)
        out_pil = output.images
        inputs = self.get_dummy_inputs()
        inputs["image"] = (
            paddle.to_tensor(np.array(inputs["image"]) / 127.5 - 1).transpose(perm=[2, 0, 1]).unsqueeze(axis=0)
        )
        inputs["mask_image"] = (
            paddle.to_tensor(np.array(inputs["mask_image"]) / 255).transpose(perm=[2, 0, 1])[:1].unsqueeze(axis=0)
        )
        output = sd_pipe(**inputs)
        out_tensor = output.images
        assert out_pil.shape == (1, 64, 64, 3)
        assert np.abs(out_pil.flatten() - out_tensor.flatten()).max() < 0.05

    # TODO, fix this nan.
    def test_float16_inference(self, expected_max_diff=1e-2):
        pass

    def test_stable_diffusion_inpaint_lora(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe = sd_pipe.to(paddle_device)
        sd_pipe.set_progress_bar_config(disable=None)

        # forward 1
        inputs = self.get_dummy_inputs()
        output = sd_pipe(**inputs)
        image = output.images
        image_slice = image[(0), -3:, -3:, -1]

        # set lora layers
        lora_attn_procs = create_lora_layers(sd_pipe.unet)
        sd_pipe.unet.set_attn_processor(lora_attn_procs)
        sd_pipe = sd_pipe.to(paddle_device)

        # forward 2
        inputs = self.get_dummy_inputs()
        output = sd_pipe(**inputs, cross_attention_kwargs={"scale": 0.0})
        image = output.images
        image_slice_1 = image[(0), -3:, -3:, -1]

        # forward 3
        inputs = self.get_dummy_inputs()
        output = sd_pipe(**inputs, cross_attention_kwargs={"scale": 0.5})
        image = output.images
        image_slice_2 = image[(0), -3:, -3:, -1]

        assert np.abs(image_slice - image_slice_1).max() < 0.01
        assert np.abs(image_slice - image_slice_2).max() > 0.01

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical()

    def test_stable_diffusion_inpaint_strength_zero_test(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe = sd_pipe.to()
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        inputs["strength"] = 0.01
        with self.assertRaises(ValueError):
            sd_pipe(**inputs).images


class StableDiffusionSimpleInpaintPipelineFastTests(StableDiffusionInpaintPipelineFastTests):
    pipeline_class = StableDiffusionInpaintPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = frozenset([])

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
        scheduler = PNDMScheduler(skip_prk_steps=True)
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

    def test_stable_diffusion_inpaint(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.3216, 0.2463, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2595, 0.4763])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    @unittest.skip("skipped here because area stays unchanged due to mask")
    def test_stable_diffusion_inpaint_lora(self):
        ...


@slow
@require_paddle_gpu
class StableDiffusionInpaintPipelineSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        init_image = load_image("https://paddlenlp.bj.bcebos.com/data/images/input_bench_image.png")
        mask_image = load_image("https://paddlenlp.bj.bcebos.com/data/images/input_bench_mask.png")
        inputs = {
            "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_inpaint_ddim(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.05978, 0.10983, 0.10514, 0.07922, 0.08483, 0.08587, 0.05302, 0.03218, 0.01636])
        assert np.abs(expected_slice - image_slice).max() < 0.0001

    def test_stable_diffusion_inpaint_fp16(self):
        pass
        # pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-inpainting", paddle_dtype=paddle.float16, safety_checker=None
        # )
        # pipe.set_progress_bar_config(disable=None)
        # pipe.enable_attention_slicing()
        # inputs = self.get_inputs()
        # image = pipe(**inputs).images
        # image_slice = image[0, 253:256, 253:256, -1].flatten()
        # assert image.shape == (1, 512, 512, 3)
        # expected_slice = np.array(
        #     [0.049, 0.049, 0.049, 0.049, 0.049, 0.049, 0.049, 0.049, 0.049]
        # )
        # assert np.abs(expected_slice - image_slice).max() < 0.05

    def test_stable_diffusion_inpaint_pndm(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.06892, 0.06994, 0.07905, 0.05366, 0.04709, 0.04890, 0.04107, 0.05083, 0.04180])
        assert np.abs(expected_slice - image_slice).max() < 0.0001

    def test_stable_diffusion_inpaint_k_lms(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.23513, 0.22413, 0.29442, 0.24243, 0.26214, 0.30329, 0.26431, 0.25025, 0.25197])
        assert np.abs(expected_slice - image_slice).max() < 0.0001

    # def test_stable_diffusion_inpaint_with_sequential_cpu_offloading(self):
    #     paddle.device.cuda.empty_cache()
    #     pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #         "runwayml/stable-diffusion-inpainting", safety_checker=None, torch_dtype="float16"
    #     )
    #     pipe = pipe.to(paddle_device)
    #     pipe.set_progress_bar_config(disable=None)
    #     pipe.enable_attention_slicing(1)
    #     inputs = self.get_inputs(dtype="float16")
    #     _ = pipe(**inputs)
    #     mem_bytes = paddle.device.cuda.max_memory_allocated()
    #     assert mem_bytes < 2.2 * 10**9

    def test_stable_diffusion_inpaint_pil_input_resolution_test(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs()
        inputs["image"] = inputs["image"].resize((127, 127))
        inputs["mask_image"] = inputs["mask_image"].resize((127, 127))
        inputs["height"] = 128
        inputs["width"] = 128
        image = pipe(**inputs).images
        assert image.shape == (1, inputs["height"], inputs["width"], 3)

    def test_stable_diffusion_inpaint_strength_test(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs()
        inputs["strength"] = 0.75
        image = pipe(**inputs).images
        assert image.shape == (1, 512, 512, 3)
        image_slice = image[(0), 253:256, 253:256, (-1)].flatten()
        expected_slice = np.array([0.0972, 0.2068, 0.1759, 0.1449, 0.1885, 0.1117, 0.1314, 0.1382, 0.0856])
        assert np.abs(expected_slice - image_slice).max() < 0.003

    def test_stable_diffusion_simple_inpaint_ddim(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        inputs = self.get_inputs()
        image = pipe(**inputs).images
        image_slice = image[(0), 253:256, 253:256, (-1)].flatten()
        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.2528, 0.2133, 0.3194, 0.2348, 0.1933, 0.332, 0.2186, 0.1664, 0.3176])
        assert np.abs(expected_slice - image_slice).max() < 0.3


@slow
@require_paddle_gpu
class StableDiffusionInpaintPipelineAsymmetricAutoencoderKLSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        init_image = load_image("https://paddlenlp.bj.bcebos.com/data/images/input_bench_image.png")
        mask_image = load_image("https://paddlenlp.bj.bcebos.com/data/images/input_bench_mask.png")
        inputs = {
            "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_inpaint_ddim(self):
        pass
        # vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
        # pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-inpainting", safety_checker=None
        # )
        # pipe.vae = vae
        # pipe.set_progress_bar_config(disable=None)
        # pipe.enable_attention_slicing()
        # inputs = self.get_inputs()
        # image = pipe(**inputs).images
        # image_slice = image[(0), 253:256, 253:256, (-1)].flatten()
        # assert image.shape == (1, 512, 512, 3)
        # expected_slice = np.array([0.0521, 0.0606, 0.0602, 0.0446, 0.0495, 0.0434, 0.1175, 0.129, 0.1431])
        # assert np.abs(expected_slice - image_slice).max() < 0.0006

    def test_stable_diffusion_inpaint_fp16(self):
        pass
        # vae = AsymmetricAutoencoderKL.from_pretrained(
        #     "cross-attention/asymmetric-autoencoder-kl-x-1-5", torch_dtype="float16"
        # )
        # pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-inpainting", torch_dtype="float16", safety_checker=None
        # )
        # pipe.vae = vae
        # pipe.set_progress_bar_config(disable=None)
        # pipe.enable_attention_slicing()
        # inputs = self.get_inputs(dtype="float16")
        # image = pipe(**inputs).images
        # image_slice = image[(0), 253:256, 253:256, (-1)].flatten()
        # assert image.shape == (1, 512, 512, 3)
        # expected_slice = np.array([0.1343, 0.1406, 0.144, 0.1504, 0.1729, 0.0989, 0.1807, 0.2822, 0.1179])
        # assert np.abs(expected_slice - image_slice).max() < 0.05

    def test_stable_diffusion_inpaint_pndm(self):
        pass
        # vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
        # pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-inpainting", safety_checker=None
        # )
        # pipe.vae = vae
        # pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        # pipe.set_progress_bar_config(disable=None)
        # pipe.enable_attention_slicing()
        # inputs = self.get_inputs()
        # image = pipe(**inputs).images
        # image_slice = image[(0), 253:256, 253:256, (-1)].flatten()
        # assert image.shape == (1, 512, 512, 3)
        # expected_slice = np.array([0.0976, 0.1071, 0.1119, 0.1363, 0.126, 0.115, 0.3745, 0.3586, 0.334])
        # assert np.abs(expected_slice - image_slice).max() < 0.005

    def test_stable_diffusion_inpaint_k_lms(self):
        pass
        # vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
        # pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-inpainting", safety_checker=None
        # )
        # pipe.vae = vae
        # pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        # pipe.set_progress_bar_config(disable=None)
        # pipe.enable_attention_slicing()
        # inputs = self.get_inputs()
        # image = pipe(**inputs).images
        # image_slice = image[(0), 253:256, 253:256, (-1)].flatten()
        # assert image.shape == (1, 512, 512, 3)
        # expected_slice = np.array([0.8909, 0.862, 0.9024, 0.8501, 0.8558, 0.9074, 0.879, 0.754, 0.9003])
        # assert np.abs(expected_slice - image_slice).max() < 0.006

    def test_stable_diffusion_inpaint_pil_input_resolution_test(self):
        pass
        # vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
        # pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-inpainting", safety_checker=None
        # )
        # pipe.vae = vae
        # pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        # pipe.set_progress_bar_config(disable=None)
        # pipe.enable_attention_slicing()
        # inputs = self.get_inputs()
        # inputs["image"] = inputs["image"].resize((127, 127))
        # inputs["mask_image"] = inputs["mask_image"].resize((127, 127))
        # inputs["height"] = 128
        # inputs["width"] = 128
        # image = pipe(**inputs).images
        # assert image.shape == (1, inputs["height"], inputs["width"], 3)

    def test_stable_diffusion_inpaint_strength_test(self):
        pass
        # vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
        # pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-inpainting", safety_checker=None
        # )
        # pipe.vae = vae
        # pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        # pipe.set_progress_bar_config(disable=None)
        # pipe.enable_attention_slicing()
        # inputs = self.get_inputs()
        # inputs["strength"] = 0.75
        # image = pipe(**inputs).images
        # assert image.shape == (1, 512, 512, 3)
        # image_slice = image[(0), 253:256, 253:256, (-1)].flatten()
        # expected_slice = np.array([0.2458, 0.2576, 0.3124, 0.2679, 0.2669, 0.2796, 0.2872, 0.2975, 0.2661])
        # assert np.abs(expected_slice - image_slice).max() < 0.003

    def test_stable_diffusion_simple_inpaint_ddim(self):
        pass
        # vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
        # pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)
        # pipe.vae = vae
        # pipe.set_progress_bar_config(disable=None)
        # pipe.enable_attention_slicing()
        # inputs = self.get_inputs()
        # image = pipe(**inputs).images
        # image_slice = image[(0), 253:256, 253:256, (-1)].flatten()
        # assert image.shape == (1, 512, 512, 3)
        # expected_slice = np.array([0.3312, 0.4052, 0.4103, 0.4153, 0.4347, 0.4154, 0.4932, 0.492, 0.4431])
        # assert np.abs(expected_slice - image_slice).max() < 0.0006


@nightly
@require_paddle_gpu
class StableDiffusionInpaintPipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        init_image = load_image("https://paddlenlp.bj.bcebos.com/data/images/input_bench_image.png")
        mask_image = load_image("https://paddlenlp.bj.bcebos.com/data/images/input_bench_mask.png")
        inputs = {
            "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_inpaint_ddim(self):
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy("https://paddlenlp.bj.bcebos.com/data/images/stable_diffusion_inpaint_ddim.npy")
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_inpaint_pndm(self):
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        sd_pipe.scheduler = PNDMScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy("https://paddlenlp.bj.bcebos.com/data/images/stable_diffusion_inpaint_pndm.npy")
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_inpaint_lms(self):
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy("https://paddlenlp.bj.bcebos.com/data/images/stable_diffusion_inpaint_lms.npy")
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001

    def test_inpaint_dpm(self):
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(sd_pipe.scheduler.config)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_inputs()
        inputs["num_inference_steps"] = 30
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy(
            "https://paddlenlp.bj.bcebos.com/data/images/stable_diffusion_inpaint_dpm_multi.npy"
        )
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 0.001


class StableDiffusionInpaintingPrepareMaskAndMaskedImageTests(unittest.TestCase):
    def test_pil_inputs(self):
        height, width = 32, 32
        im = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        im = Image.fromarray(im)
        mask = np.random.randint(0, 255, (height, width), dtype=np.uint8) > 127.5
        mask = Image.fromarray((mask * 255).astype(np.uint8))
        t_mask, t_masked, t_image = prepare_mask_and_masked_image(im, mask, height, width, return_image=True)
        self.assertTrue(isinstance(t_mask, paddle.Tensor))
        self.assertTrue(isinstance(t_masked, paddle.Tensor))
        self.assertTrue(isinstance(t_image, paddle.Tensor))
        self.assertEqual(t_mask.ndim, 4)
        self.assertEqual(t_masked.ndim, 4)
        self.assertEqual(t_image.ndim, 4)
        self.assertEqual(t_mask.shape, [1, 1, height, width])
        self.assertEqual(t_masked.shape, [1, 3, height, width])
        self.assertEqual(t_image.shape, [1, 3, height, width])
        self.assertTrue(t_mask.dtype == paddle.float32)
        self.assertTrue(t_masked.dtype == paddle.float32)
        self.assertTrue(t_image.dtype == paddle.float32)
        self.assertTrue(t_mask.min() >= 0.0)
        self.assertTrue(t_mask.max() <= 1.0)
        self.assertTrue(t_masked.min() >= -1.0)
        self.assertTrue(t_masked.min() <= 1.0)
        self.assertTrue(t_image.min() >= -1.0)
        self.assertTrue(t_image.min() >= -1.0)
        self.assertTrue(t_mask.sum() > 0.0)

    def test_np_inputs(self):
        height, width = 32, 32
        im_np = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        im_pil = Image.fromarray(im_np)
        mask_np = np.random.randint(0, 255, (height, width), dtype=np.uint8) > 127.5
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )
        t_mask_pil, t_masked_pil, t_image_pil = prepare_mask_and_masked_image(
            im_pil, mask_pil, height, width, return_image=True
        )
        self.assertTrue((t_mask_np == t_mask_pil).astype("bool").all())
        self.assertTrue((t_masked_np == t_masked_pil).astype("bool").all())
        self.assertTrue((t_image_np == t_image_pil).astype("bool").all())

    def test_paddle_3D_2D_inputs(self):
        height, width = 32, 32
        im_tensor = paddle.randint(0, 255, (3, height, width)).cast("uint8")
        mask_tensor = paddle.randint(0, 255, (height, width)).cast("uint8") > 127.5
        im_np = im_tensor.numpy().transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()
        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor.cast("int64"), height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )
        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_paddle_3D_3D_inputs(self):
        height, width = 32, 32
        im_tensor = paddle.randint(0, 255, (3, height, width)).cast("uint8")
        mask_tensor = paddle.randint(0, 255, (1, height, width)).cast("uint8") > 127.5
        im_np = im_tensor.numpy().transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()[0]
        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor.cast("int64"), height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )
        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_paddle_4D_2D_inputs(self):
        height, width = 32, 32
        im_tensor = paddle.randint(0, 255, (1, 3, height, width)).cast("uint8")
        mask_tensor = paddle.randint(0, 255, (height, width)).cast("uint8") > 127.5
        im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()
        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor.cast("int64"), height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )
        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_paddle_4D_3D_inputs(self):
        height, width = 32, 32
        im_tensor = paddle.randint(0, 255, (1, 3, height, width)).cast("uint8")
        mask_tensor = paddle.randint(0, 255, (1, height, width)).cast("uint8") > 127.5
        im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()[0]
        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor.cast("int64"), height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )
        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_paddle_4D_4D_inputs(self):
        height, width = 32, 32
        im_tensor = paddle.randint(0, 255, (1, 3, height, width)).cast("uint8")
        mask_tensor = paddle.randint(0, 255, (1, 1, height, width)).cast("uint8") > 127.5
        im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()[0][0]
        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor.cast("int64"), height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )
        self.assertTrue((t_mask_tensor == t_mask_np.cast("float64")).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_paddle_batch_4D_3D(self):
        height, width = 32, 32
        im_tensor = paddle.randint(0, 255, (2, 3, height, width)).cast("uint8")
        mask_tensor = paddle.randint(0, 255, (2, height, width)).cast("uint8") > 127.5
        im_nps = [im.numpy().transpose(1, 2, 0) for im in im_tensor]
        mask_nps = [mask.numpy() for mask in mask_tensor]
        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor.cast("int64"), height, width, return_image=True
        )
        nps = [prepare_mask_and_masked_image(i, m, height, width, return_image=True) for i, m in zip(im_nps, mask_nps)]
        t_mask_np = paddle.concat(x=[n[0] for n in nps])
        t_masked_np = paddle.concat(x=[n[1] for n in nps])
        t_image_np = paddle.concat(x=[n[2] for n in nps])
        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_paddle_batch_4D_4D(self):
        height, width = 32, 32
        im_tensor = paddle.randint(0, 255, (2, 3, height, width)).cast("uint8")
        mask_tensor = paddle.randint(0, 255, (2, 1, height, width)).cast("uint8") > 127.5
        im_nps = [im.numpy().transpose(1, 2, 0) for im in im_tensor]
        mask_nps = [mask.numpy()[0] for mask in mask_tensor]
        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor.cast("int64"), height, width, return_image=True
        )
        nps = [prepare_mask_and_masked_image(i, m, height, width, return_image=True) for i, m in zip(im_nps, mask_nps)]
        t_mask_np = paddle.concat(x=[n[0] for n in nps])
        t_masked_np = paddle.concat(x=[n[1] for n in nps])
        t_image_np = paddle.concat(x=[n[2] for n in nps])
        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_shape_mismatch(self):
        height, width = 32, 32
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(
                paddle.randn(shape=[3, height, width]), paddle.randn(shape=[64, 64]), height, width, return_image=True
            )
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(
                paddle.randn(shape=[2, 3, height, width]),
                paddle.randn(shape=[4, 64, 64]),
                height,
                width,
                return_image=True,
            )
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(
                paddle.randn(shape=[2, 3, height, width]),
                paddle.randn(shape=[4, 1, 64, 64]),
                height,
                width,
                return_image=True,
            )

    def test_type_mismatch(self):
        height, width = 32, 32
        with self.assertRaises(TypeError):
            prepare_mask_and_masked_image(
                paddle.rand(shape=[3, height, width]),
                paddle.rand(shape=[3, height, width]).numpy(),
                height,
                width,
                return_image=True,
            )
        with self.assertRaises(TypeError):
            prepare_mask_and_masked_image(
                paddle.rand(shape=[3, height, width]).numpy(),
                paddle.rand(shape=[3, height, width]),
                height,
                width,
                return_image=True,
            )

    def test_channels_first(self):
        height, width = 32, 32
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(
                paddle.rand(shape=[height, width, 3]),
                paddle.rand(shape=[3, height, width]),
                height,
                width,
                return_image=True,
            )

    def test_tensor_range(self):
        height, width = 32, 32
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(
                paddle.ones(shape=[3, height, width]) * 2,
                paddle.rand(shape=[height, width]),
                height,
                width,
                return_image=True,
            )
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(
                paddle.ones(shape=[3, height, width]) * -2,
                paddle.rand(shape=[height, width]),
                height,
                width,
                return_image=True,
            )
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(
                paddle.rand(shape=[3, height, width]),
                paddle.ones(shape=[height, width]) * 2,
                height,
                width,
                return_image=True,
            )
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(
                paddle.rand(shape=[3, height, width]),
                paddle.ones(shape=[height, width]) * -1,
                height,
                width,
                return_image=True,
            )
