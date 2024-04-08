# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 HuggingFace Inc.
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
import traceback
import unittest

import numpy as np
import paddle

# from huggingface_hub import hf_hub_download
from PIL import Image

from ppdiffusers import (  # DDIMScheduler,; AsymmetricAutoencoderKL,
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionInpaintPipeline,
    UNet2DConditionModel,
)

# from ppdiffusers.models.attention_processor import AttnProcessor
from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    prepare_mask_and_masked_image,
)
from ppdiffusers.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers.utils.testing_utils import (
    enable_full_determinism,
    floats_tensor,
    load_image,
    load_numpy,
    nightly,
    require_paddle_gpu,
    require_python39_or_higher,
    run_test_in_subprocess,
    slow,
)

from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_INPAINTING_PARAMS,
    TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS,
)
from ..test_pipelines_common import (
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)

enable_full_determinism()


# Will be run via run_test_in_subprocess
def _test_inpaint_compile(in_queue, out_queue, timeout):
    error = None
    try:
        inputs = in_queue.get(timeout=timeout)
        paddle_device = inputs.pop("paddle_device")
        seed = inputs.pop("seed")
        inputs["generator"] = paddle.Generator(device=paddle_device).manual_seed(seed)

        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.unet.set_default_attn_processor()
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)

        pipe.set_progress_bar_config(disable=None)

        # pipe.unet.to(memory_format=torch.channels_last)
        # pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

        image = pipe(**inputs).images
        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.0689, 0.0699, 0.0790, 0.0536, 0.0470, 0.0488, 0.041, 0.0508, 0.04179])
        assert np.abs(expected_slice - image_slice).max() < 3e-3
    except Exception:
        error = f"{traceback.format_exc()}"

    results = {"error": error}
    out_queue.put(results, timeout=timeout)
    out_queue.join()


class StableDiffusionInpaintPipelineFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionInpaintPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = frozenset([])
    # TO-DO: update image_params once pipeline is refactored with VaeImageProcessor.preprocess
    image_latents_params = frozenset([])
    callback_cfg_params = TEXT_TO_IMAGE_CALLBACK_CFG_PARAMS.union({"mask", "masked_image_latents"})

    def get_dummy_components(self, time_cond_proj_dim=None):
        paddle.seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            time_cond_proj_dim=time_cond_proj_dim,
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

    def get_dummy_inputs(self, device="cuda", seed=0, img_res=64, output_pil=True):
        # TODO: use tensor inputs instead of PIL, this is here just to leave the old expected_slices untouched
        if output_pil:
            # Get random floats in [0, 1] as image
            image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
            image = image.cpu().transpose([0, 2, 3, 1])[0]
            mask_image = paddle.ones_like(image)
            # Convert image and mask_image to [0, 255]
            image = 255 * image
            mask_image = 255 * mask_image
            # Convert to PIL image
            init_image = Image.fromarray(np.uint8(image)).convert("RGB").resize((img_res, img_res))
            mask_image = Image.fromarray(np.uint8(mask_image)).convert("RGB").resize((img_res, img_res))
        else:
            # Get random floats in [0, 1] as image with spatial size (img_res, img_res)
            image = floats_tensor((1, 3, img_res, img_res), rng=random.Random(seed))
            # Convert image to [-1, 1]
            init_image = 2.0 * image - 1.0
            mask_image = paddle.ones((1, 1, img_res, img_res), device=device)

        if str(device).startswith("mps"):
            generator = paddle.seed(seed)
        else:
            generator = paddle.Generator(device=device).manual_seed(seed)

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
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        # expected_slice = np.array([0.4703, 0.5697, 0.3879, 0.5470, 0.6042, 0.4413, 0.5078, 0.4728, 0.4469])
        expected_slice = np.array(
            [0.55786943, 0.628228, 0.49147403, 0.3191774, 0.39249492, 0.46521175, 0.29909956, 0.21160087, 0.42932406]
        )
        expected_slice = np.array(
            [
                [0.5156442, 0.5454932, 0.45407468],
                [0.28159654, 0.37075013, 0.45635986],
                [0.2557804, 0.19107777, 0.41810426],
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice.flatten()).max() < 1e-2

    def test_stable_diffusion_inpaint_lcm(self):
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator
        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        # expected_slice = np.array([0.4931, 0.5988, 0.4569, 0.5556, 0.6650, 0.5087, 0.5966, 0.5358, 0.5269])
        expected_slice = np.array(
            [
                [0.48848608, 0.43804702, 0.44627717],
                [0.44889155, 0.50897574, 0.5054102],
                [0.19799554, 0.22682434, 0.35125494],
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice.flatten()).max() < 1e-2

    def test_stable_diffusion_inpaint_lcm_custom_timesteps(self):
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator
        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        # expected_slice = np.array([0.4931, 0.5988, 0.4569, 0.5556, 0.6650, 0.5087, 0.5966, 0.5358, 0.5269])
        expected_slice = np.array(
            [
                [0.4884862, 0.43804696, 0.446277],
                [0.44889173, 0.50897574, 0.50541013],
                [0.19799575, 0.22682449, 0.351255],
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice.flatten()).max() < 1e-2

    def test_stable_diffusion_inpaint_image_tensor(self):
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        output = sd_pipe(**inputs)
        out_pil = output.images

        inputs = self.get_dummy_inputs(device)
        inputs["image"] = paddle.to_tensor(np.array(inputs["image"]) / 127.5 - 1).transpose([2, 0, 1]).unsqueeze(0)
        inputs["mask_image"] = (
            paddle.to_tensor(np.array(inputs["mask_image"]) / 255).transpose([2, 0, 1])[:1].unsqueeze(0)
        )
        output = sd_pipe(**inputs)
        out_tensor = output.images

        assert out_pil.shape == (1, 64, 64, 3)
        assert np.abs(out_pil.flatten() - out_tensor.flatten()).max() < 5e-2

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=3e-3)

    def test_stable_diffusion_inpaint_strength_zero_test(self):
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)

        # check that the pipeline raises value error when num_inference_steps is < 1
        inputs["strength"] = 0.01
        with self.assertRaises(ValueError):
            sd_pipe(**inputs).images


#    def test_stable_diffusion_inpaint_mask_latents(self):
#        device = "gpu"
#        components = self.get_dummy_components()
#        sd_pipe = self.pipeline_class(**components)
#        sd_pipe.set_progress_bar_config(disable=None)
#
#        # normal mask + normal image
#        #  `image`: pil, `mask_image``: pil, `masked_image_latents``: None
#        inputs = self.get_dummy_inputs(device)
#        inputs["strength"] = 0.9
#        out_0 = sd_pipe(**inputs).images
#
#        # image latents + mask latents
#        inputs = self.get_dummy_inputs(device)
#        image = sd_pipe.image_processor.preprocess(inputs["image"])
#        mask = sd_pipe.mask_processor.preprocess(inputs["mask_image"])
#        masked_image = image * (mask < 0.5)
#
#        generator = paddle.Generator(device=device).manual_seed(0)
#        image_latents = (
#            sd_pipe.vae.encode(image).latent_dist.sample(generator=generator) * sd_pipe.vae.config.scaling_factor
#        )
#        paddle.randn((1, 4, 32, 32))
#        mask_latents = (
#            sd_pipe.vae.encode(masked_image).latent_dist.sample(generator=generator)
#            * sd_pipe.vae.config.scaling_factor
#        )
#        inputs["image"] = image_latents
#        inputs["masked_image_latents"] = mask_latents
#        inputs["mask_image"] = mask
#        inputs["strength"] = 0.9
#        generator = paddle.Generator(device=device).manual_seed(0)
#        paddle.randn((1, 4, 32, 32))
#        inputs["generator"] = generator
#        out_1 = sd_pipe(**inputs).images
#        assert np.abs(out_0 - out_1).max() < 1e-2


class StableDiffusionSimpleInpaintPipelineFastTests(StableDiffusionInpaintPipelineFastTests):
    pipeline_class = StableDiffusionInpaintPipeline
    params = TEXT_GUIDED_IMAGE_INPAINTING_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_INPAINTING_BATCH_PARAMS
    image_params = frozenset([])
    # TO-DO: update image_params once pipeline is refactored with VaeImageProcessor.preprocess

    def get_dummy_components(self, time_cond_proj_dim=None):
        paddle.seed(0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            time_cond_proj_dim=time_cond_proj_dim,
            sample_size=32,
            in_channels=4,
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

    def get_dummy_inputs_2images(self, device="gpu", seed=0, img_res=64):
        # Get random floats in [0, 1] as image with spatial size (img_res, img_res)
        image1 = floats_tensor((1, 3, img_res, img_res), rng=random.Random(seed))
        image2 = floats_tensor((1, 3, img_res, img_res), rng=random.Random(seed + 22))
        # Convert images to [-1, 1]
        init_image1 = 2.0 * image1 - 1.0
        init_image2 = 2.0 * image2 - 1.0

        # empty mask
        mask_image = paddle.zeros((1, 1, img_res, img_res))

        if str(device).startswith("mps"):
            generator1 = paddle.seed(seed)
            generator2 = paddle.seed(seed)
        else:
            generator1 = paddle.Generator(device=device).manual_seed(seed)
            generator2 = paddle.Generator(device=device).manual_seed(seed)

        inputs = {
            "prompt": ["A painting of a squirrel eating a burger"] * 2,
            "image": [init_image1, init_image2],
            "mask_image": [mask_image] * 2,
            "generator": [generator1, generator2],
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
        }
        return inputs

    def test_stable_diffusion_inpaint(self):
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionInpaintPipeline(**components)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        # expected_slice = np.array([0.6584, 0.5424, 0.5649, 0.5449, 0.5897, 0.6111, 0.5404, 0.5463, 0.5214])
        # expected_slice = np.array([0.3216, 0.2463, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2595, 0.4763])
        expected_slice = np.array(
            [
                [0.26244918, 0.11772025, 0.23931459],
                [0.6366332, 0.07551852, 0.4727996],
                [0.6000204, 0.34627506, 0.51476544],
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice.flatten()).max() < 1e-2

    def test_stable_diffusion_inpaint_lcm(self):
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator
        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        # expected_slice = np.array([0.6240, 0.5355, 0.5649, 0.5378, 0.5374, 0.6242, 0.5132, 0.5347, 0.5396])
        expected_slice = np.array(
            [0.52281016, 0.54583037, 0.401865, 0.67171454, 0.34123725, 0.4368565, 0.5224986, 0.49829865, 0.5517228]
        )

        assert np.abs(image_slice.flatten() - expected_slice.flatten()).max() < 1e-2

    def test_stable_diffusion_inpaint_lcm_custom_timesteps(self):
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator
        components = self.get_dummy_components(time_cond_proj_dim=256)
        sd_pipe = StableDiffusionInpaintPipeline(**components)
        sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs(device)
        del inputs["num_inference_steps"]
        inputs["timesteps"] = [999, 499]
        image = sd_pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 64, 64, 3)
        # expected_slice = np.array([0.6240, 0.5355, 0.5649, 0.5378, 0.5374, 0.6242, 0.5132, 0.5347, 0.5396])
        expected_slice = np.array(
            [
                [0.5228102, 0.54583055, 0.40186495],
                [0.67171466, 0.3412371, 0.43685645],
                [0.52249867, 0.4982987, 0.5517229],
            ]
        )

        assert np.abs(image_slice.flatten() - expected_slice.flatten()).max() < 1e-2

    def test_stable_diffusion_inpaint_2_images(self):
        device = "cpu"  # ensure determinism for the device-dependent paddle.Generator
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)

        sd_pipe.set_progress_bar_config(disable=None)

        # test to confirm if we pass two same image, we will get same output
        inputs = self.get_dummy_inputs(device)
        gen1 = paddle.Generator(device=device).manual_seed(0)
        gen2 = paddle.Generator(device=device).manual_seed(0)
        for name in ["prompt", "image", "mask_image"]:
            inputs[name] = [inputs[name]] * 2
        inputs["generator"] = [gen1, gen2]
        images = sd_pipe(**inputs).images

        assert images.shape == (2, 64, 64, 3)

        image_slice1 = images[0, -3:, -3:, -1]
        image_slice2 = images[1, -3:, -3:, -1]
        assert np.abs(image_slice1.flatten() - image_slice2.flatten()).max() < 1e-4

        # test to confirm that if we pass two different images, we will get different output
        inputs = self.get_dummy_inputs_2images(device)
        images = sd_pipe(**inputs).images
        assert images.shape == (2, 64, 64, 3)

        image_slice1 = images[0, -3:, -3:, -1]
        image_slice2 = images[1, -3:, -3:, -1]
        assert np.abs(image_slice1.flatten() - image_slice2.flatten()).max() > 1e-2


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
        # expected_slice = np.array([0.0427, 0.0460, 0.0483, 0.0460, 0.0584, 0.0521, 0.1549, 0.1695, 0.1794])
        expected_slice = np.array([0.05978, 0.10983, 0.10514, 0.07922, 0.08483, 0.08587, 0.05302, 0.03218, 0.01636])

        assert np.abs(expected_slice - image_slice).max() < 6e-4

    def test_stable_diffusion_inpaint_fp16(self):
        pass

    #        pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #            "runwayml/stable-diffusion-inpainting", paddle_dtype=paddle.float16, safety_checker=None
    #        )
    #        pipe.unet.set_default_attn_processor()
    #
    #        pipe.set_progress_bar_config(disable=None)
    #        pipe.enable_attention_slicing()
    #
    #        inputs = self.get_inputs(dtype=paddle.float16)
    #        image = pipe(**inputs).images
    #        image_slice = image[0, 253:256, 253:256, -1].flatten()
    #
    #        assert image.shape == (1, 512, 512, 3)
    #        expected_slice = np.array([0.1509, 0.1245, 0.1672, 0.1655, 0.1519, 0.1226, 0.1462, 0.1567, 0.2451])
    #        assert np.abs(expected_slice - image_slice).max() < 1e-1

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
        # expected_slice = np.array([0.0425, 0.0273, 0.0344, 0.1694, 0.1727, 0.1812, 0.3256, 0.3311, 0.3272])
        expected_slice = np.array([0.06892, 0.06994, 0.07905, 0.05366, 0.04709, 0.04890, 0.04107, 0.05083, 0.04180])

        assert np.abs(expected_slice - image_slice).max() < 5e-3

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
        # expected_slice = np.array([0.9314, 0.7575, 0.9432, 0.8885, 0.9028, 0.7298, 0.9811, 0.9667, 0.7633])
        expected_slice = np.array([0.23513, 0.22413, 0.29442, 0.24243, 0.26214, 0.30329, 0.26431, 0.25025, 0.25197])

        assert np.abs(expected_slice - image_slice).max() < 6e-3

    # def test_stable_diffusion_inpaint_with_sequential_cpu_offloading(self):
    #     paddle.device.cuda.empty_cache()
    #     torch.cuda.reset_max_memory_allocated()
    #     torch.cuda.reset_peak_memory_stats()

    #     pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #         "runwayml/stable-diffusion-inpainting", safety_checker=None, paddle_dtype=paddle.float16
    #     )
    #     pipe = pipe.to(paddle_device)
    #     pipe.set_progress_bar_config(disable=None)
    #     pipe.enable_attention_slicing(1)
    #     pipe.enable_sequential_cpu_offload()

    #     inputs = self.get_inputs(dtype=paddle.float16)
    #     _ = pipe(**inputs)

    #     mem_bytes = torch.cuda.max_memory_allocated()
    #     # make sure that less than 2.2 GB is allocated
    #     assert mem_bytes < 2.2 * 10**9

    @require_python39_or_higher
    def test_inpaint_compile(self):
        seed = 0
        inputs = self.get_inputs(seed=seed)
        # Can't pickle a Generator object
        del inputs["generator"]
        inputs["paddle_device"] = "cuda"
        inputs["seed"] = seed
        run_test_in_subprocess(test_case=self, target_func=_test_inpaint_compile, inputs=inputs)

    def test_stable_diffusion_inpaint_pil_input_resolution_test(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        # change input image to a random size (one that would cause a tensor mismatch error)
        inputs["image"] = inputs["image"].resize((127, 127))
        inputs["mask_image"] = inputs["mask_image"].resize((127, 127))
        inputs["height"] = 128
        inputs["width"] = 128
        image = pipe(**inputs).images
        # verify that the returned image has the same height and width as the input height and width
        assert image.shape == (1, inputs["height"], inputs["width"], 3)

    def test_stable_diffusion_inpaint_strength_test(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", safety_checker=None
        )
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.unet.set_default_attn_processor()

        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        # change input strength
        inputs["strength"] = 0.75
        image = pipe(**inputs).images
        # verify that the returned image has the same height and width as the input height and width
        assert image.shape == (1, 512, 512, 3)

        image_slice = image[0, 253:256, 253:256, -1].flatten()
        # expected_slice = np.array([0.2728, 0.2803, 0.2665, 0.2511, 0.2774, 0.2586, 0.2391, 0.2392, 0.2582])
        expected_slice = np.array([0.0972, 0.2068, 0.1759, 0.1449, 0.1885, 0.1117, 0.1314, 0.1382, 0.0856])
        assert np.abs(expected_slice - image_slice).max() < 1e-3

    def test_stable_diffusion_simple_inpaint_ddim(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)
        pipe.unet.set_default_attn_processor()

        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()

        inputs = self.get_inputs()
        image = pipe(**inputs).images

        image_slice = image[0, 253:256, 253:256, -1].flatten()

        assert image.shape == (1, 512, 512, 3)
        # expected_slice = np.array([0.3757, 0.3875, 0.4445, 0.4353, 0.3780, 0.4513, 0.3965, 0.3984, 0.4362])
        # expected_slice = np.array([0.2528, 0.2133, 0.3194, 0.2348, 0.1933, 0.332, 0.2186, 0.1664, 0.3176])
        expected_slice = np.array(
            [
                0.11896983,
                0.11297423,
                0.11790517,
                0.14523566,
                0.11610284,
                0.09858531,
                0.10270301,
                0.08899242,
                0.11605743,
            ]
        )
        assert np.abs(expected_slice - image_slice).max() < 1e-3


#    def test_download_local(self):
#        filename = hf_hub_download("runwayml/stable-diffusion-inpainting", filename="sd-v1-5-inpainting.ckpt")
#
#        pipe = StableDiffusionInpaintPipeline.from_single_file(filename, paddle_dtype=paddle.float16)
#        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
#        pipe.to("cuda")
#
#        inputs = self.get_inputs()
#        inputs["num_inference_steps"] = 1
#        image_out = pipe(**inputs).images[0]
#
#        assert image_out.shape == (512, 512, 3)
#
#    def test_download_ckpt_diff_format_is_same(self):
#        ckpt_path = "https://huggingface.co/runwayml/stable-diffusion-inpainting/blob/main/sd-v1-5-inpainting.ckpt"
#
#        pipe = StableDiffusionInpaintPipeline.from_single_file(ckpt_path)
#        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
#        pipe.unet.set_attn_processor(AttnProcessor())
#        pipe.to("cuda")
#
#        inputs = self.get_inputs()
#        inputs["num_inference_steps"] = 5
#        image_ckpt = pipe(**inputs).images[0]
#
#        pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
#        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
#        pipe.unet.set_attn_processor(AttnProcessor())
#        pipe.to("cuda")
#
#        inputs = self.get_inputs()
#        inputs["num_inference_steps"] = 5
#        image = pipe(**inputs).images[0]
#
#        assert np.max(np.abs(image - image_ckpt)) < 5e-4


@slow
@require_paddle_gpu
class StableDiffusionInpaintPipelineAsymmetricAutoencoderKLSlowTests(unittest.TestCase):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, generator_device="gpu", dtype=paddle.float32, seed=0):
        generator = paddle.Generator(device=generator_device).manual_seed(seed)
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

    #
    #    def test_stable_diffusion_inpaint_ddim(self):
    #        vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
    #        pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #            "runwayml/stable-diffusion-inpainting", safety_checker=None
    #        )
    #        pipe.vae = vae
    #        pipe.unet.set_default_attn_processor()
    #
    #        pipe.set_progress_bar_config(disable=None)
    #        pipe.enable_attention_slicing()
    #
    #        inputs = self.get_inputs()
    #        image = pipe(**inputs).images
    #        image_slice = image[0, 253:256, 253:256, -1].flatten()
    #
    #        assert image.shape == (1, 512, 512, 3)
    #        expected_slice = np.array([0.0522, 0.0604, 0.0596, 0.0449, 0.0493, 0.0427, 0.1186, 0.1289, 0.1442])
    #
    #        assert np.abs(expected_slice - image_slice).max() < 1e-3
    #
    #    def test_stable_diffusion_inpaint_fp16(self):
    #        vae = AsymmetricAutoencoderKL.from_pretrained(
    #            "cross-attention/asymmetric-autoencoder-kl-x-1-5", paddle_dtype=paddle.float16
    #        )
    #        pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #            "runwayml/stable-diffusion-inpainting", paddle_dtype=paddle.float16, safety_checker=None
    #        )
    #        pipe.unet.set_default_attn_processor()
    #        pipe.vae = vae
    #
    #        pipe.set_progress_bar_config(disable=None)
    #        pipe.enable_attention_slicing()
    #
    #        inputs = self.get_inputs(dtype=paddle.float16)
    #        image = pipe(**inputs).images
    #        image_slice = image[0, 253:256, 253:256, -1].flatten()
    #
    #        assert image.shape == (1, 512, 512, 3)
    #        expected_slice = np.array([0.1343, 0.1406, 0.1440, 0.1504, 0.1729, 0.0989, 0.1807, 0.2822, 0.1179])
    #
    #        assert np.abs(expected_slice - image_slice).max() < 5e-2
    #
    #    def test_stable_diffusion_inpaint_pndm(self):
    #        vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
    #        pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #            "runwayml/stable-diffusion-inpainting", safety_checker=None
    #        )
    #        pipe.unet.set_default_attn_processor()
    #        pipe.vae = vae
    #        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    #
    #        pipe.set_progress_bar_config(disable=None)
    #        pipe.enable_attention_slicing()
    #
    #        inputs = self.get_inputs()
    #        image = pipe(**inputs).images
    #        image_slice = image[0, 253:256, 253:256, -1].flatten()
    #
    #        assert image.shape == (1, 512, 512, 3)
    #        expected_slice = np.array([0.0966, 0.1083, 0.1148, 0.1422, 0.1318, 0.1197, 0.3702, 0.3537, 0.3288])
    #
    #        assert np.abs(expected_slice - image_slice).max() < 5e-3
    #
    #    def test_stable_diffusion_inpaint_k_lms(self):
    #        vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
    #        pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #            "runwayml/stable-diffusion-inpainting", safety_checker=None
    #        )
    #        pipe.unet.set_default_attn_processor()
    #        pipe.vae = vae
    #        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    #
    #        pipe.set_progress_bar_config(disable=None)
    #        pipe.enable_attention_slicing()
    #
    #        inputs = self.get_inputs()
    #        image = pipe(**inputs).images
    #        image_slice = image[0, 253:256, 253:256, -1].flatten()
    #        assert image.shape == (1, 512, 512, 3)
    #        expected_slice = np.array([0.8931, 0.8683, 0.8965, 0.8501, 0.8592, 0.9118, 0.8734, 0.7463, 0.8990])
    #        assert np.abs(expected_slice - image_slice).max() < 6e-3

    # def test_stable_diffusion_inpaint_with_sequential_cpu_offloading(self):
    #     paddle.device.cuda.empty_cache()
    #     # torch.cuda.reset_max_memory_allocated()
    #     # torch.cuda.reset_peak_memory_stats()

    #     vae = AsymmetricAutoencoderKL.from_pretrained(
    #         "cross-attention/asymmetric-autoencoder-kl-x-1-5", paddle_dtype=paddle.float16
    #     )
    #     pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #         "runwayml/stable-diffusion-inpainting", safety_checker=None, paddle_dtype=paddle.float16
    #     )
    #     pipe.vae = vae
    #     pipe = pipe.to(paddle_device)
    #     pipe.set_progress_bar_config(disable=None)
    #     pipe.enable_attention_slicing(1)
    #     pipe.enable_sequential_cpu_offload()

    #     inputs = self.get_inputs(dtype=paddle.float16)
    #     _ = pipe(**inputs)

    #     mem_bytes = torch.cuda.max_memory_allocated()
    #     # make sure that less than 2.45 GB is allocated
    #     assert mem_bytes < 2.45 * 10**9

    @require_python39_or_higher
    def test_inpaint_compile(self):
        pass

    #    def test_stable_diffusion_inpaint_pil_input_resolution_test(self):
    #        vae = AsymmetricAutoencoderKL.from_pretrained(
    #            "cross-attention/asymmetric-autoencoder-kl-x-1-5",
    #        )
    #        pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #            "runwayml/stable-diffusion-inpainting", safety_checker=None
    #        )
    #        pipe.vae = vae
    #        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    #
    #        pipe.set_progress_bar_config(disable=None)
    #        pipe.enable_attention_slicing()
    #
    #        inputs = self.get_inputs()
    #        # change input image to a random size (one that would cause a tensor mismatch error)
    #        inputs["image"] = inputs["image"].resize((127, 127))
    #        inputs["mask_image"] = inputs["mask_image"].resize((127, 127))
    #        inputs["height"] = 128
    #        inputs["width"] = 128
    #        image = pipe(**inputs).images
    #        # verify that the returned image has the same height and width as the input height and width
    #        assert image.shape == (1, inputs["height"], inputs["width"], 3)
    #
    #    def test_stable_diffusion_inpaint_strength_test(self):
    #        vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
    #        pipe = StableDiffusionInpaintPipeline.from_pretrained(
    #            "runwayml/stable-diffusion-inpainting", safety_checker=None
    #        )
    #        pipe.unet.set_default_attn_processor()
    #        pipe.vae = vae
    #        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    #
    #        pipe.set_progress_bar_config(disable=None)
    #        pipe.enable_attention_slicing()
    #
    #        inputs = self.get_inputs()
    #        # change input strength
    #        inputs["strength"] = 0.75
    #        image = pipe(**inputs).images
    #        # verify that the returned image has the same height and width as the input height and width
    #        assert image.shape == (1, 512, 512, 3)
    #
    #        image_slice = image[0, 253:256, 253:256, -1].flatten()
    #        expected_slice = np.array([0.2458, 0.2576, 0.3124, 0.2679, 0.2669, 0.2796, 0.2872, 0.2975, 0.2661])
    #        assert np.abs(expected_slice - image_slice).max() < 3e-3
    #
    #    def test_stable_diffusion_simple_inpaint_ddim(self):
    #        vae = AsymmetricAutoencoderKL.from_pretrained("cross-attention/asymmetric-autoencoder-kl-x-1-5")
    #        pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None)
    #        pipe.vae = vae
    #        pipe.unet.set_default_attn_processor()
    #
    #        pipe.set_progress_bar_config(disable=None)
    #        pipe.enable_attention_slicing()
    #
    #        inputs = self.get_inputs()
    #        image = pipe(**inputs).images
    #
    #        image_slice = image[0, 253:256, 253:256, -1].flatten()
    #
    #        assert image.shape == (1, 512, 512, 3)
    #        expected_slice = np.array([0.3296, 0.4041, 0.4097, 0.4145, 0.4342, 0.4152, 0.4927, 0.4931, 0.4430])
    #        assert np.abs(expected_slice - image_slice).max() < 1e-3

    #    def test_download_local(self):
    #        vae = AsymmetricAutoencoderKL.from_pretrained(
    #            "cross-attention/asymmetric-autoencoder-kl-x-1-5", paddle_dtype=paddle.float16
    #        )
    #        filename = hf_hub_download("runwayml/stable-diffusion-inpainting", filename="sd-v1-5-inpainting.ckpt")
    #
    #        pipe = StableDiffusionInpaintPipeline.from_single_file(filename, paddle_dtype=paddle.float16)
    #        pipe.vae = vae
    #        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    #        pipe.to("cuda")
    #
    #        inputs = self.get_inputs()
    #        inputs["num_inference_steps"] = 1
    #        image_out = pipe(**inputs).images[0]
    #
    #        assert image_out.shape == (512, 512, 3)

    def test_download_ckpt_diff_format_is_same(self):
        pass


@nightly
class StableDiffusionInpaintPipelineNightlyTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def get_inputs(self, device="gpu", generator_device="gpu", dtype=paddle.float32, seed=0):
        generator = paddle.Generator(device=generator_device).manual_seed(seed)
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
        assert max_diff < 1e-3

    def test_inpaint_pndm(self):
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        sd_pipe.scheduler = PNDMScheduler.from_config(sd_pipe.scheduler.config)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy("https://paddlenlp.bj.bcebos.com/data/images/stable_diffusion_inpaint_pndm.npy")
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

    def test_inpaint_lms(self):
        sd_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
        sd_pipe.scheduler = LMSDiscreteScheduler.from_config(sd_pipe.scheduler.config)

        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_inputs()
        image = sd_pipe(**inputs).images[0]
        expected_image = load_numpy("https://paddlenlp.bj.bcebos.com/data/images/stable_diffusion_inpaint_lms.npy")
        max_diff = np.abs(expected_image - image).max()
        assert max_diff < 1e-3

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
        assert max_diff < 1e-3


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

        self.assertEqual(tuple(t_mask.shape), (1, 1, height, width))
        self.assertEqual(tuple(t_masked.shape), (1, 3, height, width))
        self.assertEqual(tuple(t_image.shape), (1, 3, height, width))

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
        mask_np = (
            np.random.randint(
                0,
                255,
                (
                    height,
                    width,
                ),
                dtype=np.uint8,
            )
            > 127.5
        )
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))

        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )
        t_mask_pil, t_masked_pil, t_image_pil = prepare_mask_and_masked_image(
            im_pil, mask_pil, height, width, return_image=True
        )

        self.assertTrue((t_mask_np == t_mask_pil).all())
        self.assertTrue((t_masked_np == t_masked_pil).all())
        self.assertTrue((t_image_np == t_image_pil).all())

    def test_paddle_3D_2D_inputs(self):
        height, width = 32, 32

        im_tensor = paddle.randint(
            0,
            255,
            (
                3,
                height,
                width,
            ),
        ).cast(dtype=paddle.uint8)
        mask_tensor = (
            paddle.randint(
                0,
                255,
                (
                    height,
                    width,
                ),
            )
            > 127.5
        ).cast(dtype=paddle.int32)
        im_np = im_tensor.numpy().transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_paddle_3D_3D_inputs(self):
        height, width = 32, 32

        im_tensor = paddle.randint(
            0,
            255,
            (
                3,
                height,
                width,
            ),
        ).cast(dtype=paddle.uint8)
        mask_tensor = (
            paddle.randint(
                0,
                255,
                (
                    1,
                    height,
                    width,
                ),
            ).cast(dtype=paddle.uint8)
            > 127.5
        ).cast(dtype=paddle.int32)
        im_np = im_tensor.numpy().transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()[0]

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_paddle_4D_2D_inputs(self):
        height, width = 32, 32

        im_tensor = paddle.randint(
            0,
            255,
            (
                1,
                3,
                height,
                width,
            ),
        ).cast(dtype=paddle.uint8)
        mask_tensor = (
            paddle.randint(
                0,
                255,
                (
                    height,
                    width,
                ),
            ).cast(dtype=paddle.uint8)
            > 127.5
        ).cast(dtype=paddle.int32)
        im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_paddle_4D_3D_inputs(self):
        height, width = 32, 32

        im_tensor = paddle.randint(
            0,
            255,
            (
                1,
                3,
                height,
                width,
            ),
        ).cast(dtype=paddle.uint8)
        mask_tensor = (
            paddle.randint(
                0,
                255,
                (
                    1,
                    height,
                    width,
                ),
            ).cast(dtype=paddle.uint8)
            > 127.5
        ).cast(dtype=paddle.int32)
        im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()[0]

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_paddle_4D_4D_inputs(self):
        height, width = 32, 32

        im_tensor = paddle.randint(
            0,
            255,
            (
                1,
                3,
                height,
                width,
            ),
        ).cast(dtype=paddle.uint8)
        mask_tensor = (
            paddle.randint(
                0,
                255,
                (
                    1,
                    1,
                    height,
                    width,
                ),
            ).cast(dtype=paddle.uint8)
            > 127.5
        ).cast(dtype=paddle.int32)
        im_np = im_tensor.numpy()[0].transpose(1, 2, 0)
        mask_np = mask_tensor.numpy()[0][0]

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        t_mask_np, t_masked_np, t_image_np = prepare_mask_and_masked_image(
            im_np, mask_np, height, width, return_image=True
        )

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_paddle_batch_4D_3D(self):
        height, width = 32, 32

        im_tensor = paddle.randint(
            0,
            255,
            (
                2,
                3,
                height,
                width,
            ),
        ).cast(dtype=paddle.uint8)
        mask_tensor = (
            paddle.randint(
                0,
                255,
                (
                    2,
                    height,
                    width,
                ),
            ).cast(dtype=paddle.uint8)
            > 127.5
        ).cast(dtype=paddle.int32)

        im_nps = [im.numpy().transpose(1, 2, 0) for im in im_tensor]
        mask_nps = [mask.numpy() for mask in mask_tensor]

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        nps = [prepare_mask_and_masked_image(i, m, height, width, return_image=True) for i, m in zip(im_nps, mask_nps)]
        t_mask_np = paddle.concat([n[0] for n in nps])
        t_masked_np = paddle.concat([n[1] for n in nps])
        t_image_np = paddle.concat([n[2] for n in nps])

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_paddle_batch_4D_4D(self):
        height, width = 32, 32

        im_tensor = paddle.randint(
            0,
            255,
            (
                2,
                3,
                height,
                width,
            ),
        ).cast(dtype=paddle.uint8)
        mask_tensor = (
            paddle.randint(
                0,
                255,
                (
                    2,
                    1,
                    height,
                    width,
                ),
            ).cast(dtype=paddle.uint8)
            > 127.5
        ).cast(dtype=paddle.int32)

        im_nps = [im.numpy().transpose(1, 2, 0) for im in im_tensor]
        mask_nps = [mask.numpy()[0] for mask in mask_tensor]

        t_mask_tensor, t_masked_tensor, t_image_tensor = prepare_mask_and_masked_image(
            im_tensor / 127.5 - 1, mask_tensor, height, width, return_image=True
        )
        nps = [prepare_mask_and_masked_image(i, m, height, width, return_image=True) for i, m in zip(im_nps, mask_nps)]
        t_mask_np = paddle.concat([n[0] for n in nps])
        t_masked_np = paddle.concat([n[1] for n in nps])
        t_image_np = paddle.concat([n[2] for n in nps])

        self.assertTrue((t_mask_tensor == t_mask_np).all())
        self.assertTrue((t_masked_tensor == t_masked_np).all())
        self.assertTrue((t_image_tensor == t_image_np).all())

    def test_shape_mismatch(self):
        height, width = 32, 32

        # test height and width
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(
                paddle.randn(
                    (
                        3,
                        height,
                        width,
                    )
                ),
                paddle.randn((64, 64)),
                height,
                width,
                return_image=True,
            )
        # test batch dim
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(
                paddle.randn(
                    (
                        2,
                        3,
                        height,
                        width,
                    )
                ),
                paddle.randn((4, 64, 64)),
                height,
                width,
                return_image=True,
            )
        # test batch dim
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(
                paddle.randn(
                    (
                        2,
                        3,
                        height,
                        width,
                    )
                ),
                paddle.randn((4, 1, 64, 64)),
                height,
                width,
                return_image=True,
            )

    def test_type_mismatch(self):
        height, width = 32, 32

        # test tensors-only
        with self.assertRaises(TypeError):
            prepare_mask_and_masked_image(
                paddle.rand(
                    3,
                    height,
                    width,
                ),
                paddle.rand(
                    3,
                    height,
                    width,
                ).numpy(),
                height,
                width,
                return_image=True,
            )
        # test tensors-only
        with self.assertRaises(TypeError):
            prepare_mask_and_masked_image(
                paddle.rand(
                    3,
                    height,
                    width,
                ).numpy(),
                paddle.rand(
                    3,
                    height,
                    width,
                ),
                height,
                width,
                return_image=True,
            )

    def test_channels_first(self):
        height, width = 32, 32

        # test channels first for 3D tensors
        with self.assertRaises(AssertionError):
            prepare_mask_and_masked_image(
                paddle.rand((height, width, 3)),
                paddle.rand(
                    (
                        3,
                        height,
                        width,
                    )
                ),
                height,
                width,
                return_image=True,
            )

    def test_tensor_range(self):
        height, width = 32, 32

        # test im <= 1
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(
                paddle.ones(
                    (
                        3,
                        height,
                        width,
                    )
                )
                * 2,
                paddle.rand(
                    (
                        height,
                        width,
                    )
                ),
                height,
                width,
                return_image=True,
            )
        # test im >= -1
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(
                paddle.ones(
                    (
                        3,
                        height,
                        width,
                    )
                )
                * (-2),
                paddle.rand(
                    (
                        height,
                        width,
                    )
                ),
                height,
                width,
                return_image=True,
            )
        # test mask <= 1
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(
                paddle.rand(
                    (
                        3,
                        height,
                        width,
                    )
                ),
                paddle.ones(
                    (
                        height,
                        width,
                    )
                )
                * 2,
                height,
                width,
                return_image=True,
            )
        # test mask >= 0
        with self.assertRaises(ValueError):
            prepare_mask_and_masked_image(
                paddle.rand(
                    (
                        3,
                        height,
                        width,
                    )
                ),
                paddle.ones(
                    (
                        height,
                        width,
                    )
                )
                * -1,
                height,
                width,
                return_image=True,
            )
