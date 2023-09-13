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
import tempfile
import unittest

import numpy as np
import paddle
from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from ppdiffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    EulerDiscreteScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import (
    MultiControlNetModel,
)
from ppdiffusers.utils import load_image, load_numpy, paddle_device, randn_tensor, slow
from ppdiffusers.utils.import_utils import is_ppxformers_available
from ppdiffusers.utils.testing_utils import enable_full_determinism, require_paddle_gpu

from ..pipeline_params import (
    IMAGE_TO_IMAGE_IMAGE_PARAMS,
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


class ControlNetPipelineFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionControlNetPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
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
        paddle.seed(seed=0)
        controlnet = ControlNetModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
        )
        paddle.seed(seed=0)
        scheduler = DDIMScheduler(
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
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = paddle.seed(seed=seed)
        else:
            generator = paddle.framework.core.default_cpu_generator().manual_seed(seed)
        controlnet_embedder_scale_factor = 2
        image = randn_tensor(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
            generator=generator,
            device=str(device).replace("cuda", "gpu"),
        )
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
            "image": image,
        }
        return inputs

    def test_attention_slicing_forward_pass(self):
        return self._test_attention_slicing_forward_pass(expected_max_diff=0.002)

    @unittest.skipIf(
        not is_ppxformers_available(), reason="XFormers attention is only available with CUDA and `xformers` installed"
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=0.002)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=0.002)


class StableDiffusionMultiControlNetPipelineFastTests(
    PipelineTesterMixin, PipelineKarrasSchedulerTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionControlNetPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
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
        paddle.seed(seed=0)

        def init_weights(m):
            if isinstance(m, paddle.nn.Conv2D):
                paddle.nn.initializer.normal(m.weight)
                m.bias.data.fill_(1.0)

        controlnet1 = ControlNetModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
        )
        controlnet1.controlnet_down_blocks.apply(init_weights)
        paddle.seed(seed=0)
        controlnet2 = ControlNetModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
        )
        controlnet2.controlnet_down_blocks.apply(init_weights)
        paddle.seed(seed=0)
        scheduler = DDIMScheduler(
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
        controlnet = MultiControlNetModel([controlnet1, controlnet2])
        components = {
            "unet": unet,
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = paddle.seed(seed=seed)
        else:
            generator = paddle.framework.core.default_cpu_generator().manual_seed(seed)
        controlnet_embedder_scale_factor = 2
        images = [
            randn_tensor(
                (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
                generator=generator,
                device=str(device).replace("cuda", "gpu"),
            ),
            randn_tensor(
                (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
                generator=generator,
                device=str(device).replace("cuda", "gpu"),
            ),
        ]
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
            "image": images,
        }
        return inputs

    def test_control_guidance_switch(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(paddle_device)
        scale = 10.0
        steps = 4
        inputs = self.get_dummy_inputs(paddle_device)
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_1 = pipe(**inputs)[0]
        inputs = self.get_dummy_inputs(paddle_device)
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_2 = pipe(**inputs, control_guidance_start=0.1, control_guidance_end=0.2)[0]
        inputs = self.get_dummy_inputs(paddle_device)
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_3 = pipe(**inputs, control_guidance_start=[0.1, 0.3], control_guidance_end=[0.2, 0.7])[0]
        inputs = self.get_dummy_inputs(paddle_device)
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_4 = pipe(**inputs, control_guidance_start=0.4, control_guidance_end=[0.5, 0.8])[0]
        assert np.sum(np.abs(output_1 - output_2)) > 0.001
        assert np.sum(np.abs(output_1 - output_3)) > 0.001
        assert np.sum(np.abs(output_1 - output_4)) > 0.001

    def test_attention_slicing_forward_pass(self):
        return self._test_attention_slicing_forward_pass(expected_max_diff=0.002)

    @unittest.skipIf(
        not is_ppxformers_available(), reason="XFormers attention is only available with CUDA and `xformers` installed"
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=0.002)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=0.002)

    def test_save_pretrained_raise_not_implemented_exception(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                pipe.save_pretrained(tmpdir)
            except NotImplementedError:
                pass


class StableDiffusionMultiControlNetOneModelPipelineFastTests(
    PipelineTesterMixin, PipelineKarrasSchedulerTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionControlNetPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
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
        paddle.seed(seed=0)

        def init_weights(m):
            if isinstance(m, paddle.nn.Conv2D):
                paddle.nn.initializer.normal(m.weight)
                m.bias.data.fill_(1.0)

        controlnet = ControlNetModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
        )
        controlnet.controlnet_down_blocks.apply(init_weights)
        paddle.seed(seed=0)
        scheduler = DDIMScheduler(
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
        controlnet = MultiControlNetModel([controlnet])
        components = {
            "unet": unet,
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "safety_checker": None,
            "feature_extractor": None,
        }
        return components

    def get_dummy_inputs(self, device, seed=0):
        if str(device).startswith("mps"):
            generator = paddle.seed(seed=seed)
        else:
            generator = paddle.framework.core.default_cpu_generator().manual_seed(seed)
        controlnet_embedder_scale_factor = 2
        images = [
            randn_tensor(
                (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
                generator=generator,
                device=str(device).replace("cuda", "gpu"),
            )
        ]
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "numpy",
            "image": images,
        }
        return inputs

    def test_control_guidance_switch(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(paddle_device)
        scale = 10.0
        steps = 4
        inputs = self.get_dummy_inputs(paddle_device)
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_1 = pipe(**inputs)[0]
        inputs = self.get_dummy_inputs(paddle_device)
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_2 = pipe(**inputs, control_guidance_start=0.1, control_guidance_end=0.2)[0]
        inputs = self.get_dummy_inputs(paddle_device)
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_3 = pipe(**inputs, control_guidance_start=[0.1], control_guidance_end=[0.2])[0]
        inputs = self.get_dummy_inputs(paddle_device)
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_4 = pipe(**inputs, control_guidance_start=0.4, control_guidance_end=[0.5])[0]
        assert np.sum(np.abs(output_1 - output_2)) > 0.001
        assert np.sum(np.abs(output_1 - output_3)) > 0.001
        assert np.sum(np.abs(output_1 - output_4)) > 0.001

    def test_attention_slicing_forward_pass(self):
        return self._test_attention_slicing_forward_pass(expected_max_diff=0.002)

    @unittest.skipIf(
        not is_ppxformers_available(), reason="XFormers attention is only available with CUDA and `xformers` installed"
    )
    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=0.002)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=0.002)

    def test_save_pretrained_raise_not_implemented_exception(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.to(paddle_device)
        pipe.set_progress_bar_config(disable=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                pipe.save_pretrained(tmpdir)
            except NotImplementedError:
                pass


@slow
@require_paddle_gpu
class ControlNetPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_canny(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.framework.core.default_cpu_generator().manual_seed(0)
        prompt = "bird"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )
        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)
        image = output.images[0]
        assert image.shape == (768, 512, 3)
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny_out.npy"
        )
        assert np.abs(expected_image - image).max() < 0.09

    def test_depth(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.framework.core.default_cpu_generator().manual_seed(0)
        prompt = "Stormtrooper's lecture"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )
        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)
        image = output.images[0]
        assert image.shape == (512, 512, 3)
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth_out.npy"
        )
        assert np.abs(expected_image - image).max() < 0.8

    def test_hed(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.framework.core.default_cpu_generator().manual_seed(0)
        prompt = "oil painting of handsome old man, masterpiece"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/man_hed.png"
        )
        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)
        image = output.images[0]
        assert image.shape == (704, 512, 3)
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/man_hed_out.npy"
        )
        assert np.abs(expected_image - image).max() < 0.08

    def test_mlsd(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.framework.core.default_cpu_generator().manual_seed(0)
        prompt = "room"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/room_mlsd.png"
        )
        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)
        image = output.images[0]
        assert image.shape == (704, 512, 3)
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/room_mlsd_out.npy"
        )
        assert np.abs(expected_image - image).max() < 0.05

    def test_normal(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-normal")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.framework.core.default_cpu_generator().manual_seed(0)
        prompt = "cute toy"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/cute_toy_normal.png"
        )
        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)
        image = output.images[0]
        assert image.shape == (512, 512, 3)
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/cute_toy_normal_out.npy"
        )
        assert np.abs(expected_image - image).max() < 0.05

    def test_openpose(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.framework.core.default_cpu_generator().manual_seed(0)
        prompt = "Chef in the kitchen"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/pose.png"
        )
        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)
        image = output.images[0]
        assert image.shape == (768, 512, 3)
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/chef_pose_out.npy"
        )
        assert np.abs(expected_image - image).max() < 0.08

    def test_scribble(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.framework.core.default_cpu_generator().manual_seed(5)
        prompt = "bag"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bag_scribble.png"
        )
        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)
        image = output.images[0]
        assert image.shape == (640, 512, 3)
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bag_scribble_out.npy"
        )
        assert np.abs(expected_image - image).max() < 0.08

    def test_seg(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.framework.core.default_cpu_generator().manual_seed(5)
        prompt = "house"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/house_seg.png"
        )
        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)
        image = output.images[0]
        assert image.shape == (512, 512, 3)
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/house_seg_out.npy"
        )
        assert np.abs(expected_image - image).max() < 0.08

    def test_sequential_cpu_offloading(self):
        paddle.device.cuda.empty_cache()
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.set_progress_bar_config(disable=None)
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
        prompt = "house"
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/house_seg.png"
        )
        _ = pipe(prompt, image, num_inference_steps=2, output_type="np")
        mem_bytes = paddle.device.cuda.max_memory_allocated()
        assert mem_bytes < 4 * 10**9

    def test_canny_guess_mode(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.framework.core.default_cpu_generator().manual_seed(0)
        prompt = ""
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )
        output = pipe(
            prompt,
            image,
            generator=generator,
            output_type="np",
            num_inference_steps=3,
            guidance_scale=3.0,
            guess_mode=True,
        )
        image = output.images[0]
        assert image.shape == (768, 512, 3)
        image_slice = image[-3:, -3:, (-1)]
        expected_slice = np.array([0.2724, 0.2846, 0.2724, 0.3843, 0.3682, 0.2736, 0.4675, 0.3862, 0.2887])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_canny_guess_mode_euler(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.framework.core.default_cpu_generator().manual_seed(0)
        prompt = ""
        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )
        output = pipe(
            prompt,
            image,
            generator=generator,
            output_type="np",
            num_inference_steps=3,
            guidance_scale=3.0,
            guess_mode=True,
        )
        image = output.images[0]
        assert image.shape == (768, 512, 3)
        image_slice = image[-3:, -3:, (-1)]
        expected_slice = np.array([0.1655, 0.1721, 0.1623, 0.1685, 0.1711, 0.1646, 0.1651, 0.1631, 0.1494])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_v11_shuffle_global_pool_conditions(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11e_sd15_shuffle")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.framework.core.default_cpu_generator().manual_seed(0)
        prompt = "New York"
        image = load_image(
            "https://huggingface.co/lllyasviel/control_v11e_sd15_shuffle/resolve/main/images/control.png"
        )
        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3, guidance_scale=7.0)
        image = output.images[0]
        assert image.shape == (512, 640, 3)
        image_slice = image[-3:, -3:, (-1)]
        expected_slice = np.array([0.1338, 0.1597, 0.1202, 0.1687, 0.1377, 0.1017, 0.207, 0.1574, 0.1348])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_load_local(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")
        pipe_1 = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        controlnet = ControlNetModel.from_single_file(
            "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth"
        )
        pipe_2 = StableDiffusionControlNetPipeline.from_single_file(
            "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors",
            safety_checker=None,
            controlnet=controlnet,
        )
        pipes = [pipe_1, pipe_2]
        images = []
        for pipe in pipes:
            pipe.enable_model_cpu_offload()
            pipe.set_progress_bar_config(disable=None)
            generator = paddle.framework.core.default_cpu_generator().manual_seed(0)
            prompt = "bird"
            image = load_image(
                "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
            )
            output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)
            images.append(output.images[0])
            del pipe
            gc.collect()
            paddle.device.cuda.empty_cache()
        assert np.abs(images[0] - images[1]).sum() < 0.001


@slow
@require_paddle_gpu
class StableDiffusionMultiControlNetPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_pose_and_canny(self):
        controlnet_canny = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
        controlnet_pose = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=[controlnet_pose, controlnet_canny]
        )
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.framework.core.default_cpu_generator().manual_seed(0)
        prompt = "bird and Chef"
        image_canny = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )
        image_pose = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/pose.png"
        )
        output = pipe(prompt, [image_pose, image_canny], generator=generator, output_type="np", num_inference_steps=3)
        image = output.images[0]
        assert image.shape == (768, 512, 3)
        expected_image = load_numpy(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/pose_canny_out.npy"
        )
        assert np.abs(expected_image - image).max() < 0.05
