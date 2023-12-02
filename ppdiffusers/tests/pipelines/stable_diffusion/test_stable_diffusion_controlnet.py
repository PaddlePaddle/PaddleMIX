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
from ppdiffusers.initializer import normal_, ones_
from ppdiffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_controlnet import (
    MultiControlNetModel,
)
from ppdiffusers.utils import load_image, randn_tensor, slow
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

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        controlnet_embedder_scale_factor = 2
        image = randn_tensor(
            (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
            generator=generator,
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

    def test_save_load_local(self):
        pass

    def test_save_load_optional_components(self):
        pass

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
                normal_(m.weight)
                ones_(m.bias)

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

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        controlnet_embedder_scale_factor = 2
        images = [
            randn_tensor(
                (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
                generator=generator,
            ),
            randn_tensor(
                (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
                generator=generator,
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
        scale = 10.0
        steps = 4
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_1 = pipe(**inputs)[0]
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_2 = pipe(**inputs, control_guidance_start=0.1, control_guidance_end=0.2)[0]
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_3 = pipe(**inputs, control_guidance_start=[0.1, 0.3], control_guidance_end=[0.2, 0.7])[0]
        inputs = self.get_dummy_inputs()
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

    def test_save_load_local(self):
        pass

    def test_save_load_optional_components(self):
        pass
    
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
                normal_(m.weight)
                ones_(m.bias)

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

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        controlnet_embedder_scale_factor = 2
        images = [
            randn_tensor(
                (1, 3, 32 * controlnet_embedder_scale_factor, 32 * controlnet_embedder_scale_factor),
                generator=generator,
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
        scale = 10.0
        steps = 4
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_1 = pipe(**inputs)[0]
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_2 = pipe(**inputs, control_guidance_start=0.1, control_guidance_end=0.2)[0]
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = steps
        inputs["controlnet_conditioning_scale"] = scale
        output_3 = pipe(**inputs, control_guidance_start=[0.1], control_guidance_end=[0.2])[0]
        inputs = self.get_dummy_inputs()
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
        # pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        prompt = "bird"
        image = load_image(
            "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )
        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)
        image = output.images[0]
        assert image.shape == (768, 512, 3)
        # expected_image = load_numpy(
        #     "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny_out.npy"
        # )
        image_slice = image[-3:, -3:, (-1)].flatten()
        expected_slice = np.array([0.4075, 0.4689, 0.5782, 0.4584, 0.4629, 0.4236, 0.4906, 0.4784, 0.4211])
        assert np.abs(expected_slice - image_slice).max() < 0.09

    def test_depth(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        # pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        prompt = "Stormtrooper's lecture"
        image = load_image(
            "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth.png"
        )
        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)
        image = output.images[0]
        assert image.shape == (512, 512, 3)
        # expected_image = load_numpy(
        #     "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/stormtrooper_depth_out.npy"
        # )
        image_slice = image[-3:, -3:, (-1)].flatten()
        expected_slice = np.array([0.2724, 0.2846, 0.2724, 0.3843, 0.3682, 0.2736, 0.4675, 0.3862, 0.2887])
        assert np.abs(expected_slice - image_slice).max() < 0.8

    def test_hed(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-hed")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        # pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        prompt = "oil painting of handsome old man, masterpiece"
        image = load_image(
            "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/man_hed.png"
        )
        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)
        image = output.images[0]
        assert image.shape == (704, 512, 3)
        # expected_image = load_numpy(
        #     "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/man_hed_out.npy"
        # )
        image_slice = image[-3:, -3:, (-1)].flatten()
        expected_slice = np.array([0.1532, 0.1748, 0.1616, 0.1534, 0.1568, 0.143, 0.187, 0.1695, 0.099])
        assert np.abs(expected_slice - image_slice).max() < 0.08

    def test_mlsd(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-mlsd")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        # pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        prompt = "room"
        image = load_image(
            "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/room_mlsd.png"
        )
        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)
        image = output.images[0]
        assert image.shape == (704, 512, 3)
        # expected_image = load_numpy(
        #     "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/room_mlsd_out.npy"
        # )
        image_slice = image[-3:, -3:, (-1)].flatten()
        expected_slice = np.array([0.0774, 0.0527, 0.103, 0.0847, 0.145, 0.1387, 0.0771, 0.0658, 0.0297])
        assert np.abs(expected_slice - image_slice).max() < 0.05

    def test_normal(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-normal")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        # pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        prompt = "cute toy"
        image = load_image(
            "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/cute_toy_normal.png"
        )
        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)
        image = output.images[0]
        assert image.shape == (512, 512, 3)
        # expected_image = load_numpy(
        #     "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/cute_toy_normal_out.npy"
        # )
        image_slice = image[-3:, -3:, (-1)].flatten()
        expected_slice = np.array([0.7348, 0.7332, 0.7101, 0.7802, 0.7513, 0.7457, 0.7834, 0.7948, 0.7722])
        assert np.abs(expected_slice - image_slice).max() < 0.05

    def test_openpose(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        # pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        prompt = "Chef in the kitchen"
        image = load_image(
            "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/pose.png"
        )
        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)
        image = output.images[0]
        assert image.shape == (768, 512, 3)
        # expected_image = load_numpy(
        #     "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/chef_pose_out.npy"
        # )
        image_slice = image[-3:, -3:, (-1)].flatten()
        expected_slice = np.array([0.4762, 0.5183, 0.5624, 0.5271, 0.5165, 0.4066, 0.5489, 0.5187, 0.453])
        assert np.abs(expected_slice - image_slice).max() < 0.08

    def test_scribble(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        # pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(5)
        prompt = "bag"
        image = load_image(
            "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bag_scribble.png"
        )
        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)
        image = output.images[0]
        assert image.shape == (640, 512, 3)
        # expected_image = load_numpy(
        #     "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bag_scribble_out.npy"
        # )
        image_slice = image[-3:, -3:, (-1)].flatten()
        expected_slice = np.array([0.6685, 0.6627, 0.6579, 0.7025, 0.6568, 0.5743, 0.6975, 0.6534, 0.5165])
        assert np.abs(expected_slice - image_slice).max() < 0.15

    def test_seg(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        # pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(5)
        prompt = "house"
        image = load_image(
            "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/house_seg.png"
        )
        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)
        image = output.images[0]
        assert image.shape == (512, 512, 3)
        # expected_image = load_numpy(
        #     "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/house_seg_out.npy"
        # )
        image_slice = image[-3:, -3:, (-1)].flatten()
        expected_slice = np.array([0.3855, 0.5808, 0.4667, 0.3666, 0.4816, 0.477, 0.3183, 0.4102, 0.4634])
        assert np.abs(expected_slice - image_slice).max() < 0.08

    def test_canny_guess_mode(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        # pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        prompt = ""
        image = load_image(
            "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
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
        image_slice = image[-3:, -3:, (-1)].flatten()
        expected_slice = np.array([0.3819, 0.4531, 0.5459, 0.4561, 0.4604, 0.4131, 0.4962, 0.4982, 0.448])
        assert np.abs(image_slice - expected_slice).max() < 0.01

    def test_canny_guess_mode_euler(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        # pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        prompt = ""
        image = load_image(
            "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
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
        image_slice = image[-3:, -3:, (-1)].flatten()
        expected_slice = np.array([0.3637, 0.3756, 0.3697, 0.3673, 0.3894, 0.3658, 0.3738, 0.3704, 0.374])
        assert np.abs(image_slice - expected_slice).max() < 0.01

    def test_v11_shuffle_global_pool_conditions(self):
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11e_sd15_shuffle")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
        )
        # pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        prompt = "New York"
        image = load_image(
            "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/lllyasviel/control_v11e_sd15_shuffle/control.png"
        )
        output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3, guidance_scale=7.0)
        image = output.images[0]
        assert image.shape == (512, 640, 3)
        image_slice = image[-3:, -3:, (-1)].flatten()
        expected_slice = np.array([0.9927, 0.9621, 0.8508, 0.8435, 0.7473, 0.539, 0.5302, 0.456, 0.3869])
        assert np.abs(image_slice - expected_slice).max() < 0.01

    # def test_load_local(self):
    #     pass
    #     controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")
    #     pipe_1 = StableDiffusionControlNetPipeline.from_pretrained(
    #         "runwayml/stable-diffusion-v1-5", safety_checker=None, controlnet=controlnet
    #     )
    #     controlnet = ControlNetModel.from_single_file(
    #         "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth"
    #     )
    #     pipe_2 = StableDiffusionControlNetPipeline.from_single_file(
    #         "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors",
    #         safety_checker=None,
    #         controlnet=controlnet,
    #     )
    #     pipes = [pipe_1, pipe_2]
    #     images = []
    #     for pipe in pipes:
    #         #pipe.enable_model_cpu_offload()
    #         pipe.set_progress_bar_config(disable=None)
    #         generator = paddle.Generator().manual_seed(0)
    #         prompt = "bird"
    #         image = load_image(
    #             "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
    #         )
    #         output = pipe(prompt, image, generator=generator, output_type="np", num_inference_steps=3)
    #         images.append(output.images[0])
    #         del pipe
    #         gc.collect()
    #         paddle.device.cuda.empty_cache()
    #     assert np.abs(images[0] - images[1]).sum() < 0.001


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
        # pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)
        generator = paddle.Generator().manual_seed(0)
        prompt = "bird and Chef"
        image_canny = load_image(
            "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png"
        )
        image_pose = load_image(
            "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/pose.png"
        )
        output = pipe(prompt, [image_pose, image_canny], generator=generator, output_type="np", num_inference_steps=3)
        image = output.images[0]
        assert image.shape == (768, 512, 3)
        # expected_image = load_numpy(
        #     "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/pose_canny_out.npy"
        # )
        image_slice = image[-3:, -3:, (-1)].flatten()
        expected_slice = np.array([0.3953, 0.4717, 0.5532, 0.4267, 0.4481, 0.4056, 0.4556, 0.44, 0.3725])
        assert np.abs(expected_slice - image_slice).max() < 0.05
