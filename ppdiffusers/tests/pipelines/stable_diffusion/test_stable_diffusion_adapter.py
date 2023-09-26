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
from paddlenlp.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from ppdiffusers import (
    AutoencoderKL,
    PNDMScheduler,
    StableDiffusionAdapterPipeline,
    T2IAdapter,
    UNet2DConditionModel,
)
from ppdiffusers.utils import floats_tensor, slow
from ppdiffusers.utils.import_utils import is_ppxformers_available
from ppdiffusers.utils.testing_utils import (
    enable_full_determinism,
    paddle_device,
    require_paddle_gpu,
)

from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
)
from ..test_pipelines_common import PipelineTesterMixin

enable_full_determinism()


class AdapterTests:
    pipeline_class = StableDiffusionAdapterPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS

    def get_dummy_components(self, adapter_type):
        paddle.seed(seed=0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
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
        paddle.seed(seed=0)
        adapter = T2IAdapter(
            in_channels=3, channels=[32, 64], num_res_blocks=2, downscale_factor=2, adapter_type=adapter_type
        )
        components = {
            "adapter": adapter,
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
        image = floats_tensor((1, 3, 64, 64), rng=random.Random(seed))
        generator = paddle.Generator().manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 6.0,
            "output_type": "np",
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


class StableDiffusionFullAdapterPipelineFastTests(AdapterTests, PipelineTesterMixin, unittest.TestCase):
    def get_dummy_components(self):
        return super().get_dummy_components("full_adapter")

    def test_stable_diffusion_adapter_default_case(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionAdapterPipeline(**components)
        sd_pipe = sd_pipe.to(paddle_device)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.2799, 0.2306, 0.3051, 0.2956, 0.3126, 0.5048, 0.3467, 0.2293, 0.4326])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.005


class StableDiffusionLightAdapterPipelineFastTests(AdapterTests, PipelineTesterMixin, unittest.TestCase):
    def get_dummy_components(self):
        return super().get_dummy_components("light_adapter")

    def test_stable_diffusion_adapter_default_case(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionAdapterPipeline(**components)
        sd_pipe = sd_pipe.to(paddle_device)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.4709, 0.2308, 0.2841, 0.2938, 0.2866, 0.5022, 0.3457, 0.2516, 0.4701])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.005


@slow
@require_paddle_gpu
class StableDiffusionAdapterPipelineSlowTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_stable_diffusion_adapter(self):
        pass
        # no those weights now
        # test_cases = [
        #     (
        #         "TencentARC/t2iadapter_color_sd14v1",
        #         "CompVis/stable-diffusion-v1-4",
        #         "snail",
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/color.png",
        #         3,
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/t2iadapter_color_sd14v1.npy",
        #     ),
        #     (
        #         "TencentARC/t2iadapter_depth_sd14v1",
        #         "CompVis/stable-diffusion-v1-4",
        #         "desk",
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/desk_depth.png",
        #         3,
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/t2iadapter_depth_sd14v1.npy",
        #     ),
        #     (
        #         "TencentARC/t2iadapter_depth_sd15v2",
        #         "runwayml/stable-diffusion-v1-5",
        #         "desk",
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/desk_depth.png",
        #         3,
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/t2iadapter_depth_sd15v2.npy",
        #     ),
        #     (
        #         "TencentARC/t2iadapter_keypose_sd14v1",
        #         "CompVis/stable-diffusion-v1-4",
        #         "person",
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/person_keypose.png",
        #         3,
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/t2iadapter_keypose_sd14v1.npy",
        #     ),
        #     (
        #         "TencentARC/t2iadapter_openpose_sd14v1",
        #         "CompVis/stable-diffusion-v1-4",
        #         "person",
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/iron_man_pose.png",
        #         3,
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/t2iadapter_openpose_sd14v1.npy",
        #     ),
        #     (
        #         "TencentARC/t2iadapter_seg_sd14v1",
        #         "CompVis/stable-diffusion-v1-4",
        #         "motorcycle",
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/motor.png",
        #         3,
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/t2iadapter_seg_sd14v1.npy",
        #     ),
        #     (
        #         "TencentARC/t2iadapter_zoedepth_sd15v1",
        #         "runwayml/stable-diffusion-v1-5",
        #         "motorcycle",
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/motorcycle.png",
        #         3,
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/t2iadapter_zoedepth_sd15v1.npy",
        #     ),
        #     (
        #         "TencentARC/t2iadapter_canny_sd14v1",
        #         "CompVis/stable-diffusion-v1-4",
        #         "toy",
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/toy_canny.png",
        #         1,
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/t2iadapter_canny_sd14v1.npy",
        #     ),
        #     (
        #         "TencentARC/t2iadapter_canny_sd15v2",
        #         "runwayml/stable-diffusion-v1-5",
        #         "toy",
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/toy_canny.png",
        #         1,
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/t2iadapter_canny_sd15v2.npy",
        #     ),
        #     (
        #         "TencentARC/t2iadapter_sketch_sd14v1",
        #         "CompVis/stable-diffusion-v1-4",
        #         "cat",
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/edge.png",
        #         1,
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/t2iadapter_sketch_sd14v1.npy",
        #     ),
        #     (
        #         "TencentARC/t2iadapter_sketch_sd15v2",
        #         "runwayml/stable-diffusion-v1-5",
        #         "cat",
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/edge.png",
        #         1,
        #         "https://bj.bcebos.com/v1/paddlenlp/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/t2iadapter_sketch_sd15v2.npy",
        #     ),
        # ]
        # for adapter_model, sd_model, prompt, image_url, input_channels, out_url in test_cases:
        #     image = load_image(image_url)
        #     expected_out = load_numpy(out_url)
        #     if input_channels == 1:
        #         image = image.convert("L")
        #     adapter = T2IAdapter.from_pretrained(adapter_model, paddle_dtype=paddle.float16)
        #     pipe = StableDiffusionAdapterPipeline.from_pretrained(sd_model, adapter=adapter, safety_checker=None)
        #     pipe.to(paddle_device)
        #     pipe.set_progress_bar_config(disable=None)
        #     pipe.enable_attention_slicing()
        #     generator = paddle.Generator().manual_seed(0)
        #     out = pipe(prompt=prompt, image=image, generator=generator, num_inference_steps=2, output_type="np").images
        #     self.assertTrue(np.allclose(out, expected_out))

    # def test_stable_diffusion_adapter_pipeline_with_sequential_cpu_offloading(self):
    #     paddle.device.cuda.empty_cache()
    #     adapter = T2IAdapter.from_pretrained("TencentARC/t2iadapter_seg_sd14v1")
    #     pipe = StableDiffusionAdapterPipeline.from_pretrained(
    #         "CompVis/stable-diffusion-v1-4", adapter=adapter, safety_checker=None
    #     )
    #     pipe = pipe.to(paddle_device)
    #     pipe.set_progress_bar_config(disable=None)
    #     pipe.enable_attention_slicing(1)
    #     # pipe.enable_sequential_cpu_offload()
    #     image = load_image(
    #         "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/t2i_adapter/motor.png"
    #     )
    #     pipe(prompt="foo", image=image, num_inference_steps=2)
    #     mem_bytes = paddle.device.cuda.max_memory_allocated()
    #     assert mem_bytes < 5 * 10**9
