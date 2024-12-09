# coding=utf-8
# Copyright 2024 HuggingFace Inc and The InstantX Team.
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

from ppdiffusers.transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel

from ppdiffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3ControlNetPipeline,
)
from ppdiffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from ppdiffusers.utils import load_image
from ppdiffusers.utils.testing_utils import (
    enable_full_determinism,
    require_paddle_gpu,
    slow,
)
from ppdiffusers.utils import randn_tensor

from ..test_pipelines_common import PipelineTesterMixin


enable_full_determinism()


class StableDiffusion3ControlNetPipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = StableDiffusion3ControlNetPipeline
    params = frozenset(
        [
            "prompt",
            "height",
            "width",
            "guidance_scale",
            "negative_prompt",
            "prompt_embeds",
            "negative_prompt_embeds",
        ]
    )
    batch_params = frozenset(["prompt", "negative_prompt"])

    def get_dummy_components(self):
        paddle.seed(seed=0)
        transformer = SD3Transformer2DModel(
            sample_size=32,
            patch_size=1,
            in_channels=8,
            num_layers=4,
            attention_head_dim=8,
            num_attention_heads=4,
            joint_attention_dim=32,
            caption_projection_dim=32,
            pooled_projection_dim=64,
            out_channels=8,
        )

        paddle.seed(seed=0)
        controlnet = SD3ControlNetModel(
            sample_size=32,
            patch_size=1,
            in_channels=8,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=4,
            joint_attention_dim=32,
            caption_projection_dim=32,
            pooled_projection_dim=64,
            out_channels=8,
        )
        clip_text_encoder_config = CLIPTextConfig(
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
            projection_dim=32,
        )

        paddle.seed(seed=0)
        text_encoder = CLIPTextModelWithProjection(clip_text_encoder_config)

        paddle.seed(seed=0)
        text_encoder_2 = CLIPTextModelWithProjection(clip_text_encoder_config)

        paddle.seed(seed=0)
        text_encoder_3 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_3 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        paddle.seed(seed=0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=8,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        scheduler = FlowMatchEulerDiscreteScheduler()

        return {
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "text_encoder_2": text_encoder_2,
            "text_encoder_3": text_encoder_3,
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
            "tokenizer_3": tokenizer_3,
            "transformer": transformer,
            "vae": vae,
            "controlnet": controlnet,
        }

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        control_image = randn_tensor(
            (1, 3, 32, 32),
            generator=generator,
            dtype=paddle.float16,
        )

        controlnet_conditioning_scale = 0.5

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "control_image": control_image,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
        }

        return inputs

    def test_controlnet_sd3(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusion3ControlNetPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        output = sd_pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)

        expected_slice = np.array([0.7100817,  0.01452562, 0.8383021,  0.58670187, 0.76902485, 0.38530028, 0.8903022,  0.37712285, 0.5624174])

        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        ), f"Expected: {expected_slice}, got: {image_slice.flatten()}"

    @unittest.skip("xFormersAttnProcessor does not work with SD3 Joint Attention")
    def test_xformers_attention_forwardGenerator_pass(self):
        pass


@slow
@require_paddle_gpu
class StableDiffusion3ControlNetPipelineSlowTests(unittest.TestCase):
    pipeline_class = StableDiffusion3ControlNetPipeline

    def setUp(self):
        super().setUp()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_canny(self):
        controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny", paddle_dtype=paddle.float16)
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, paddle_dtype=paddle.float16
        )
        # pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        paddle.seed(seed=0)
        prompt = "Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text 'InstantX' on image"
        n_prompt = "NSFW, nude, naked, porn, ugly"
        control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")

        output = pipe(
            prompt,
            negative_prompt=n_prompt,
            control_image=control_image,
            controlnet_conditioning_scale=0.5,
            guidance_scale=5.0,
            num_inference_steps=2,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (1024, 1024, 3)

        original_image = image[-3:, -3:, -1].flatten()

        expected_image = np.array(
            [0.8154297,  0.79785156, 0.77490234, 0.6542969,  0.8066406,  0.85546875,  0.6899414,  0.77246094, 0.7597656 ]
        )

        assert np.abs(original_image.flatten() - expected_image).max() < 1e-2

    def test_pose(self):
        controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Pose", paddle_dtype=paddle.float16)
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, paddle_dtype=paddle.float16
        )
        # pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        paddle.seed(seed=0)
        prompt = 'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image'
        n_prompt = "NSFW, nude, naked, porn, ugly"
        control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Pose/resolve/main/pose.jpg")

        output = pipe(
            prompt,
            negative_prompt=n_prompt,
            control_image=control_image,
            controlnet_conditioning_scale=0.5,
            guidance_scale=5.0,
            num_inference_steps=2,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (1024, 1024, 3)

        original_image = image[-3:, -3:, -1].flatten()

        expected_image = np.array(
            [0.86083984, 0.83496094, 0.7734375,  0.76123047, 0.76171875, 0.90234375,  0.72265625, 0.7558594,  0.79296875]
        )

        assert np.abs(original_image.flatten() - expected_image).max() < 1e-2

    def test_tile(self):
        controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Tile", paddle_dtype=paddle.float16)
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, paddle_dtype=paddle.float16
        )
        # pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        paddle.seed(seed=0)
        prompt = 'Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text "InstantX" on image'
        n_prompt = "NSFW, nude, naked, porn, ugly"
        control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Tile/resolve/main/tile.jpg")

        output = pipe(
            prompt,
            negative_prompt=n_prompt,
            control_image=control_image,
            controlnet_conditioning_scale=0.5,
            guidance_scale=5.0,
            num_inference_steps=2,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (1024, 1024, 3)

        original_image = image[-3:, -3:, -1].flatten()

        expected_image = np.array(
            [0.6816406,  0.70166016, 0.6611328,  0.65722656, 0.70214844, 0.68847656,  0.68359375, 0.65722656, 0.6254883]
        )

        assert np.abs(original_image.flatten() - expected_image).max() < 1e-2

    def test_multi_controlnet(self):
        controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny", paddle_dtype=paddle.float16)
        controlnet = SD3MultiControlNetModel([controlnet, controlnet])

        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet, paddle_dtype=paddle.float16
        )
        # pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=None)

        paddle.seed(seed=0)
        prompt = "Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching. text 'InstantX' on image"
        n_prompt = "NSFW, nude, naked, porn, ugly"
        control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")

        output = pipe(
            prompt,
            negative_prompt=n_prompt,
            control_image=[control_image, control_image],
            controlnet_conditioning_scale=[0.25, 0.25],
            guidance_scale=5.0,
            num_inference_steps=2,
            output_type="np",
        )
        image = output.images[0]

        assert image.shape == (1024, 1024, 3)

        original_image = image[-3:, -3:, -1].flatten()
        expected_image = np.array(
            [0.87890625, 0.7392578,  0.6123047,  0.7128906,  0.7734375,  0.81933594, 0.60058594, 0.75,       0.75634766]
        )


        assert np.abs(original_image.flatten() - expected_image).max() < 1e-2
