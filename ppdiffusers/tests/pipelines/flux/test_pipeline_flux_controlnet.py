# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from ppdiffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxControlNetPipeline,
    FluxTransformer2DModel,
)
from ppdiffusers.models import FluxControlNetModel, FluxMultiControlNetModel
from ppdiffusers.transformers import (
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
)
from ppdiffusers.utils import load_image, randn_tensor
from ppdiffusers.utils.testing_utils import (
    enable_full_determinism,
    numpy_cosine_similarity_distance,
    require_paddle_gpu,
    slow,
)

from ..test_pipelines_common import PipelineTesterMixin

enable_full_determinism()


class FluxControlNetPipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = FluxControlNetPipeline
    params = frozenset(["prompt", "height", "width", "guidance_scale", "prompt_embeds", "pooled_prompt_embeds"])
    batch_params = frozenset(["prompt"])

    # there is no xformers processor for Flux
    test_xformers_attention = False
    test_layerwise_casting = True
    test_group_offloading = True

    def get_dummy_components(self):
        paddle.seed(seed=0)
        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
        )

        paddle.seed(seed=0)
        controlnet = FluxControlNetModel(
            patch_size=1,
            in_channels=16,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
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
        text_encoder = CLIPTextModel(clip_text_encoder_config)

        paddle.seed(seed=0)
        text_encoder_2 = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")

        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        tokenizer_2 = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")

        paddle.seed(seed=0)
        vae = AutoencoderKL(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            block_out_channels=(4,),
            layers_per_block=1,
            latent_channels=4,
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
            "tokenizer": tokenizer,
            "tokenizer_2": tokenizer_2,
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
            "guidance_scale": 3.5,
            "output_type": "np",
            "control_image": control_image,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
        }

        return inputs

    def test_controlnet_flux(self):
        components = self.get_dummy_components()
        pipe = FluxControlNetPipeline(**components)
        pipe.set_progress_bar_config(disable=None)

        inputs = self.get_dummy_inputs()
        output = pipe(**inputs)
        image = output.images

        image_slice = image[0, -3:, -3:, -1]

        assert image.shape == (1, 32, 32, 3)

        expected_slice = np.array(
            [
                0.89820540,
                0.32847900,
                0.94486995,
                0.47045115,
                0.24701830,
                0.00000005,
                0.82854380,
                0.25686050,
                0.54220625,
            ]
        )

        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
        ), f"Expected: {expected_slice}, got: {image_slice.flatten()}"

    def test_flux_different_prompts(self):
        pipe = self.pipeline_class(**self.get_dummy_components())

        inputs = self.get_dummy_inputs()
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs()
        inputs["prompt_2"] = "a different prompt"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = np.abs(output_same_prompt - output_different_prompts).max()

        # Outputs should be different here
        assert max_diff > 1e-6

    def test_flux_prompt_embeds(self):
        pipe = self.pipeline_class(**self.get_dummy_components())
        inputs = self.get_dummy_inputs()

        output_with_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs()
        prompt = inputs.pop("prompt")

        (prompt_embeds, pooled_prompt_embeds, text_ids) = pipe.encode_prompt(
            prompt,
            prompt_2=None,
            max_sequence_length=512,  # Use a default value for testing
        )
        output_with_embeds = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            **inputs,
        ).images[0]

        max_diff = np.abs(output_with_prompt - output_with_embeds).max()
        assert max_diff < 1e-4

    def test_flux_image_output_shape(self):
        pipe = self.pipeline_class(**self.get_dummy_components())
        inputs = self.get_dummy_inputs()

        height_width_pairs = [(32, 32), (72, 56)]
        for height, width in height_width_pairs:
            expected_height = height - height % (pipe.vae_scale_factor * 2)
            expected_width = width - width % (pipe.vae_scale_factor * 2)

            generator = paddle.Generator().manual_seed(0)
            control_image = randn_tensor(
                (1, 3, height, width),
                generator=generator,
                dtype=paddle.float16,
            )

            inputs.update({"control_image": control_image})
            image = pipe(**inputs).images[0]
            output_height, output_width, _ = image.shape
            assert (output_height, output_width) == (expected_height, expected_width)


@slow
@require_paddle_gpu
class FluxControlNetPipelineSlowTests(unittest.TestCase):
    pipeline_class = FluxControlNetPipeline

    def setUp(self):
        super().setUp()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_canny(self):
        controlnet = FluxControlNetModel.from_pretrained(
            "InstantX/FLUX.1-dev-Controlnet-Canny", paddle_dtype=paddle.bfloat16
        )
        pipe = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", controlnet=controlnet, paddle_dtype=paddle.bfloat16
        )
        pipe.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        control_image = load_image(
            "https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg"
        ).resize((512, 512))

        prompt = "A girl in city, 25 years old, cool, futuristic"
        prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            max_sequence_length=256,
        )

        output = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            control_image=control_image,
            controlnet_conditioning_scale=0.6,
            width=512,
            height=512,
            guidance_scale=3.5,
            num_inference_steps=2,
            max_sequence_length=256,
            output_type="np",
            generator=generator,
        )
        image = output.images[0]

        assert image.shape == (512, 512, 3)

        original_image = image[-3:, -3:, -1].flatten()

        expected_image = np.array([0.2734, 0.2852, 0.2852, 0.2734, 0.2754, 0.2891, 0.2617, 0.2637, 0.2773])

        assert numpy_cosine_similarity_distance(original_image.flatten(), expected_image) < 2e-2

    def test_multi_controlnet(self):
        controlnet = FluxControlNetModel.from_pretrained(
            "InstantX/FLUX.1-dev-Controlnet-Canny", paddle_dtype=paddle.bfloat16
        )
        controlnet = FluxMultiControlNetModel([controlnet, controlnet])

        pipe = FluxControlNetPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", controlnet=controlnet, paddle_dtype=paddle.bfloat16
        )
        pipe.set_progress_bar_config(disable=None)

        generator = paddle.Generator().manual_seed(0)
        prompt = "A girl in city, 25 years old, cool, futuristic"
        control_image = load_image(
            "https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg"
        ).resize((512, 512))

        output = pipe(
            prompt,
            control_image=[control_image, control_image],
            controlnet_conditioning_scale=[0.3, 0.3],
            width=512,
            height=512,
            guidance_scale=3.5,
            num_inference_steps=2,
            max_sequence_length=256,
            output_type="np",
            generator=generator,
        )
        image = output.images[0]

        assert image.shape == (512, 512, 3)

        original_image = image[-3:, -3:, -1].flatten()

        expected_image = np.array([0.2744, 0.2862, 0.2862, 0.2744, 0.2764, 0.2881, 0.2627, 0.2647, 0.2763])

        assert numpy_cosine_similarity_distance(original_image.flatten(), expected_image) < 2e-2
