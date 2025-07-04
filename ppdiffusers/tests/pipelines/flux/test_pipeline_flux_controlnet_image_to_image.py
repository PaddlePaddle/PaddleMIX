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

import unittest

import numpy as np
import paddle

from ppdiffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxControlNetImg2ImgPipeline,
    FluxTransformer2DModel,
)
from ppdiffusers.models import FluxControlNetModel
from ppdiffusers.transformers import (
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
)
from ppdiffusers.utils import randn_tensor
from ppdiffusers.utils.testing_utils import enable_full_determinism

from ..test_pipelines_common import PipelineTesterMixin

enable_full_determinism()


class FluxControlNetImg2ImgPipelineFastTests(unittest.TestCase, PipelineTesterMixin):
    pipeline_class = FluxControlNetImg2ImgPipeline
    params = frozenset(
        [
            "prompt",
            "image",
            "control_image",
            "height",
            "width",
            "strength",
            "guidance_scale",
            "controlnet_conditioning_scale",
            "prompt_embeds",
            "pooled_prompt_embeds",
        ]
    )
    batch_params = frozenset(["prompt", "image", "control_image"])

    test_xformers_attention = False

    def get_dummy_components(self):
        paddle.seed(seed=0)
        transformer = FluxTransformer2DModel(
            patch_size=1,
            in_channels=4,
            out_channels=4,
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
            latent_channels=1,
            norm_num_groups=1,
            use_quant_conv=False,
            use_post_quant_conv=False,
            shift_factor=0.0609,
            scaling_factor=1.5035,
        )

        paddle.seed(seed=0)
        controlnet = FluxControlNetModel(
            in_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            joint_attention_dim=32,
            pooled_projection_dim=32,
            axes_dims_rope=[4, 4, 8],
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

        # Generar y normalizar al rango [0,1]
        image = paddle.randn([1, 3, 32, 32])
        image = (image + 1) / 2.0  # Normaliza de [-1,1] a [0,1]
        image = paddle.clip(image, 0.0, 1.0)  # Asegura que está en [0,1]

        control_image = paddle.randn([1, 3, 32, 32])
        control_image = (control_image + 1) / 2.0
        control_image = paddle.clip(control_image, 0.0, 1.0)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "control_image": control_image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "controlnet_conditioning_scale": 1.0,
            "strength": 0.8,
            "height": 32,
            "width": 32,
            "max_sequence_length": 48,
            "output_type": "np",
        }
        return inputs

    def test_flux_controlnet_different_prompts(self):
        pipe = self.pipeline_class(**self.get_dummy_components())

        inputs = self.get_dummy_inputs()
        output_same_prompt = pipe(**inputs).images[0]

        inputs = self.get_dummy_inputs()
        inputs["prompt_2"] = "a different prompt"
        output_different_prompts = pipe(**inputs).images[0]

        max_diff = np.abs(output_same_prompt - output_different_prompts).max()

        assert max_diff > 1e-6

    def test_flux_image_output_shape(self):
        pipe = self.pipeline_class(**self.get_dummy_components())
        inputs = self.get_dummy_inputs()

        height_width_pairs = [(32, 32), (72, 56)]
        for height, width in height_width_pairs:
            expected_height = height - height % (pipe.vae_scale_factor * 2)
            expected_width = width - width % (pipe.vae_scale_factor * 2)

            generator = paddle.Generator().manual_seed(0)
            # Generar y normalizar
            control_image = randn_tensor(
                (1, 3, height, width),
                generator=generator,
                dtype=paddle.float16,
            )
            control_image = (control_image + 1) / 2.0
            control_image = paddle.clip(control_image, 0.0, 1.0)

            image = randn_tensor(
                (1, 3, height, width),
                generator=generator,
                dtype=paddle.float16,
            )
            image = (image + 1) / 2.0
            image = paddle.clip(image, 0.0, 1.0)

            inputs.update(
                {
                    "control_image": control_image,
                    "image": image,
                    "height": height,
                    "width": width,
                }
            )

            output = pipe(**inputs).images[0]
            output_height, output_width, _ = output.shape
            assert (output_height, output_width) == (expected_height, expected_width)
