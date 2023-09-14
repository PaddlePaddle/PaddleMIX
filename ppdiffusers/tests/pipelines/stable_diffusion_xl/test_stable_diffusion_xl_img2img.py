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

import random
import unittest

import numpy as np
import paddle
from paddlenlp.transformers import (
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)

from ppdiffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import floats_tensor
from ppdiffusers.utils.testing_utils import enable_full_determinism

from ..pipeline_params import (
    IMAGE_TO_IMAGE_IMAGE_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
)
from ..test_pipelines_common import PipelineLatentTesterMixin, PipelineTesterMixin

enable_full_determinism()


class StableDiffusionXLImg2ImgPipelineFastTests(PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionXLImg2ImgPipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {"height", "width"}
    required_optional_params = PipelineTesterMixin.required_optional_params - {"latents"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS
    image_params = IMAGE_TO_IMAGE_IMAGE_PARAMS
    image_latents_params = IMAGE_TO_IMAGE_IMAGE_PARAMS

    def get_dummy_components(self, skip_first_text_encoder=False):
        paddle.seed(seed=0)
        unet = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=72,
            cross_attention_dim=64 if not skip_first_text_encoder else 32,
        )
        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            steps_offset=1,
            beta_schedule="scaled_linear",
            timestep_spacing="leading",
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
            projection_dim=32,
        )
        text_encoder = CLIPTextModel(text_encoder_config)
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        text_encoder_2 = CLIPTextModelWithProjection(text_encoder_config)
        tokenizer_2 = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        components = {
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder if not skip_first_text_encoder else None,
            "tokenizer": tokenizer if not skip_first_text_encoder else None,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
            "requires_aesthetics_score": True,
        }
        return components

    def test_components_function(self):
        init_components = self.get_dummy_components()
        init_components.pop("requires_aesthetics_score")
        pipe = self.pipeline_class(**init_components)
        self.assertTrue(hasattr(pipe, "components"))
        self.assertTrue(set(pipe.components.keys()) == set(init_components.keys()))

    def get_dummy_inputs(self, seed=0):
        image = floats_tensor((1, 3, 32, 32), rng=random.Random(seed))
        image = image / 2 + 0.5
        generator = paddle.Generator().manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": image,
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
            "strength": 0.8,
        }
        return inputs

    def test_stable_diffusion_xl_img2img_euler(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.0202, 0.7641, 0.6398, 0.1545, 0.9152, 0.4367, 0.3589, 0.6175, 0.4299])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_xl_refiner(self):
        components = self.get_dummy_components(skip_first_text_encoder=True)
        sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 32, 32, 3)
        expected_slice = np.array([0.0665, 0.8092, 0.645, 0.1931, 0.894, 0.4368, 0.3969, 0.5983, 0.4375])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_attention_slicing_forward_pass(self):
        super().test_attention_slicing_forward_pass()

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical()

    def test_save_load_optional_components(self):
        pass

    def test_stable_diffusion_xl_img2img_negative_prompt_embeds(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        negative_prompt = 3 * ["this is a negative prompt"]
        inputs["negative_prompt"] = negative_prompt
        inputs["prompt"] = 3 * [inputs["prompt"]]
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[(0), -3:, -3:, (-1)]
        inputs = self.get_dummy_inputs()
        negative_prompt = 3 * ["this is a negative prompt"]
        prompt = 3 * [inputs.pop("prompt")]
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = sd_pipe.encode_prompt(prompt, negative_prompt=negative_prompt)
        output = sd_pipe(
            **inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )
        image_slice_2 = output.images[(0), -3:, -3:, (-1)]
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 0.0001

    # @require_paddle_gpu
    # def test_stable_diffusion_xl_offloads(self):
    #     pipes = []
    #     components = self.get_dummy_components()
    #     sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
    #     pipes.append(sd_pipe)
    #     components = self.get_dummy_components()
    #     sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
    #     sd_pipe.enable_model_cpu_offload()
    #     pipes.append(sd_pipe)
    #     components = self.get_dummy_components()
    #     sd_pipe = StableDiffusionXLImg2ImgPipeline(**components)
    #     sd_pipe.enable_sequential_cpu_offload()
    #     pipes.append(sd_pipe)
    #     image_slices = []
    #     for pipe in pipes:
    #         pipe.unet.set_default_attn_processor()
    #         inputs = self.get_dummy_inputs()
    #         image = pipe(**inputs).images
    #         image_slices.append(image[(0), -3:, -3:, (-1)].flatten())
    #     assert np.abs(image_slices[0] - image_slices[1]).max() < 0.001
    #     assert np.abs(image_slices[0] - image_slices[2]).max() < 0.001

    def test_stable_diffusion_xl_multi_prompts(self):
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[(0), -3:, -3:, (-1)]
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        inputs["prompt_2"] = inputs["prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[(0), -3:, -3:, (-1)]
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 0.0001
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        inputs["prompt_2"] = "different prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[(0), -3:, -3:, (-1)]
        assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 0.0001
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        inputs["negative_prompt"] = "negative prompt"
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[(0), -3:, -3:, (-1)]
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = inputs["negative_prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[(0), -3:, -3:, (-1)]
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 0.0001
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 5
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = "different negative prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[(0), -3:, -3:, (-1)]
        assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 0.0001
