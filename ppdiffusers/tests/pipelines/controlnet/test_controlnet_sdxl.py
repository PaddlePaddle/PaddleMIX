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
    ControlNetModel,
    EulerDiscreteScheduler,
    StableDiffusionXLControlNetPipeline,
    UNet2DConditionModel,
)
from ppdiffusers.utils import randn_tensor
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


class ControlNetPipelineSDXLFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionXLControlNetPipeline
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
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,
            cross_attention_dim=64,
        )
        paddle.seed(seed=0)
        controlnet = ControlNetModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            conditioning_embedding_out_channels=(16, 32),
            attention_head_dim=(2, 4),
            use_linear_projection=True,
            addition_embed_type="text_time",
            addition_time_embed_dim=8,
            transformer_layers_per_block=(1, 2),
            projection_class_embeddings_input_dim=80,
            cross_attention_dim=64,
        )
        paddle.seed(seed=0)
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
            "controlnet": controlnet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
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

    def test_xformers_attention_forwardGenerator_pass(self):
        self._test_xformers_attention_forwardGenerator_pass(expected_max_diff=0.002)

    def test_inference_batch_single_identical(self):
        self._test_inference_batch_single_identical(expected_max_diff=0.002)

    @require_paddle_gpu
    def test_stable_diffusion_xl_offloads(self):
        pipes = []
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        pipes.append(sd_pipe)
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)

        pipes.append(sd_pipe)
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)

        pipes.append(sd_pipe)
        image_slices = []
        for pipe in pipes:
            pipe.unet.set_default_attn_processor()
            inputs = self.get_dummy_inputs()
            image = pipe(**inputs).images
            image_slices.append(image[(0), -3:, -3:, (-1)].flatten())
        assert np.abs(image_slices[0] - image_slices[1]).max() < 0.001
        assert np.abs(image_slices[0] - image_slices[2]).max() < 0.001

    def test_stable_diffusion_xl_multi_prompts(self):
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        inputs = self.get_dummy_inputs()
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[(0), -3:, -3:, (-1)]
        inputs = self.get_dummy_inputs()
        inputs["prompt_2"] = inputs["prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[(0), -3:, -3:, (-1)]
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 0.0001
        inputs = self.get_dummy_inputs()
        inputs["prompt_2"] = "different prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[(0), -3:, -3:, (-1)]
        assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 0.0001
        inputs = self.get_dummy_inputs()
        inputs["negative_prompt"] = "negative prompt"
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[(0), -3:, -3:, (-1)]
        inputs = self.get_dummy_inputs()
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = inputs["negative_prompt"]
        output = sd_pipe(**inputs)
        image_slice_2 = output.images[(0), -3:, -3:, (-1)]
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 0.0001
        inputs = self.get_dummy_inputs()
        inputs["negative_prompt"] = "negative prompt"
        inputs["negative_prompt_2"] = "different negative prompt"
        output = sd_pipe(**inputs)
        image_slice_3 = output.images[(0), -3:, -3:, (-1)]
        assert np.abs(image_slice_1.flatten() - image_slice_3.flatten()).max() > 0.0001
