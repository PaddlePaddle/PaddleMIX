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

import ppdiffusers
from ppdiffusers import (
    AutoencoderKL,
    EulerDiscreteScheduler,
    StableDiffusionLatentUpscalePipeline,
    UNet2DConditionModel,
)
from ppdiffusers.schedulers import KarrasDiffusionSchedulers
from ppdiffusers.transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer
from ppdiffusers.utils import floats_tensor, slow
from ppdiffusers.utils.testing_utils import enable_full_determinism, require_paddle_gpu

from ..pipeline_params import (
    TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS,
    TEXT_GUIDED_IMAGE_VARIATION_PARAMS,
)
from ..test_pipelines_common import (
    PipelineKarrasSchedulerTesterMixin,
    PipelineLatentTesterMixin,
    PipelineTesterMixin,
)

enable_full_determinism()


def check_same_shape(tensor_list):
    shapes = [tensor.shape for tensor in tensor_list]
    return all(shape == shapes[0] for shape in shapes[1:])


class StableDiffusionLatentUpscalePipelineFastTests(
    PipelineLatentTesterMixin, PipelineKarrasSchedulerTesterMixin, PipelineTesterMixin, unittest.TestCase
):
    pipeline_class = StableDiffusionLatentUpscalePipeline
    params = TEXT_GUIDED_IMAGE_VARIATION_PARAMS - {
        "height",
        "width",
        "cross_attention_kwargs",
        "negative_prompt_embeds",
        "prompt_embeds",
    }
    required_optional_params = PipelineTesterMixin.required_optional_params - {"num_images_per_prompt"}
    batch_params = TEXT_GUIDED_IMAGE_VARIATION_BATCH_PARAMS

    image_params = frozenset(
        []
    )  # TO-DO: update image_params once pipeline is refactored with VaeImageProcessor.preprocess
    image_latents_params = frozenset([])

    @property
    def dummy_image(self):
        batch_size = 1
        num_channels = 4
        sizes = 16, 16
        image = floats_tensor((batch_size, num_channels) + sizes, rng=random.Random(0))
        return image

    def get_dummy_components(self):
        paddle.seed(0)
        model = UNet2DConditionModel(
            act_fn="gelu",
            attention_head_dim=8,
            norm_num_groups=None,
            block_out_channels=[32, 32, 64, 64],
            time_cond_proj_dim=160,
            conv_in_kernel=1,
            conv_out_kernel=1,
            cross_attention_dim=32,
            down_block_types=(
                "KDownBlock2D",
                "KCrossAttnDownBlock2D",
                "KCrossAttnDownBlock2D",
                "KCrossAttnDownBlock2D",
            ),
            in_channels=8,
            mid_block_type=None,
            only_cross_attention=False,
            out_channels=5,
            resnet_time_scale_shift="scale_shift",
            time_embedding_type="fourier",
            timestep_post_act="gelu",
            up_block_types=("KCrossAttnUpBlock2D", "KCrossAttnUpBlock2D", "KCrossAttnUpBlock2D", "KUpBlock2D"),
        )
        vae = AutoencoderKL(
            block_out_channels=[32, 32, 64, 64],
            in_channels=3,
            out_channels=3,
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            latent_channels=4,
        )
        scheduler = EulerDiscreteScheduler(prediction_type="sample")
        text_config = CLIPTextConfig(
            bos_token_id=0,
            eos_token_id=2,
            hidden_size=32,
            intermediate_size=37,
            layer_norm_eps=1e-05,
            num_attention_heads=4,
            num_hidden_layers=5,
            pad_token_id=1,
            vocab_size=1000,
            hidden_act="quick_gelu",
            projection_dim=512,
        )
        text_encoder = CLIPTextModel(text_config).eval()
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
        components = {
            "unet": model.eval(),
            "vae": vae.eval(),
            "scheduler": scheduler,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
        }
        return components

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)

        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "image": self.dummy_image.cpu(),
            "generator": generator,
            "num_inference_steps": 2,
            "output_type": "np",
        }
        return inputs

    def test_inference(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = pipe(**inputs).images
        image_slice = image[0, -3:, -3:, -1]
        self.assertEqual(image.shape, (1, 256, 256, 3))
        expected_slice = np.array(
            [
                0.5665861368179321,
                0.7449524402618408,
                0.0,
                0.1325536072254181,
                0.4274534583091736,
                0.0,
                0.0,
                0.14426982402801514,
                0.0,
            ]
        )
        max_diff = np.abs(image_slice.flatten() - expected_slice).max()
        self.assertLessEqual(max_diff, 0.001)

    def test_attention_slicing_forward_pass(self):
        super().test_attention_slicing_forward_pass()

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=7e-3)

    def test_pd_np_pil_outputs_equivalent(self):
        pass
        # super().test_pd_np_pil_outputs_equivalent()

    def test_save_load_local(self):
        super().test_save_load_local()

    def test_save_load_optional_components(self):
        super().test_save_load_optional_components()

    def test_karras_schedulers_shape(self):
        skip_schedulers = [
            "DDIMScheduler",
            "DDPMScheduler",
            "PNDMScheduler",
            "HeunDiscreteScheduler",
            "EulerAncestralDiscreteScheduler",
            "KDPM2DiscreteScheduler",
            "KDPM2AncestralDiscreteScheduler",
            "DPMSolverSDEScheduler",
        ]
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.scheduler.register_to_config(skip_prk_steps=True)
        pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        inputs["num_inference_steps"] = 2
        outputs = []
        for scheduler_enum in KarrasDiffusionSchedulers:
            if scheduler_enum.name in skip_schedulers:
                continue
            scheduler_cls = getattr(ppdiffusers, scheduler_enum.name)
            pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
            output = pipe(**inputs)[0]
            outputs.append(output)
        assert check_same_shape(outputs)


@require_paddle_gpu
@slow
class StableDiffusionLatentUpscalePipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        paddle.device.cuda.empty_cache()

    def test_latent_upscaler_fp16(self):
        pass

    def test_latent_upscaler_fp16_image(self):
        pass
