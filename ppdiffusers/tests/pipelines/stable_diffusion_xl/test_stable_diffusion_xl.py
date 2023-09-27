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

import copy
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
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from ppdiffusers.utils.testing_utils import enable_full_determinism

from ..pipeline_params import (
    TEXT_TO_IMAGE_BATCH_PARAMS,
    TEXT_TO_IMAGE_IMAGE_PARAMS,
    TEXT_TO_IMAGE_PARAMS,
)
from ..test_pipelines_common import PipelineLatentTesterMixin, PipelineTesterMixin

enable_full_determinism()


class StableDiffusionXLPipelineFastTests(PipelineLatentTesterMixin, PipelineTesterMixin, unittest.TestCase):
    pipeline_class = StableDiffusionXLPipeline
    params = TEXT_TO_IMAGE_PARAMS
    batch_params = TEXT_TO_IMAGE_BATCH_PARAMS
    image_params = TEXT_TO_IMAGE_IMAGE_PARAMS
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
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "text_encoder_2": text_encoder_2,
            "tokenizer_2": tokenizer_2,
        }
        return components

    def get_dummy_inputs(self, seed=0):
        generator = paddle.Generator().manual_seed(seed)
        inputs = {
            "prompt": "A painting of a squirrel eating a burger",
            "generator": generator,
            "num_inference_steps": 2,
            "guidance_scale": 5.0,
            "output_type": "np",
        }
        return inputs

    def test_stable_diffusion_xl_euler(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        image = sd_pipe(**inputs).images
        image_slice = image[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.0, 0.0, 0.3135, 0.1048, 0.1784, 0.4079, 0.0893, 0.0314, 0.3707])
        assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

    def test_stable_diffusion_xl_prompt_embeds(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**components)
        sd_pipe.set_progress_bar_config(disable=None)
        inputs = self.get_dummy_inputs()
        inputs["prompt"] = 2 * [inputs["prompt"]]
        inputs["num_images_per_prompt"] = 2
        output = sd_pipe(**inputs)
        image_slice_1 = output.images[(0), -3:, -3:, (-1)]
        inputs = self.get_dummy_inputs()
        prompt = 2 * [inputs.pop("prompt")]
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = sd_pipe.encode_prompt(prompt)
        output = sd_pipe(
            **inputs,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )
        image_slice_2 = output.images[(0), -3:, -3:, (-1)]
        assert np.abs(image_slice_1.flatten() - image_slice_2.flatten()).max() < 0.0001

    def test_stable_diffusion_xl_negative_prompt_embeds(self):
        components = self.get_dummy_components()
        sd_pipe = StableDiffusionXLPipeline(**components)
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

    def test_attention_slicing_forward_pass(self):
        # super().test_attention_slicing_forward_pass()
        pass

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical()

    # @require_paddle_gpu
    # def test_stable_diffusion_xl_offloads(self):
    #     pipes = []
    #     components = self.get_dummy_components()
    #     sd_pipe = StableDiffusionXLPipeline(**components)
    #     pipes.append(sd_pipe)
    #     components = self.get_dummy_components()
    #     sd_pipe = StableDiffusionXLPipeline(**components)
    #     sd_pipe.enable_model_cpu_offload()
    #     pipes.append(sd_pipe)
    #     components = self.get_dummy_components()
    #     sd_pipe = StableDiffusionXLPipeline(**components)
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

    def test_stable_diffusion_two_xl_mixture_of_denoiser(self):
        components = self.get_dummy_components()
        pipe_1 = StableDiffusionXLPipeline(**components)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = StableDiffusionXLImg2ImgPipeline(**components)
        pipe_2.unet.set_default_attn_processor()

        def assert_run_mixture(
            num_steps,
            split,
            scheduler_cls_orig,
            expected_tss,
            num_train_timesteps=pipe_1.scheduler.config.num_train_timesteps,
        ):
            inputs = self.get_dummy_inputs()
            inputs["num_inference_steps"] = num_steps

            class scheduler_cls(scheduler_cls_orig):
                pass

            pipe_1.scheduler = scheduler_cls.from_config(pipe_1.scheduler.config)
            pipe_2.scheduler = scheduler_cls.from_config(pipe_2.scheduler.config)
            pipe_1.scheduler.set_timesteps(num_steps)
            expected_steps = pipe_1.scheduler.timesteps.tolist()
            expected_steps_1 = list(filter(lambda ts: ts >= split, expected_tss))
            expected_steps_2 = list(filter(lambda ts: ts < split, expected_tss))
            done_steps = []
            old_step = copy.copy(scheduler_cls.step)

            def new_step(self, *args, **kwargs):
                done_steps.append(args[1].cpu().item())
                return old_step(self, *args, **kwargs)

            scheduler_cls.step = new_step
            inputs_1 = {**inputs, **{"denoising_end": 1.0 - split / num_train_timesteps, "output_type": "latent"}}
            latents = pipe_1(**inputs_1).images[0]
            assert expected_steps_1 == done_steps, f"Failure with {scheduler_cls.__name__} and {num_steps} and {split}"
            inputs_2 = {**inputs, **{"denoising_start": 1.0 - split / num_train_timesteps, "image": latents}}
            pipe_2(**inputs_2).images[0]
            assert expected_steps_2 == done_steps[len(expected_steps_1) :]
            assert expected_steps == done_steps, f"Failure with {scheduler_cls.__name__} and {num_steps} and {split}"

        steps = 10
        for split in [300, 500, 700]:
            for scheduler_cls_timesteps in [
                (DDIMScheduler, [901, 801, 701, 601, 501, 401, 301, 201, 101, 1]),
                (EulerDiscreteScheduler, [901, 801, 701, 601, 501, 401, 301, 201, 101, 1]),
                (DPMSolverMultistepScheduler, [901, 811, 721, 631, 541, 451, 361, 271, 181, 91]),
                (UniPCMultistepScheduler, [901, 811, 721, 631, 541, 451, 361, 271, 181, 91]),
                (
                    HeunDiscreteScheduler,
                    [
                        901.0,
                        801.0,
                        801.0,
                        701.0,
                        701.0,
                        601.0,
                        601.0,
                        501.0,
                        501.0,
                        401.0,
                        401.0,
                        301.0,
                        301.0,
                        201.0,
                        201.0,
                        101.0,
                        101.0,
                        1.0,
                        1.0,
                    ],
                ),
            ]:
                assert_run_mixture(steps, split, scheduler_cls_timesteps[0], scheduler_cls_timesteps[1])
        steps = 25
        for split in [300, 500, 700]:
            for scheduler_cls_timesteps in [
                (
                    DDIMScheduler,
                    [
                        961,
                        921,
                        881,
                        841,
                        801,
                        761,
                        721,
                        681,
                        641,
                        601,
                        561,
                        521,
                        481,
                        441,
                        401,
                        361,
                        321,
                        281,
                        241,
                        201,
                        161,
                        121,
                        81,
                        41,
                        1,
                    ],
                ),
                (
                    EulerDiscreteScheduler,
                    [
                        961.0,
                        921.0,
                        881.0,
                        841.0,
                        801.0,
                        761.0,
                        721.0,
                        681.0,
                        641.0,
                        601.0,
                        561.0,
                        521.0,
                        481.0,
                        441.0,
                        401.0,
                        361.0,
                        321.0,
                        281.0,
                        241.0,
                        201.0,
                        161.0,
                        121.0,
                        81.0,
                        41.0,
                        1.0,
                    ],
                ),
                (
                    DPMSolverMultistepScheduler,
                    [
                        951,
                        913,
                        875,
                        837,
                        799,
                        761,
                        723,
                        685,
                        647,
                        609,
                        571,
                        533,
                        495,
                        457,
                        419,
                        381,
                        343,
                        305,
                        267,
                        229,
                        191,
                        153,
                        115,
                        77,
                        39,
                    ],
                ),
                (
                    UniPCMultistepScheduler,
                    [
                        951,
                        913,
                        875,
                        837,
                        799,
                        761,
                        723,
                        685,
                        647,
                        609,
                        571,
                        533,
                        495,
                        457,
                        419,
                        381,
                        343,
                        305,
                        267,
                        229,
                        191,
                        153,
                        115,
                        77,
                        39,
                    ],
                ),
                (
                    HeunDiscreteScheduler,
                    [
                        961.0,
                        921.0,
                        921.0,
                        881.0,
                        881.0,
                        841.0,
                        841.0,
                        801.0,
                        801.0,
                        761.0,
                        761.0,
                        721.0,
                        721.0,
                        681.0,
                        681.0,
                        641.0,
                        641.0,
                        601.0,
                        601.0,
                        561.0,
                        561.0,
                        521.0,
                        521.0,
                        481.0,
                        481.0,
                        441.0,
                        441.0,
                        401.0,
                        401.0,
                        361.0,
                        361.0,
                        321.0,
                        321.0,
                        281.0,
                        281.0,
                        241.0,
                        241.0,
                        201.0,
                        201.0,
                        161.0,
                        161.0,
                        121.0,
                        121.0,
                        81.0,
                        81.0,
                        41.0,
                        41.0,
                        1.0,
                        1.0,
                    ],
                ),
            ]:
                assert_run_mixture(steps, split, scheduler_cls_timesteps[0], scheduler_cls_timesteps[1])

    def test_stable_diffusion_three_xl_mixture_of_denoiser(self):
        components = self.get_dummy_components()
        pipe_1 = StableDiffusionXLPipeline(**components)
        pipe_1.unet.set_default_attn_processor()
        pipe_2 = StableDiffusionXLImg2ImgPipeline(**components)
        pipe_2.unet.set_default_attn_processor()
        pipe_3 = StableDiffusionXLImg2ImgPipeline(**components)
        pipe_3.unet.set_default_attn_processor()

        def assert_run_mixture(
            num_steps,
            split_1,
            split_2,
            scheduler_cls_orig,
            num_train_timesteps=pipe_1.scheduler.config.num_train_timesteps,
        ):
            inputs = self.get_dummy_inputs()
            inputs["num_inference_steps"] = num_steps

            class scheduler_cls(scheduler_cls_orig):
                pass

            pipe_1.scheduler = scheduler_cls.from_config(pipe_1.scheduler.config)
            pipe_2.scheduler = scheduler_cls.from_config(pipe_2.scheduler.config)
            pipe_3.scheduler = scheduler_cls.from_config(pipe_3.scheduler.config)
            pipe_1.scheduler.set_timesteps(num_steps)
            expected_steps = pipe_1.scheduler.timesteps.tolist()
            split_1_ts = num_train_timesteps - int(round(num_train_timesteps * split_1))
            split_2_ts = num_train_timesteps - int(round(num_train_timesteps * split_2))
            expected_steps_1 = expected_steps[:split_1_ts]
            expected_steps_2 = expected_steps[split_1_ts:split_2_ts]
            expected_steps_3 = expected_steps[split_2_ts:]
            expected_steps_1 = list(filter(lambda ts: ts >= split_1_ts, expected_steps))
            expected_steps_2 = list(filter(lambda ts: ts >= split_2_ts and ts < split_1_ts, expected_steps))
            expected_steps_3 = list(filter(lambda ts: ts < split_2_ts, expected_steps))
            done_steps = []
            old_step = copy.copy(scheduler_cls.step)

            def new_step(self, *args, **kwargs):
                done_steps.append(args[1].cpu().item())
                return old_step(self, *args, **kwargs)

            scheduler_cls.step = new_step
            inputs_1 = {**inputs, **{"denoising_end": split_1, "output_type": "latent"}}
            latents = pipe_1(**inputs_1).images[0]
            assert (
                expected_steps_1 == done_steps
            ), f"Failure with {scheduler_cls.__name__} and {num_steps} and {split_1} and {split_2}"
            with self.assertRaises(ValueError) as cm:
                inputs_2 = {
                    **inputs,
                    **{
                        "denoising_start": split_2,
                        "denoising_end": split_1,
                        "image": latents,
                        "output_type": "latent",
                    },
                }
                pipe_2(**inputs_2).images[0]
            assert "cannot be larger than or equal to `denoising_end`" in str(cm.exception)
            inputs_2 = {
                **inputs,
                **{"denoising_start": split_1, "denoising_end": split_2, "image": latents, "output_type": "latent"},
            }
            pipe_2(**inputs_2).images[0]
            assert expected_steps_2 == done_steps[len(expected_steps_1) :]
            inputs_3 = {**inputs, **{"denoising_start": split_2, "image": latents}}
            pipe_3(**inputs_3).images[0]
            assert expected_steps_3 == done_steps[len(expected_steps_1) + len(expected_steps_2) :]
            assert (
                expected_steps == done_steps
            ), f"Failure with {scheduler_cls.__name__} and {num_steps} and {split_1} and {split_2}"

        for steps in [7, 11, 20]:
            for split_1, split_2 in zip([0.19, 0.32], [0.81, 0.68]):
                for scheduler_cls in [
                    DDIMScheduler,
                    EulerDiscreteScheduler,
                    DPMSolverMultistepScheduler,
                    UniPCMultistepScheduler,
                    HeunDiscreteScheduler,
                ]:
                    assert_run_mixture(steps, split_1, split_2, scheduler_cls)

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
