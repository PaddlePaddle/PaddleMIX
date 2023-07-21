# # Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import tempfile
# import unittest

# import numpy as np

# from ppdiffusers import (
#     DDIMScheduler,
#     DPMSolverMultistepScheduler,
#     EulerAncestralDiscreteScheduler,
#     EulerDiscreteScheduler,
#     LMSDiscreteScheduler,
#     OnnxStableDiffusionPipeline,
#     PNDMScheduler,
# )
# from ppdiffusers.utils.testing_utils import (
#     is_onnx_available,
#     nightly,
#     require_onnxruntime,
#     require_paddle_gpu,
# )

# from ...test_pipelines_onnx_common import OnnxPipelineTesterMixin

# if is_onnx_available():
#     import onnxruntime as ort

# class OnnxStableDiffusionPipelineFastTests(OnnxPipelineTesterMixin, unittest.TestCase):
#     hub_checkpoint = "hf-internal-testing/tiny-random-OnnxStableDiffusionPipeline"

#     def get_dummy_inputs(self, seed=0):
#         generator = np.random.RandomState(seed)
#         inputs = {
#             "prompt": "A painting of a squirrel eating a burger",
#             "generator": generator,
#             "num_inference_steps": 2,
#             "guidance_scale": 7.5,
#             "output_type": "numpy",
#         }
#         return inputs

#     def test_pipeline_default_ddim(self):
#         pipe = OnnxStableDiffusionPipeline.from_pretrained(self.hub_checkpoint, provider="CPUExecutionProvider")
#         pipe.set_progress_bar_config(disable=None)
#         inputs = self.get_dummy_inputs()
#         image = pipe(**inputs).images
#         image_slice = image[0, -3:, -3:, -1]
#         assert image.shape == (1, 128, 128, 3)
#         expected_slice = np.array([0.65072, 0.58492, 0.48219, 0.55521, 0.5318, 0.55939, 0.50697, 0.398, 0.46455])
#         assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

#     def test_pipeline_pndm(self):
#         pipe = OnnxStableDiffusionPipeline.from_pretrained(self.hub_checkpoint, provider="CPUExecutionProvider")
#         pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config, skip_prk_steps=True)
#         pipe.set_progress_bar_config(disable=None)
#         inputs = self.get_dummy_inputs()
#         image = pipe(**inputs).images
#         image_slice = image[0, -3:, -3:, -1]
#         assert image.shape == (1, 128, 128, 3)
#         expected_slice = np.array([0.65863, 0.59425, 0.49326, 0.56313, 0.53875, 0.56627, 0.51065, 0.39777, 0.4633])
#         assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

#     def test_pipeline_lms(self):
#         pipe = OnnxStableDiffusionPipeline.from_pretrained(self.hub_checkpoint, provider="CPUExecutionProvider")
#         pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
#         pipe.set_progress_bar_config(disable=None)
#         inputs = self.get_dummy_inputs()
#         image = pipe(**inputs).images
#         image_slice = image[0, -3:, -3:, -1]
#         assert image.shape == (1, 128, 128, 3)
#         expected_slice = np.array([0.53755, 0.60786, 0.47402, 0.49488, 0.51869, 0.49819, 0.47985, 0.38957, 0.44279])
#         assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

#     def test_pipeline_euler(self):
#         pipe = OnnxStableDiffusionPipeline.from_pretrained(self.hub_checkpoint, provider="CPUExecutionProvider")
#         pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
#         pipe.set_progress_bar_config(disable=None)
#         inputs = self.get_dummy_inputs()
#         image = pipe(**inputs).images
#         image_slice = image[0, -3:, -3:, -1]
#         assert image.shape == (1, 128, 128, 3)
#         expected_slice = np.array([0.53755, 0.60786, 0.47402, 0.49488, 0.51869, 0.49819, 0.47985, 0.38957, 0.44279])
#         assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

#     def test_pipeline_euler_ancestral(self):
#         pipe = OnnxStableDiffusionPipeline.from_pretrained(self.hub_checkpoint, provider="CPUExecutionProvider")
#         pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
#         pipe.set_progress_bar_config(disable=None)
#         inputs = self.get_dummy_inputs()
#         image = pipe(**inputs).images
#         image_slice = image[0, -3:, -3:, -1]
#         assert image.shape == (1, 128, 128, 3)
#         expected_slice = np.array([0.53817, 0.60812, 0.47384, 0.4953, 0.51894, 0.49814, 0.47984, 0.38958, 0.44271])
#         assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

#     def test_pipeline_dpm_multistep(self):
#         pipe = OnnxStableDiffusionPipeline.from_pretrained(self.hub_checkpoint, provider="CPUExecutionProvider")
#         pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
#         pipe.set_progress_bar_config(disable=None)
#         inputs = self.get_dummy_inputs()
#         image = pipe(**inputs).images
#         image_slice = image[0, -3:, -3:, -1]
#         assert image.shape == (1, 128, 128, 3)
#         expected_slice = np.array([0.53895, 0.60808, 0.47933, 0.49608, 0.51886, 0.4995, 0.48053, 0.38957, 0.442])
#         assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01

# @nightly
# @require_onnxruntime
# @require_paddle_gpu
# class OnnxStableDiffusionPipelineIntegrationTests(unittest.TestCase):
#     @property
#     def gpu_provider(self):
#         return "CUDAExecutionProvider", {"gpu_mem_limit": "15000000000", "arena_extend_strategy": "kSameAsRequested"}

#     @property
#     def gpu_options(self):
#         options = ort.SessionOptions()
#         options.enable_mem_pattern = False
#         return options

#     def test_inference_default_pndm(self):
#         sd_pipe = OnnxStableDiffusionPipeline.from_pretrained(
#             "CompVis/stable-diffusion-v1-4",
#             revision="onnx",
#             safety_checker=None,
#             feature_extractor=None,
#             provider=self.gpu_provider,
#             sess_options=self.gpu_options,
#         )
#         sd_pipe.set_progress_bar_config(disable=None)
#         prompt = "A painting of a squirrel eating a burger"
#         np.random.seed(0)
#         output = sd_pipe([prompt], guidance_scale=6.0, num_inference_steps=10, output_type="np")
#         image = output.images
#         image_slice = image[0, -3:, -3:, -1]
#         assert image.shape == (1, 512, 512, 3)
#         expected_slice = np.array([0.0452, 0.039, 0.0087, 0.035, 0.0617, 0.0364, 0.0544, 0.0523, 0.072])
#         assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

#     def test_inference_ddim(self):
#         ddim_scheduler = DDIMScheduler.from_pretrained(
#             "runwayml/stable-diffusion-v1-5", subfolder="scheduler", revision="onnx"
#         )
#         sd_pipe = OnnxStableDiffusionPipeline.from_pretrained(
#             "runwayml/stable-diffusion-v1-5",
#             revision="onnx",
#             scheduler=ddim_scheduler,
#             safety_checker=None,
#             feature_extractor=None,
#             provider=self.gpu_provider,
#             sess_options=self.gpu_options,
#         )
#         sd_pipe.set_progress_bar_config(disable=None)
#         prompt = "open neural network exchange"
#         generator = np.random.RandomState(0)
#         output = sd_pipe([prompt], guidance_scale=7.5, num_inference_steps=10, generator=generator, output_type="np")
#         image = output.images
#         image_slice = image[0, -3:, -3:, -1]
#         assert image.shape == (1, 512, 512, 3)
#         expected_slice = np.array([0.2867, 0.1974, 0.1481, 0.7294, 0.7251, 0.6667, 0.4194, 0.5642, 0.6486])
#         assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

#     def test_inference_k_lms(self):
#         lms_scheduler = LMSDiscreteScheduler.from_pretrained(
#             "runwayml/stable-diffusion-v1-5", subfolder="scheduler", revision="onnx"
#         )
#         sd_pipe = OnnxStableDiffusionPipeline.from_pretrained(
#             "runwayml/stable-diffusion-v1-5",
#             revision="onnx",
#             scheduler=lms_scheduler,
#             safety_checker=None,
#             feature_extractor=None,
#             provider=self.gpu_provider,
#             sess_options=self.gpu_options,
#         )
#         sd_pipe.set_progress_bar_config(disable=None)
#         prompt = "open neural network exchange"
#         generator = np.random.RandomState(0)
#         output = sd_pipe([prompt], guidance_scale=7.5, num_inference_steps=10, generator=generator, output_type="np")
#         image = output.images
#         image_slice = image[0, -3:, -3:, -1]
#         assert image.shape == (1, 512, 512, 3)
#         expected_slice = np.array([0.2306, 0.1959, 0.1593, 0.6549, 0.6394, 0.5408, 0.5065, 0.601, 0.6161])
#         assert np.abs(image_slice.flatten() - expected_slice).max() < 0.001

#     def test_intermediate_state(self):
#         number_of_steps = 0

#         def test_callback_fn(step: int, timestep: int, latents: np.ndarray) -> None:
#             test_callback_fn.has_been_called = True
#             nonlocal number_of_steps
#             number_of_steps += 1
#             if step == 0:
#                 assert latents.shape == (1, 4, 64, 64)
#                 latents_slice = latents[0, -3:, -3:, -1]
#                 expected_slice = np.array(
#                     [-0.6772, -0.3835, -1.2456, 0.1905, -1.0974, 0.6967, -1.9353, 0.0178, 1.0167]
#                 )
#                 assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.001
#             elif step == 5:
#                 assert latents.shape == (1, 4, 64, 64)
#                 latents_slice = latents[0, -3:, -3:, -1]
#                 expected_slice = np.array(
#                     [-0.3351, 0.2241, -0.1837, -0.2325, -0.6577, 0.3393, -0.0241, 0.5899, 1.3875]
#                 )
#                 assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.001

#         test_callback_fn.has_been_called = False
#         pipe = OnnxStableDiffusionPipeline.from_pretrained(
#             "runwayml/stable-diffusion-v1-5",
#             revision="onnx",
#             safety_checker=None,
#             feature_extractor=None,
#             provider=self.gpu_provider,
#             sess_options=self.gpu_options,
#         )
#         pipe.set_progress_bar_config(disable=None)
#         prompt = "Andromeda galaxy in a bottle"
#         generator = np.random.RandomState(0)
#         pipe(
#             prompt=prompt,
#             num_inference_steps=5,
#             guidance_scale=7.5,
#             generator=generator,
#             callback=test_callback_fn,
#             callback_steps=1,
#         )
#         assert test_callback_fn.has_been_called
#         assert number_of_steps == 6

#     def test_stable_diffusion_no_safety_checker(self):
#         pipe = OnnxStableDiffusionPipeline.from_pretrained(
#             "runwayml/stable-diffusion-v1-5",
#             revision="onnx",
#             safety_checker=None,
#             feature_extractor=None,
#             provider=self.gpu_provider,
#             sess_options=self.gpu_options,
#         )
#         assert isinstance(pipe, OnnxStableDiffusionPipeline)
#         assert pipe.safety_checker is None
#         image = pipe("example prompt", num_inference_steps=2).images[0]
#         assert image is not None
#         with tempfile.TemporaryDirectory() as tmpdirname:
#             pipe.save_pretrained(tmpdirname)
#             pipe = OnnxStableDiffusionPipeline.from_pretrained(tmpdirname)
#         assert pipe.safety_checker is None
#         image = pipe("example prompt", num_inference_steps=2).images[0]
#         assert image is not None
