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

# import unittest

# import numpy as np

# from ppdiffusers import (
#     KandinskyCombinedPipeline,
#     KandinskyImg2ImgCombinedPipeline,
#     KandinskyInpaintCombinedPipeline,
# )
# from ppdiffusers.utils.testing_utils import enable_full_determinism

# from ..test_pipelines_common import PipelineTesterMixin
# from .test_kandinsky import Dummies
# from .test_kandinsky_img2img import Dummies as Img2ImgDummies
# from .test_kandinsky_inpaint import Dummies as InpaintDummies
# from .test_kandinsky_prior import Dummies as PriorDummies

# enable_full_determinism()


# class KandinskyPipelineCombinedFastTests(PipelineTesterMixin, unittest.TestCase):
#     pipeline_class = KandinskyCombinedPipeline
#     params = ["prompt"]
#     batch_params = ["prompt", "negative_prompt"]
#     required_optional_params = [
#         "generator",
#         "height",
#         "width",
#         "latents",
#         "guidance_scale",
#         "negative_prompt",
#         "num_inference_steps",
#         "return_dict",
#         "guidance_scale",
#         "num_images_per_prompt",
#         "output_type",
#         "return_dict",
#     ]
#     test_xformers_attention = False

#     def get_dummy_components(self):
#         dummy = Dummies()
#         prior_dummy = PriorDummies()
#         components = dummy.get_dummy_components()
#         components.update({f"prior_{k}": v for k, v in prior_dummy.get_dummy_components().items()})
#         return components

#     def get_dummy_inputs(self, seed=0):
#         prior_dummy = PriorDummies()
#         inputs = prior_dummy.get_dummy_inputs(seed=seed)
#         inputs.update({"height": 64, "width": 64})
#         return inputs

#     def test_kandinsky(self):
#         components = self.get_dummy_components()
#         pipe = self.pipeline_class(**components)
#         pipe.set_progress_bar_config(disable=None)
#         output = pipe(**self.get_dummy_inputs())
#         image = output.images
#         image_from_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]
#         image_slice = image[(0), -3:, -3:, (-1)]
#         image_from_tuple_slice = image_from_tuple[(0), -3:, -3:, (-1)]
#         assert image.shape == (1, 64, 64, 3)
#         expected_slice = np.array([0.0, 0.0, 0.6777, 0.1363, 0.3624, 0.7868, 0.3869, 0.3395, 0.5068])
#         assert (
#             np.abs(image_slice.flatten() - expected_slice).max() < 0.01
#         ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"
#         assert (
#             np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01
#         ), f" expected_slice {expected_slice}, but got {image_from_tuple_slice.flatten()}"

#     def test_inference_batch_single_identical(self):
#         super().test_inference_batch_single_identical(expected_max_diff=0.01)


# class KandinskyPipelineImg2ImgCombinedFastTests(PipelineTesterMixin, unittest.TestCase):
#     pipeline_class = KandinskyImg2ImgCombinedPipeline
#     params = ["prompt", "image"]
#     batch_params = ["prompt", "negative_prompt", "image"]
#     required_optional_params = [
#         "generator",
#         "height",
#         "width",
#         "latents",
#         "guidance_scale",
#         "negative_prompt",
#         "num_inference_steps",
#         "return_dict",
#         "guidance_scale",
#         "num_images_per_prompt",
#         "output_type",
#         "return_dict",
#     ]
#     test_xformers_attention = False

#     def get_dummy_components(self):
#         dummy = Img2ImgDummies()
#         prior_dummy = PriorDummies()
#         components = dummy.get_dummy_components()
#         components.update({f"prior_{k}": v for k, v in prior_dummy.get_dummy_components().items()})
#         return components

#     def get_dummy_inputs(self, seed=0):
#         prior_dummy = PriorDummies()
#         dummy = Img2ImgDummies()
#         inputs = prior_dummy.get_dummy_inputs(seed=seed)
#         inputs.update(dummy.get_dummy_inputs(seed=seed))
#         inputs.pop("image_embeds")
#         inputs.pop("negative_image_embeds")
#         return inputs

#     def test_kandinsky(self):
#         components = self.get_dummy_components()
#         pipe = self.pipeline_class(**components)
#         pipe.set_progress_bar_config(disable=None)
#         output = pipe(**self.get_dummy_inputs())
#         image = output.images
#         image_from_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]
#         image_slice = image[(0), -3:, -3:, (-1)]
#         image_from_tuple_slice = image_from_tuple[(0), -3:, -3:, (-1)]
#         assert image.shape == (1, 64, 64, 3)
#         expected_slice = np.array([0.426, 0.3596, 0.4571, 0.389, 0.4087, 0.5137, 0.4819, 0.4116, 0.5053])
#         assert (
#             np.abs(image_slice.flatten() - expected_slice).max() < 0.01
#         ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"
#         assert (
#             np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01
#         ), f" expected_slice {expected_slice}, but got {image_from_tuple_slice.flatten()}"

#     def test_inference_batch_single_identical(self):
#         super().test_inference_batch_single_identical(expected_max_diff=0.01)


# class KandinskyPipelineInpaintCombinedFastTests(PipelineTesterMixin, unittest.TestCase):
#     pipeline_class = KandinskyInpaintCombinedPipeline
#     params = ["prompt", "image", "mask_image"]
#     batch_params = ["prompt", "negative_prompt", "image", "mask_image"]
#     required_optional_params = [
#         "generator",
#         "height",
#         "width",
#         "latents",
#         "guidance_scale",
#         "negative_prompt",
#         "num_inference_steps",
#         "return_dict",
#         "guidance_scale",
#         "num_images_per_prompt",
#         "output_type",
#         "return_dict",
#     ]
#     test_xformers_attention = False

#     def get_dummy_components(self):
#         dummy = InpaintDummies()
#         prior_dummy = PriorDummies()
#         components = dummy.get_dummy_components()
#         components.update({f"prior_{k}": v for k, v in prior_dummy.get_dummy_components().items()})
#         return components

#     def get_dummy_inputs(self, seed=0):
#         prior_dummy = PriorDummies()
#         dummy = InpaintDummies()
#         inputs = prior_dummy.get_dummy_inputs(seed=seed)
#         inputs.update(dummy.get_dummy_inputs(seed=seed))
#         inputs.pop("image_embeds")
#         inputs.pop("negative_image_embeds")
#         return inputs

#     def test_kandinsky(self):
#         components = self.get_dummy_components()
#         pipe = self.pipeline_class(**components)
#         pipe.set_progress_bar_config(disable=None)
#         output = pipe(**self.get_dummy_inputs())
#         image = output.images
#         image_from_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]
#         image_slice = image[(0), -3:, -3:, (-1)]
#         image_from_tuple_slice = image_from_tuple[(0), -3:, -3:, (-1)]
#         assert image.shape == (1, 64, 64, 3)
#         expected_slice = np.array([0.0477, 0.0808, 0.2972, 0.2705, 0.362, 0.6247, 0.4464, 0.287, 0.353])
#         assert (
#             np.abs(image_slice.flatten() - expected_slice).max() < 0.01
#         ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"
#         assert (
#             np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01
#         ), f" expected_slice {expected_slice}, but got {image_from_tuple_slice.flatten()}"

#     def test_inference_batch_single_identical(self):
#         super().test_inference_batch_single_identical(expected_max_diff=0.01)
