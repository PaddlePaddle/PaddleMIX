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

from ppdiffusers import (
    KandinskyV22CombinedPipeline,
    KandinskyV22Img2ImgCombinedPipeline,
    KandinskyV22InpaintCombinedPipeline,
)
from ppdiffusers.utils.testing_utils import enable_full_determinism

from ..test_pipelines_common import PipelineTesterMixin
from .test_kandinsky import Dummies
from .test_kandinsky_img2img import Dummies as Img2ImgDummies
from .test_kandinsky_inpaint import Dummies as InpaintDummies
from .test_kandinsky_prior import Dummies as PriorDummies

enable_full_determinism()


class KandinskyV22PipelineCombinedFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KandinskyV22CombinedPipeline
    params = ["prompt"]
    batch_params = ["prompt", "negative_prompt"]
    required_optional_params = [
        "generator",
        "height",
        "width",
        "latents",
        "guidance_scale",
        "negative_prompt",
        "num_inference_steps",
        "return_dict",
        "guidance_scale",
        "num_images_per_prompt",
        "output_type",
        "return_dict",
    ]

    def get_dummy_components(self):
        dummy = Dummies()
        prior_dummy = PriorDummies()
        components = dummy.get_dummy_components()
        components.update({f"prior_{k}": v for k, v in prior_dummy.get_dummy_components().items()})
        return components

    def get_dummy_inputs(self, seed=0):
        prior_dummy = PriorDummies()
        inputs = prior_dummy.get_dummy_inputs(seed=seed)
        inputs.update({"height": 64, "width": 64})
        return inputs

    def test_kandinsky(self):
        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)
        pipe.set_progress_bar_config(disable=None)
        output = pipe(**self.get_dummy_inputs())
        image = output.images
        image_from_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]
        image_slice = image[(0), -3:, -3:, (-1)]
        image_from_tuple_slice = image_from_tuple[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.8085, 0.99, 1.0, 1.0])
        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"
        assert (
            np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01
        ), f" expected_slice {expected_slice}, but got {image_from_tuple_slice.flatten()}"

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=0.01)


class KandinskyV22PipelineImg2ImgCombinedFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KandinskyV22Img2ImgCombinedPipeline
    params = ["prompt", "image"]
    batch_params = ["prompt", "negative_prompt", "image"]
    required_optional_params = [
        "generator",
        "height",
        "width",
        "latents",
        "guidance_scale",
        "negative_prompt",
        "num_inference_steps",
        "return_dict",
        "guidance_scale",
        "num_images_per_prompt",
        "output_type",
        "return_dict",
    ]

    def get_dummy_components(self):
        dummy = Img2ImgDummies()
        prior_dummy = PriorDummies()
        components = dummy.get_dummy_components()
        components.update({f"prior_{k}": v for k, v in prior_dummy.get_dummy_components().items()})
        return components

    def get_dummy_inputs(self, seed=0):
        prior_dummy = PriorDummies()
        dummy = Img2ImgDummies()
        inputs = prior_dummy.get_dummy_inputs(seed=seed)
        inputs.update(dummy.get_dummy_inputs(seed=seed))
        inputs.pop("image_embeds")
        inputs.pop("negative_image_embeds")
        return inputs

    def test_kandinsky(self):

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        pipe.set_progress_bar_config(disable=None)
        output = pipe(**self.get_dummy_inputs())
        image = output.images
        image_from_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]
        image_slice = image[(0), -3:, -3:, (-1)]
        image_from_tuple_slice = image_from_tuple[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([0.529, 0.326, 1.0, 0.06, 0.78, 1.0, 0.167, 1.0, 1.0])
        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"
        assert (
            np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01
        ), f" expected_slice {expected_slice}, but got {image_from_tuple_slice.flatten()}"

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=0.01)


class KandinskyV22PipelineInpaintCombinedFastTests(PipelineTesterMixin, unittest.TestCase):
    pipeline_class = KandinskyV22InpaintCombinedPipeline
    params = ["prompt", "image", "mask_image"]
    batch_params = ["prompt", "negative_prompt", "image", "mask_image"]
    required_optional_params = [
        "generator",
        "height",
        "width",
        "latents",
        "guidance_scale",
        "negative_prompt",
        "num_inference_steps",
        "return_dict",
        "guidance_scale",
        "num_images_per_prompt",
        "output_type",
        "return_dict",
    ]

    def get_dummy_components(self):
        dummy = InpaintDummies()
        prior_dummy = PriorDummies()
        components = dummy.get_dummy_components()
        components.update({f"prior_{k}": v for k, v in prior_dummy.get_dummy_components().items()})
        return components

    def get_dummy_inputs(self, seed=0):
        prior_dummy = PriorDummies()
        dummy = InpaintDummies()
        inputs = prior_dummy.get_dummy_inputs(seed=seed)
        inputs.update(dummy.get_dummy_inputs(seed=seed))
        inputs.pop("image_embeds")
        inputs.pop("negative_image_embeds")
        return inputs

    def test_kandinsky(self):

        components = self.get_dummy_components()
        pipe = self.pipeline_class(**components)

        pipe.set_progress_bar_config(disable=None)
        output = pipe(**self.get_dummy_inputs())
        image = output.images
        image_from_tuple = pipe(**self.get_dummy_inputs(), return_dict=False)[0]
        image_slice = image[(0), -3:, -3:, (-1)]
        image_from_tuple_slice = image_from_tuple[(0), -3:, -3:, (-1)]
        assert image.shape == (1, 64, 64, 3)
        expected_slice = np.array([1.0, 0.907, 0.310, 0.970, 1.0, 0.797, 0.736, 1.0, 0.843])
        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"
        assert (
            np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01
        ), f" expected_slice {expected_slice}, but got {image_from_tuple_slice.flatten()}"

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=0.01)
