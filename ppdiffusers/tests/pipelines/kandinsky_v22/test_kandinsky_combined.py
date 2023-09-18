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
from ppdiffusers.utils.testing_utils import enable_full_determinism, require_paddle_gpu

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
        expected_slice = np.array([0.3013, 0.0471, 0.5176, 0.1817, 0.2566, 0.7076, 0.6712, 0.4421, 0.7503])
        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"
        assert (
            np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01
        ), f" expected_slice {expected_slice}, but got {image_from_tuple_slice.flatten()}"

    @require_paddle_gpu
    def test_offloads(self):
        pipes = []
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        pipes.append(sd_pipe)
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe.enable_model_cpu_offload()
        pipes.append(sd_pipe)
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe.enable_sequential_cpu_offload()
        pipes.append(sd_pipe)
        image_slices = []
        for pipe in pipes:
            inputs = self.get_dummy_inputs()
            image = pipe(**inputs).images
            image_slices.append(image[(0), -3:, -3:, (-1)].flatten())
        assert np.abs(image_slices[0] - image_slices[1]).max() < 0.001
        assert np.abs(image_slices[0] - image_slices[2]).max() < 0.001

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
        expected_slice = np.array([0.4353, 0.471, 0.5128, 0.4806, 0.5054, 0.5348, 0.5224, 0.4603, 0.5025])
        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"
        assert (
            np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01
        ), f" expected_slice {expected_slice}, but got {image_from_tuple_slice.flatten()}"

    @require_paddle_gpu
    def test_offloads(self):
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
            inputs = self.get_dummy_inputs()
            image = pipe(**inputs).images
            image_slices.append(image[(0), -3:, -3:, (-1)].flatten())
        assert np.abs(image_slices[0] - image_slices[1]).max() < 0.001
        assert np.abs(image_slices[0] - image_slices[2]).max() < 0.001

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
        expected_slice = np.array([0.5039, 0.4926, 0.4898, 0.4978, 0.4838, 0.4942, 0.4738, 0.4702, 0.4816])
        assert (
            np.abs(image_slice.flatten() - expected_slice).max() < 0.01
        ), f" expected_slice {expected_slice}, but got {image_slice.flatten()}"
        assert (
            np.abs(image_from_tuple_slice.flatten() - expected_slice).max() < 0.01
        ), f" expected_slice {expected_slice}, but got {image_from_tuple_slice.flatten()}"

    @require_paddle_gpu
    def test_offloads(self):
        pipes = []
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        pipes.append(sd_pipe)
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe.enable_model_cpu_offload()
        pipes.append(sd_pipe)
        components = self.get_dummy_components()
        sd_pipe = self.pipeline_class(**components)
        sd_pipe.enable_sequential_cpu_offload()
        pipes.append(sd_pipe)
        image_slices = []
        for pipe in pipes:
            inputs = self.get_dummy_inputs()
            image = pipe(**inputs).images
            image_slices.append(image[(0), -3:, -3:, (-1)].flatten())
        assert np.abs(image_slices[0] - image_slices[1]).max() < 0.001
        assert np.abs(image_slices[0] - image_slices[2]).max() < 0.001

    def test_inference_batch_single_identical(self):
        super().test_inference_batch_single_identical(expected_max_diff=0.01)