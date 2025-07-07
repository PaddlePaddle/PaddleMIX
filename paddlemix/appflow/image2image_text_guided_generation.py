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

import paddle

from ppdiffusers import (
    StableDiffusionImg2ImgPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionXLImg2ImgPipeline,
)

from .apptask import AppTask


class StableDiffusionImg2ImgTask(AppTask):
    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        self._strength = kwargs.get("strength", 0.75)
        self._guidance_scale = kwargs.get("guidance_scale", 7.5)

        # Default to static mode
        self._static_mode = False
        self._construct_model(model)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """

        # build model
        model_instance = StableDiffusionImg2ImgPipeline.from_pretrained(
            model, safety_checker=None, from_hf_hub=True, from_diffusers=True
        )

        self._model = model_instance

    def _preprocess(self, inputs):
        """ """
        image = inputs.get("image", None)
        assert image is not None, "The image is None"
        prompt = inputs.get("prompt", None)
        assert prompt is not None, "The prompt is None"
        negative_prompt = inputs.get("negative_prompt", None)
        assert negative_prompt is not None, "The negative_prompt is None"

        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """

        result = self._model(
            prompt=inputs["prompt"],
            negative_prompt=inputs["negative_prompt"],
            image=inputs["image"],
            guidance_scale=self._guidance_scale,
            strength=self._strength,
        ).images[0]

        inputs.pop("prompt", None)
        inputs.pop("negative_prompt", None)
        inputs.pop("image", None)
        inputs["result"] = result

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """

        return inputs


class StableDiffusionUpscaleTask(AppTask):
    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        # Default to static mode
        self._static_mode = False
        self._construct_model(model)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """

        # build model
        model_instance = StableDiffusionUpscalePipeline.from_pretrained(model)

        self._model = model_instance

    def _preprocess(self, inputs):
        """ """
        image = inputs.get("image", None)
        assert image is not None, "The image is None"
        prompt = inputs.get("prompt", None)
        assert prompt is not None, "The prompt is None"

        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """

        result = self._model(
            prompt=inputs["prompt"],
            image=inputs["image"],
        ).images[0]

        inputs.pop("prompt", None)
        inputs.pop("image", None)
        inputs["result"] = result

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """

        return inputs


class StableDiffusionXLImg2ImgTask(AppTask):
    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        # Default to static mode
        self._static_mode = False
        self._construct_model(model)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """

        # build model
        model_instance = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model, paddle_dtype=paddle.float16, variant="fp16"
        )

        self._model = model_instance

    def _preprocess(self, inputs):
        """ """
        image = inputs.get("image", None)
        assert image is not None, "The image is None"
        prompt = inputs.get("prompt", None)
        assert prompt is not None, "The prompt is None"

        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """

        result = self._model(
            prompt=inputs["prompt"],
            image=inputs["image"],
        ).images[0]

        inputs.pop("prompt", None)
        inputs.pop("image", None)
        inputs["result"] = result

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """

        return inputs
