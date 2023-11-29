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


from ppdiffusers import DPMSolverMultistepScheduler, TextToVideoSDPipeline

from .apptask import AppTask


class TextToVideoSDTask(AppTask):
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
        model_instance = TextToVideoSDPipeline.from_pretrained(model)
        model_instance.scheduler = DPMSolverMultistepScheduler.from_config(model_instance.scheduler.config)
        self._model = model_instance

    def _preprocess(self, inputs):
        """ """
        prompt = inputs.get("prompt", None)
        assert prompt is not None, "The prompt is None"
        num_inference_steps = inputs.get("num_inference_steps", 25)
        inputs["num_inference_steps"] = num_inference_steps

        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """

        result = self._model(
            prompt=inputs["prompt"],
            num_inference_steps=inputs["num_inference_steps"],
        ).frames

        inputs.pop("prompt", None)

        inputs["result"] = result

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """

        return inputs
