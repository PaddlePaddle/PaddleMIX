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

from paddlenlp import Taskflow

from .apptask import AppTask


class ChatGlmTask(AppTask):
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
        model_instance = Taskflow("text2text_generation", model=model)

        self._model = model_instance

    def _preprocess(self, inputs):
        """ """
        # e.g.
        # prompt = (
        #     "Given caption,extract the main object to be replaced and marked it as 'main_object', "
        #     + "Extract the remaining part as 'other prompt', "
        #     + "Return main_object, other prompt in English"
        #     + "Given caption: {}.".format(prompt)
        # )

        prompt = inputs.get("prompt")
        assert prompt is not None, "The prompt is None"

        inputs["prompt"] = prompt

        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """

        result = self._model(inputs["prompt"])["result"][0]

        inputs.pop("prompt", None)
        inputs["result"] = result

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """

        prompt, inpaint_prompt = (
            inputs["result"].split("\n")[0].split(":")[-1].strip(),
            inputs["result"].split("\n")[-1].split(":")[-1].strip(),
        )

        inputs.pop("result", None)

        inputs["prompt"] = prompt
        inputs["inpaint_prompt"] = inpaint_prompt

        return inputs
