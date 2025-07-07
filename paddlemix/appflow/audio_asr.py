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

from paddlespeech.cli.asr import ASRExecutor

from paddlemix.utils.log import logger

from .apptask import AppTask


class AudioASRTask(AppTask):
    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        # Default to static mode
        self._static_mode = False

        self._construct_model()

        self.model = model

    def _construct_model(self):
        """
        Construct the inference model for the predictor.
        """

        # build model
        s2t_executor = ASRExecutor()
        self._model = s2t_executor

    def _preprocess(self, inputs):
        """ """
        audio = inputs.get("audio", None)
        assert audio is not None, "The image is None"
        prompt = inputs.get("prompt")
        assert prompt is not None, "The prompt is None"
        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """
        result = self._model(audio_file=inputs["audio"])

        logger.info("Audio File ASR Result: {}".format(result))

        inputs["prompt"] = inputs["prompt"].format(result)

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """

        return inputs
