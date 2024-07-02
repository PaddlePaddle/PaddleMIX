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
import paddle.nn as nn
from paddlespeech.cli.tts import TTSExecutor

from paddlemix.utils.log import logger

from .apptask import AppTask


def get_parameter_dtype(parameter: nn.Layer) -> paddle.dtype:
    try:
        return next(parameter.named_parameters())[1].dtype
    except StopIteration:
        try:
            return next(parameter.named_buffers())[1].dtype
        except StopIteration:
            return parameter._dtype


@property
def dtype_getter(self):
    if hasattr(self, "__dtype"):
        return self.__dtype
    return get_parameter_dtype(self)


nn.Layer.dtype = dtype_getter


@nn.Layer.dtype.setter
def dtype_setter(self, value):
    self.__dtype = value


nn.Layer.dtype = dtype_setter


class AudioTTSTask(AppTask):
    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        # Default to static mode
        self._static_mode = False

        self._construct_model()

    def _construct_model(self):
        """
        Construct the inference model for the predictor.
        """

        # build model
        tts_executor = TTSExecutor()

        self._model = tts_executor

    def _preprocess(self, inputs):
        """ """

        prompt = inputs.get("prompt", None)
        assert prompt is not None, "The prompt is None"

        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """

        paddle.device.cuda.empty_cache()
        _prompt = inputs.get("prompt", None)
        _output = inputs.get("output", "tmp.wav")
        _am = inputs.get("am", "fastspeech2_csmsc")
        _am_config = inputs.get("am_config", None)
        _am_ckpt = inputs.get("am_ckpt", None)
        _am_stat = inputs.get("am_stat", None)
        _spk_id = inputs.get("spk_id", 0)
        _phones_dict = inputs.get("phones_dict", None)
        _tones_dict = inputs.get("tones_dict", None)
        _speaker_dict = inputs.get("speaker_dict", None)
        _voc = inputs.get("voc", "pwgan_csmsc")
        _voc_config = inputs.get("voc_config", None)
        _voc_ckpt = inputs.get("voc_ckpt", None)
        _voc_stat = inputs.get("voc_stat", None)
        _lang = inputs.get("lang", "zh")

        wav_file = self._model(
            text=_prompt,
            output=_output,
            am=_am,
            am_config=_am_config,
            am_ckpt=_am_ckpt,
            am_stat=_am_stat,
            spk_id=_spk_id,
            phones_dict=_phones_dict,
            tones_dict=_tones_dict,
            speaker_dict=_speaker_dict,
            voc=_voc,
            voc_config=_voc_config,
            voc_ckpt=_voc_ckpt,
            voc_stat=_voc_stat,
            lang=_lang,
        )

        inputs.pop("audio", None)
        inputs["audio"] = wav_file
        logger.info("Wave file has been generated: {}".format(wav_file))

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """

        return inputs
