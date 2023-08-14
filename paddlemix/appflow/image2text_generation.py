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

import os

import nltk
from paddlenlp.transformers import AutoTokenizer

from paddlemix.models.blip2.modeling import Blip2ForConditionalGeneration
from paddlemix.processors.blip_processing import (
    Blip2Processor,
    BlipImageProcessor,
    BlipTextProcessor,
)
from paddlemix.utils.log import logger

from .apptask import AppTask


class Blip2CaptionTask(AppTask):
    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        self._text_model = kwargs.get("blip2_text_model", "facebook/opt-2.7b")
        # Default to static mode
        self._static_mode = False

        self._construct_processor(model)
        self._construct_model(model)

    def _construct_processor(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        # bulid processor
        tokenizer_class = AutoTokenizer.from_pretrained(self._text_model, use_fast=False)
        image_processor = BlipImageProcessor.from_pretrained(os.path.join(model, "processor", "eval"))
        text_processor_class = BlipTextProcessor.from_pretrained(os.path.join(model, "processor", "eval"))

        self._processor = Blip2Processor(image_processor, text_processor_class, tokenizer_class)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        # bulid model
        model_instance = Blip2ForConditionalGeneration.from_pretrained(model, cache_dir=self._model_dir)

        self._model = model_instance
        self._model.eval()

    def _preprocess(self, inputs):
        """ """
        image = inputs.get("image", None)
        assert image is not None, "The image is None"

        prompt = "describe the image"

        blip2_input = self._processor(
            images=image,
            text=prompt,
            return_tensors="pd",
            return_attention_mask=True,
            mode="test",
        )

        inputs["blip2_input"] = blip2_input

        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """

        generated_ids, _ = self._model.generate(**inputs["blip2_input"])

        inputs.pop("blip2_input", None)

        inputs["result"] = generated_ids

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """
        generated_text = self._processor.batch_decode(inputs["result"], skip_special_tokens=True)[0].strip()
        logger.info("Generate text: {}".format(generated_text))

        inputs.pop("result", None)

        inputs["prompt"] = self._generate_tags(generated_text)

        return inputs

    def _generate_tags(self, caption):
        lemma = nltk.wordnet.WordNetLemmatizer()

        nltk.download(["punkt", "averaged_perceptron_tagger", "wordnet"])
        tags_list = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(caption)) if pos[0] == "N"]
        tags_lemma = [lemma.lemmatize(w) for w in tags_list]
        tags = ", ".join(map(str, tags_lemma))

        return tags
