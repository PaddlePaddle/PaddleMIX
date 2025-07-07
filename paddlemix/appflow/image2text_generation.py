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
import paddle
from paddlenlp.transformers import AutoTokenizer

from paddlemix import QWenLMHeadModel, QwenVLProcessor, QWenVLTokenizer
from paddlemix.models import MiniGPT4ForConditionalGeneration
from paddlemix.models.blip2.modeling import Blip2ForConditionalGeneration
from paddlemix.processors import MiniGPT4Processor
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
        # build processor
        tokenizer_class = AutoTokenizer.from_pretrained(self._text_model)
        image_processor = BlipImageProcessor.from_pretrained(os.path.join(model, "processor", "eval"))
        text_processor_class = BlipTextProcessor.from_pretrained(os.path.join(model, "processor", "eval"))

        self._processor = Blip2Processor(image_processor, text_processor_class, tokenizer_class)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        # build model
        model_instance = Blip2ForConditionalGeneration.from_pretrained(model, cache_dir=self._model_dir)

        self._model = model_instance
        self._model.eval()

    def _preprocess(self, inputs):
        """ """
        image = inputs.get("image", None)
        assert image is not None, "The image is None"
        prompt = inputs.get("blip2_prompt", None)
        assert image is not None, "The blip2_prompt is None"

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

        inputs["result"] = generated_text
        inputs["prompt"] = self._generate_tags(generated_text)

        return inputs

    def _generate_tags(self, caption):
        lemma = nltk.wordnet.WordNetLemmatizer()

        nltk.download(
            ["punkt", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "wordnet"]
        )
        tags_list = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(caption)) if pos[0] == "N"]
        tags_lemma = [lemma.lemmatize(w) for w in tags_list]
        tags = ",".join(map(str, tags_lemma))
        tags = set(tags.split(","))
        new_tags = ",".join(tags)
        return new_tags


class MiniGPT4Task(AppTask):
    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        self._generate_kwargs = {
            "max_length": 300,
            "num_beams": 1,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "length_penalty": 0.0,
            "temperature": 1.0,
            "decode_strategy": "greedy_search",
            "eos_token_id": [[835], [2277, 29937]],
        }
        # Default to static mode
        self._static_mode = False

        self._construct_processor(model)
        self._construct_model(model)

    def _construct_processor(self, model):
        """
        Construct the tokenizer for the predictor.
        """

        self._processor = MiniGPT4Processor.from_pretrained(model)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        # build model
        model_instance = MiniGPT4ForConditionalGeneration.from_pretrained(self._task_path)

        self._model = model_instance
        self._model.eval()

    def _preprocess(self, inputs):
        """ """
        image = inputs.get("image", None)
        assert image is not None, "The image is None"
        minigpt4_text = inputs.get("minigpt4_text", None)
        assert minigpt4_text is not None, "The minigpt4_text is None"

        prompt = "Give the following image: <Img>ImageContent</Img>. You will be able to see the image once I provide it to you. Please answer my questions.###Human: <Img><ImageHere></Img> <TextHere>###Assistant:"
        minigpt4_input = self._processor([image], minigpt4_text, prompt)

        inputs.pop("minigpt4_text", None)
        inputs["minigpt4_input"] = minigpt4_input

        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """
        generate_kwargs = inputs.get("generate_kwargs", None)
        generate_kwargs = self._generate_kwargs if generate_kwargs is None else generate_kwargs
        outputs = self._model.generate(**inputs["minigpt4_input"], **generate_kwargs)

        inputs.pop("minigpt4_input", None)

        inputs["result"] = outputs

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """
        generated_text = self._processor.batch_decode(inputs["result"][0])[0]
        logger.info("Generate text: {}".format(generated_text))

        inputs["result"] = generated_text

        return inputs


class QwenVLChatTask(AppTask):
    def __init__(self, task, model, **kwargs):
        super().__init__(task=task, model=model, **kwargs)

        self.dtype = kwargs.get("dtype", "bfloat16")
        if not paddle.amp.is_bfloat16_supported() and self.dtype == "bfloat16":
            logger.warning("bfloat16 is not supported on your device,change to float32")
            self.dtype = "float32"
        # Default to static mode
        self._static_mode = False

        self._construct_processor(model)
        self._construct_model(model)

        self.history = None

    def _construct_processor(self, model):
        """
        Construct the tokenizer for the predictor.
        """
        self.tokenizer = QWenVLTokenizer.from_pretrained(model, dtype=self.dtype)
        self._processor = QwenVLProcessor(tokenizer=self.tokenizer)

    def _construct_model(self, model):
        """
        Construct the inference model for the predictor.
        """
        # build model
        model_instance = QWenLMHeadModel.from_pretrained(model, dtype=self.dtype)

        self._model = model_instance
        self._model.eval()

    def _preprocess(self, inputs):
        """ """
        image = inputs.pop("image", None)
        prompt = inputs.pop("prompt", None)
        query = []
        if image is None and prompt is None:
            raise ValueError("The image and prompt are None")

        if image is not None and not isinstance(image, list):
            image = [image]
            for img in image:
                query.append({"image": img})
        if prompt is not None and not isinstance(prompt, list):
            prompt = [prompt]
            for text in prompt:
                query.append({"text": text})

        if len(prompt) == 0 and len(image) > 0:
            query.append({"text": "描述所有图片"})

        process_output = self._processor(query=query, return_tensors="pd")
        query = self.tokenizer.from_list_format(query)

        inputs["query"] = query
        inputs["images"] = process_output["images"]
        return inputs

    def _run_model(self, inputs):
        """
        Run the task model from the outputs of the `_preprocess` function.
        """
        query = inputs.pop("query")
        images = inputs.get("images", None)

        response, history = self._model.chat(self.tokenizer, query=query, history=self.history, images=images)

        self.history = history
        inputs["result"] = response

        return inputs

    def _postprocess(self, inputs):
        """
        The model output is tag ids, this function will convert the model output to raw text.
        """

        return inputs
