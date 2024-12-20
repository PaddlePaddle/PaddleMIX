# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import io
import json
import os

import yaml
from paddlenlp.transformers import AutoTokenizer
#fix paddlenlp 3.0b3 auto
from paddlemix.models.llava.language_model.tokenizer import LLavaTokenizer
from paddlemix.models.llava.language_model.llava_llama import LlavaConfig
try:
    AutoTokenizer.register(LlavaConfig, LLavaTokenizer)
    print('LLavaTokenizer register success!!!!')
except:
    pass

from paddlenlp.utils.import_utils import import_module
from paddlenlp.utils.log import logger

__all__ = ["AutoTokenizerMIX"]


class AutoTokenizerMIX(AutoTokenizer):
    """
    AutoClass can help you automatically retrieve the relevant model given the provided
    pretrained weights/vocabulary.
    AutoTokenizer is a generic tokenizer class that will be instantiated as one of the
    base tokenizer classes when created with the AutoTokenizer.from_pretrained() classmethod.
    """

    @classmethod
    def _update_name_mapping(cls):

        tokenizer_mapping = os.path.join(os.path.dirname(__file__), "tokenizer_mapping.yaml")

        with open(tokenizer_mapping) as f:
            cfg = yaml.load(f, Loader=yaml.Loader)

        for key, value in cfg.items():
            cls._name_mapping[key] = value

    @classmethod
    def _get_tokenizer_class_from_config(cls, pretrained_model_name_or_path, config_file_path, use_fast=None):
        cls._update_name_mapping()
        with io.open(config_file_path, encoding="utf-8") as f:
            init_kwargs = json.load(f)
        # class name corresponds to this configuration
        init_class = init_kwargs.pop("init_class", None)
        if init_class is None:
            init_class = init_kwargs.pop("tokenizer_class", None)
        if init_class:
            class_name = cls._name_mapping[init_class]
            import_class = import_module(f"paddlenlp.transformers.{class_name}.tokenizer")
            if import_class is None:
                if class_name == "processors":
                    import_class = import_module(f"paddlemix.{class_name}.tokenizer")
                else:
                    # import_class = import_module(f"paddlemix.models.{class_name}.tokenizer")
                    import_class = import_module(f"paddlemix.models.{class_name}")

            tokenizer_class = getattr(import_class, init_class)
            if use_fast:
                fast_tokenizer_class = cls._get_fast_tokenizer_class(init_class, class_name)
                tokenizer_class = fast_tokenizer_class if fast_tokenizer_class else tokenizer_class
            return tokenizer_class
        # If no `init_class`, we use pattern recognition to recognize the tokenizer class.
        else:
            # TODO: Potential issue https://github.com/PaddlePaddle/PaddleNLP/pull/3786#discussion_r1024689810
            logger.info("We use pattern recognition to recognize the Tokenizer class.")
            for key, pattern in cls._name_mapping.items():
                if pattern in pretrained_model_name_or_path.lower():
                    init_class = key
                    class_name = cls._name_mapping[init_class]
                    import_class = import_module(f"paddlenlp.transformers.{class_name}.tokenizer")
                    if import_class is None:
                        import_class = import_module(f"paddlemix.models.{class_name}.tokenizer")
                    tokenizer_class = getattr(import_class, init_class)

                    if use_fast:
                        fast_tokenizer_class = cls._get_fast_tokenizer_class(init_class, class_name)
                        tokenizer_class = fast_tokenizer_class if fast_tokenizer_class else tokenizer_class
                    break
            return tokenizer_class
