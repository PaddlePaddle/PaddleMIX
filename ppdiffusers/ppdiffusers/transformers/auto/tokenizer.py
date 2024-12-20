# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
import inspect
import io
import json
import os
from collections import OrderedDict, defaultdict

from paddlenlp.transformers import PretrainedTokenizer
from paddlenlp.transformers.auto.tokenizer import AutoTokenizer as PPNLPAutoTokenizer
from paddlenlp.utils.import_utils import import_module

from ...utils import logging

logger = logging.get_logger(__name__)

__all__ = [
    "AutoTokenizer",
]

from paddlenlp.transformers.auto.tokenizer import TOKENIZER_MAPPING_NAMES

NEW_TOKENIZER_MAPPING_NAMES = OrderedDict(
    [
        ("CLIPTokenizer", "clip"),
        ("T5Tokenizer", "t5"),
        ("BertTokenizer", "bert"),
        ("XLMRobertaTokenizer", "xlm_roberta"),
        ("GPT2Tokenizer", "gpt2"),
        ("RobertaTokenizer", "roberta"),
    ]
)
TOKENIZER_MAPPING_NAMES.update(NEW_TOKENIZER_MAPPING_NAMES)


def get_configurations():
    """load the configurations of PretrainedConfig mapping: {<model-name>: [<class-name>, <class-name>, ...], }

    Returns:
        dict[str, str]: the mapping of model-name to model-classes
    """
    # 1. search the subdir<model-name> to find model-names
    ppdiffusers_transformers_dir = os.path.dirname(os.path.dirname(__file__))
    exclude_models = ["auto"]

    mappings = defaultdict(list)
    for model_name in os.listdir(ppdiffusers_transformers_dir):
        if model_name in exclude_models:
            continue

        model_dir = os.path.join(ppdiffusers_transformers_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        # 2. find the `configuration.py` file as the identifier of PretrainedConfig class
        tokenizer_path = os.path.join(model_dir, "tokenizer.py")
        if not os.path.exists(tokenizer_path):
            continue

        for package in ["paddlenlp", "ppdiffusers"]:
            tokenizezr_module = import_module(f"{package}.transformers.{model_name}.tokenizer")
            for key in dir(tokenizezr_module):
                value = getattr(tokenizezr_module, key)
                if inspect.isclass(value) and issubclass(value, PretrainedTokenizer):
                    mappings[model_name].append(value)

    return mappings


class AutoTokenizer(PPNLPAutoTokenizer):
    MAPPING_NAMES = get_configurations()
    _tokenizer_mapping = MAPPING_NAMES
    _name_mapping = TOKENIZER_MAPPING_NAMES

    @classmethod
    def _get_tokenizer_class_from_config(cls, pretrained_model_name_or_path, config_file_path, use_fast=None):
        with io.open(config_file_path, encoding="utf-8") as f:
            init_kwargs = json.load(f)
        # class name corresponds to this configuration
        init_class = init_kwargs.pop("init_class", None)
        if init_class is None:
            init_class = init_kwargs.pop("tokenizer_class", None)

        if init_class:
            class_name = cls._name_mapping[init_class]
            for package in ["ppdiffusers", "paddlenlp"]:
                import_class = import_module(f"{package}.transformers.{class_name}.tokenizer")
                if import_class is not None:
                    break
            if import_class is None:
                raise ImportError(f"Cannot find the {class_name} from paddlenlp or ppdiffusers.")
            tokenizer_class = getattr(import_class, init_class)
            if use_fast:
                fast_tokenizer_class = cls._get_fast_tokenizer_class(init_class, class_name)
                tokenizer_class = fast_tokenizer_class if fast_tokenizer_class else tokenizer_class
            return tokenizer_class
        # If no `init_class`, we use pattern recognition to recognize the tokenizer class.
        else:
            logger.info("We use pattern recognition to recognize the Tokenizer class.")
            tokenizer_class = None
            for key, pattern in cls._name_mapping.items():
                if pattern in pretrained_model_name_or_path.lower():
                    init_class = key
                    class_name = cls._name_mapping[init_class]
                    for package in ["ppdiffusers", "paddlenlp"]:
                        import_class = import_module(f"{package}.transformers.{class_name}.tokenizer")
                        if import_class is not None:
                            break
                    if import_class is None:
                        raise ImportError(f"Cannot find the {class_name} from paddlenlp or ppdiffusers.")
                    tokenizer_class = getattr(import_class, init_class)
                    if use_fast:
                        fast_tokenizer_class = cls._get_fast_tokenizer_class(init_class, class_name)
                        tokenizer_class = fast_tokenizer_class if fast_tokenizer_class else tokenizer_class
                    break
            if tokenizer_class is None:
                raise ImportError("Cannot find the tokenizer from paddlenlp or ppdiffusers.")
            return tokenizer_class
