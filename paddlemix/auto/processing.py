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

import inspect
import os
from collections import defaultdict

from paddlenlp.utils.import_utils import import_module

from paddlemix.processors.base_processing import ProcessorMixin
from paddlemix.processors.processing_utils import (
    BaseAudioProcessor,
    BaseImageProcessor,
    BaseTextProcessor,
)

from .tokenizer import AutoTokenizerMIX

__all__ = [
    "AutoProcessorMIX",
]


def get_processor_mapping():

    # 1. search the subdir<model-name> to find model-names
    processors_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processors")
    exclude_file = ["base_processing.py", "image_processing_utils.py", "processing_utils.py"]
    mappings = defaultdict(dict)
    for file_name in os.listdir(processors_dir):
        if file_name in exclude_file:
            continue

        if "processing" not in file_name:
            continue

        # 2. find the `*processing.py` file as the identifier of ProcessorMixin class,
        model_name = file_name[:-14]  # remove `.processing.py` 

        model_module = import_module(f"paddlemix.processors.{file_name[:-3]}")
        for key in dir(model_module):
            if key == "ProcessorMixin" or key == "BaseImageProcessor" or key == "BaseTextProcessor":
                continue
            value = getattr(model_module, key)
            if inspect.isclass(value):
                if issubclass(value, ProcessorMixin):
                    mappings[model_name]["processor"] = value
                elif issubclass(value, BaseImageProcessor):
                    mappings[model_name]["image_processor"] = value
                elif issubclass(value, BaseTextProcessor):
                    mappings[model_name]["text_processor"] = value
                elif issubclass(value, BaseAudioProcessor):
                    mappings[model_name]["audio_processor"] = value

    return mappings


class AutoProcessorMIX:
    """
    AutoProcessor is a generic processor class that will be instantiated as one of the
    base processor classes when created with the AutoProcessor.from_pretrained() classmethod.
    """

    MAPPING_NAMES = get_processor_mapping()
    _processor_mapping = MAPPING_NAMES

    processor_config_file = "preprocessor_config.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path).`"
        )

    @classmethod
    def _get_processor_class(cls, pretrained_model_name_or_path, text_model_name_or_path=None, **kwargs):

        name_or_path = None
        processor = None
        tokenizer = None

        train = kwargs.pop("train", None)
        eval = kwargs.pop("eval", None)

        if train is not None and eval is not None:
            raise ValueError("You cannot specify both `train` and `eval`.")
        elif train is not None:
            name_or_path = (
                os.path.join(pretrained_model_name_or_path, "processor", "train")
                if train
                else pretrained_model_name_or_path
            )
        elif eval is not None:
            name_or_path = (
                os.path.join(pretrained_model_name_or_path, "processor", "eval")
                if eval
                else pretrained_model_name_or_path
            )
        else:
            name_or_path = pretrained_model_name_or_path

        if text_model_name_or_path is None:
            text_model_name_or_path = pretrained_model_name_or_path

        for names, processor_class in cls._processor_mapping.items():
            if 'intern' in names.lower():
                import pdb
                pdb.set_trace()
            if names.lower() in pretrained_model_name_or_path.lower().replace("-", "_").replace("vicuna", "llava"):
                attributes = processor_class["processor"].attributes
                attributes_dict = {}

                for attr in attributes:
                    if attr == "tokenizer":
                        continue
                    if attr in processor_class:
                        attributes_dict[attr] = processor_class[attr].from_pretrained(name_or_path, **kwargs)
                    else:
                        class_name = attr + "_class"
                        class_name = getattr(processor_class["processor"], class_name)
                        attributes_dict[attr] = import_module(f"paddlenlp.transformers.{class_name}").from_pretrained(
                            name_or_path, **kwargs
                        )

                if "tokenizer" in attributes:
                    if "max_length" in kwargs:
                        kwargs["model_max_length"] = kwargs.get("max_length", None)
                    tokenizer = AutoTokenizerMIX.from_pretrained(text_model_name_or_path, **kwargs)
                    attributes_dict["tokenizer"] = tokenizer

                for key, value in attributes_dict.items():
                    kwargs[key] = value

                processor = processor_class["processor"](**kwargs)

                break

        return processor, tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, text_model_name_or_path=None, **kwargs):

        return cls._get_processor_class(pretrained_model_name_or_path, text_model_name_or_path, **kwargs)
