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
from __future__ import annotations

import inspect
import io
import json
import os
from collections import defaultdict
from typing import Dict, List, Type

from paddlenlp.transformers import AutoConfig
#fix paddlenlp 3.0b3 auto
from paddlemix.models.llava.language_model.llava_llama import LlavaConfig
try:
    AutoConfig.register("llava", LlavaConfig)
    print('LlavaConfig register success!!!!!')
except:
    pass

from paddlenlp.transformers.configuration_utils import PretrainedConfig
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.utils.import_utils import import_module

__all__ = [
    "AutoConfigMIX",
]


def get_configurations() -> Dict[str, List[Type[PretrainedConfig]]]:
    """load the configurations of PretrainedConfig mapping: {<model-name>: [<class-name>, <class-name>, ...], }
    Returns:
        dict[str, str]: the mapping of model-name to model-classes
    """
    # 1. search the subdir<model-name> to find model-names
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    exclude_dir = ["__pycache__", "common"]
    mappings = defaultdict(list)
    for model_name in os.listdir(models_dir):
        if model_name in exclude_dir:
            continue

        model_dir = os.path.join(models_dir, model_name)
        if not os.path.isdir(model_dir):
            continue

        # 2. find the `configuration.py` file as the identifier of PretrainedConfig class
        # for llava:
        if model_name == 'llava':
            for file_name in ['language_model.llava_llama','language_model.llava_qwen','multimodal_encoder.clip_model']:
                configuration_module = import_module(f"paddlemix.models.{model_name}.{file_name}")

                for key in dir(configuration_module):
                    value = getattr(configuration_module, key)
                    if inspect.isclass(value) and issubclass(value, PretrainedConfig):
                        mappings[model_name].append(value)
        else:            
            for file_name in os.listdir(model_dir):
                if "model" not in file_name and "configuration" not in file_name:
                    continue

                configuration_module = import_module(f"paddlemix.models.{model_name}.{file_name[:-3]}")

                for key in dir(configuration_module):
                    value = getattr(configuration_module, key)
                    if inspect.isclass(value) and issubclass(value, PretrainedConfig):
                        mappings[model_name].append(value)

    return mappings


class AutoConfigMIX(AutoConfig):
    """
    AutoConfigMIX is a generic config class that will be instantiated as one of the
    base PretrainedConfig classes when created with the AutoConfigMIX.from_pretrained() classmethod.
    """

    MAPPING_NAMES: Dict[str, List[Type[PretrainedConfig]]] = get_configurations()

    # cache the builtin pretrained-model-name to Model Class
    name2class = None
    config_file = "config.json"

    legacy_config_file = "model_config.json"

    @classmethod
    def _get_config_class_from_config(
        cls, pretrained_model_name_or_path: str, config_file_path: str
    ) -> PretrainedConfig:
        with io.open(config_file_path, encoding="utf-8") as f:
            config = json.load(f)

        # add support for legacy config
        if "init_class" in config:
            architectures = [config.pop("init_class")]
        else:
            architectures = config.pop("architectures", None)
            if architectures is None:
                return cls

        model_name = architectures[0]
        model_class = import_module(f"paddlemix.models.{model_name}")

        assert inspect.isclass(model_class) and issubclass(
            model_class, PretrainedModel
        ), f"<{model_class}> should be a PretrainedModel class, but <{type(model_class)}>"

        return cls if model_class.config_class is None else model_class.config_class
