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

import inspect
import io
import json
import os
from collections import defaultdict

from paddlenlp.transformers.auto.configuration import AutoConfig as PPNLPAutoConfig
from paddlenlp.utils.import_utils import import_module

from ..model_utils import PretrainedConfig, PretrainedModel


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
        configuration_path = os.path.join(model_dir, "configuration.py")
        if not os.path.exists(configuration_path):
            continue

        for package in ["paddlenlp", "ppdiffusers"]:
            configuration_module = import_module(f"{package}.transformers.{model_name}.configuration")
            for key in dir(configuration_module):
                value = getattr(configuration_module, key)
                if inspect.isclass(value) and issubclass(value, PretrainedConfig):
                    mappings[model_name].append(value)
    return mappings


class AutoConfig(PPNLPAutoConfig):
    MAPPING_NAMES = get_configurations()

    @classmethod
    def _get_config_class_from_config(cls, pretrained_model_name_or_path: str, config_file_path: str):
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
        for package in ["ppdiffusers", "paddlenlp"]:
            model_class = import_module(f"{package}.transformers.{model_name}")
            if model_class is not None:
                break
        if model_class is None:
            raise ImportError(f"Cannot find the {model_class} from paddlenlp or ppdiffusers.")
        assert inspect.isclass(model_class) and issubclass(
            model_class, PretrainedModel
        ), f"<{model_class}> should be a PretarinedModel class, but <{type(model_class)}>"

        return cls if model_class.config_class is None else model_class.config_class
