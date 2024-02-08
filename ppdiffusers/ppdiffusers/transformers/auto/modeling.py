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

from paddlenlp.transformers.auto.modeling import AutoModel as PPNLPAutoModel
from paddlenlp.utils.import_utils import import_module

from ..model_utils import PretrainedModel

__all__ = [
    "AutoModel",
]

from paddlenlp.transformers.auto.modeling import MAPPING_NAMES

NEW_MAPPING_NAMES = OrderedDict(
    [
        ("CLIPText", "clip"),
        ("CLIPVision", "clip"),
        ("CLIP", "clip"),
        ("T5Encoder", "t5"),
        ("T5", "t5"),
        ("Bert", "bert"),
        ("Roberta", "roberta"),
        ("XLMRoberta", "xlm_roberta"),
        ("GPT2", "gpt2"),
    ]
)
MAPPING_NAMES.update(NEW_MAPPING_NAMES)


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
        modeling_path = os.path.join(model_dir, "modeling.py")
        if not os.path.exists(modeling_path):
            continue

        for package in ["paddlenlp", "ppdiffusers"]:
            modeling_module = import_module(f"{package}.transformers.{model_name}.modeling")
            for key in dir(modeling_module):
                value = getattr(modeling_module, key)
                if inspect.isclass(value) and issubclass(value, PretrainedModel):
                    mappings[model_name].append(value)

    return mappings


class AutoModel(PPNLPAutoModel):
    """
    AutoClass can help you automatically retrieve the relevant model given the provided
    pretrained weights/vocabulary.
    AutoModel is a generic model class that will be instantiated as one of the base model classes
    when created with the from_pretrained() classmethod.
    """

    CONFIGURATION_MODEL_MAPPING = get_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING

    @classmethod
    def _get_model_class_from_config(cls, pretrained_model_name_or_path, config_file_path, config=None):
        if config is None:
            with io.open(config_file_path, encoding="utf-8") as f:
                config = json.load(f)

        # Get class name corresponds to this configuration
        architectures = config["architectures"]
        init_class = architectures.pop() if len(architectures) > 0 else None
        assert init_class is not None, f"Unable to parse 'architectures' from {config_file_path}"
        model_name = None
        class_name = None
        for model_flag, name in MAPPING_NAMES.items():
            if model_flag in init_class:
                model_name = model_flag + "Model"
                class_name = name
                break

        if model_name is None or class_name is None:
            raise AttributeError(
                f"Unable to parse 'architectures' or 'init_class' from {config_file_path}. Also unable to infer model class from '{pretrained_model_name_or_path}'"
            )
        for package in ["ppdiffusers", "paddlenlp"]:
            import_class = import_module(f"{package}.transformers.{class_name}.modeling")
            if import_class is not None:
                break
        if import_class is None:
            raise ImportError(f"Cannot find the {class_name} from paddlenlp or ppdiffusers.")
        model_class = getattr(import_class, model_name)
        return model_class
