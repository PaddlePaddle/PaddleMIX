# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.transformers.auto.processing import AutoProcessor as PPNLPAutoProcessor
from paddlenlp.transformers.processing_utils import ProcessorMixin
from paddlenlp.utils.import_utils import import_module

from ...utils import logging

logger = logging.get_logger(__name__)

__all__ = [
    "AutoProcessor",
]

PROCESSOR_MAPPING_NAMES = OrderedDict(
    [
        ("CLIPProcessor", "clip"),
    ]
)


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
        processing_path = os.path.join(model_dir, "processing.py")
        if not os.path.exists(processing_path):
            continue

        processing_module = import_module(f"ppdiffusers.transformers.{model_name}.processing")
        for key in dir(processing_module):
            value = getattr(processing_module, key)
            if inspect.isclass(value) and issubclass(value, ProcessorMixin):
                mappings[model_name].append(value)

    return mappings


class AutoProcessor(PPNLPAutoProcessor):
    MAPPING_NAMES = get_configurations()
    _processor_mapping = MAPPING_NAMES
    _name_mapping = PROCESSOR_MAPPING_NAMES

    @classmethod
    def _get_processor_class_from_config(cls, pretrained_model_name_or_path, config_file_path):
        with io.open(config_file_path, encoding="utf-8") as f:
            init_kwargs = json.load(f)
        # class name corresponds to this configuration
        init_class = init_kwargs.pop("init_class", None)
        if init_class is None:
            init_class = init_kwargs.pop("processor_class", None)
            if init_class is None:
                init_class = init_kwargs.pop("image_processor_type", None)
                # replace old name to new name
                if init_class is not None and init_class.endswith("ImageProcessor"):
                    init_class = init_class.replace("ImageProcessor", "Processor")
            if init_class is None:
                init_class = init_kwargs.pop("feature_extractor_type", None)
                # replace old name to new name
                if init_class is not None and init_class.endswith("FeatureExtractor"):
                    init_class = init_class.replace("FeatureExtractor", "Processor")

        if init_class:
            try:
                class_name = cls._name_mapping[init_class]
                import_class = import_module(f"ppdiffusers.transformers.{class_name}.processing")
                processor_class = getattr(import_class, init_class)
                return processor_class
            except Exception:
                init_class = None

        # If no `init_class`, we use pattern recognition to recognize the processor class.
        if init_class is None:
            logger.info("We use pattern recognition to recognize the processor class.")
            for key, pattern in cls._name_mapping.items():
                if pattern in pretrained_model_name_or_path.lower():
                    init_class = key
                    class_name = cls._name_mapping[init_class]
                    import_class = import_module(f"ppdiffusers.transformers.{class_name}.processor")
                    processor_class = getattr(import_class, init_class)
                    break
            return processor_class
