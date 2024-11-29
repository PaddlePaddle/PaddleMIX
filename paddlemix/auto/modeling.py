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
import io
import json
import os
from collections import defaultdict
from typing import Optional

from huggingface_hub import hf_hub_download
from paddlenlp.transformers.configuration_utils import is_standard_config
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.utils.downloader import (
    COMMUNITY_MODEL_PREFIX,
    get_path_from_url_with_filelock,
    hf_file_exists,
    url_file_exists,
)
from paddlenlp.utils.env import HF_CACHE_HOME as PPNLP_HF_CACHE_HOME
from paddlenlp.utils.import_utils import import_module
from paddlenlp.utils.log import logger

from paddlemix.utils.env import MODEL_HOME as PPMIX_MODEL_HOME

from .configuration import get_configurations

__all__ = [
    "AutoModelMIX",
]


# This mapping specifies a unique calling class name for the model.
# When 'architecture' is not defined in the config file, the calling class is specified based on this mapping.
# If the class name specified in this map cannot meet the calling requirements,
# please set 'architecture' in the config file
ASSIGN_MAPPING = {
    # Assign model mapping
    "blip2": "Blip2ForConditionalGeneration",
    "clip": "CLIP",
    "coca": "CLIP",
    "eva02": "EVA02VisionTransformer",  # unsupport EVA02ForPretrain
    "evaclip": "EVACLIP",
    "groundingdino": "GroundingDinoModel",
    "imagebind": "ImageBindModel",
    "minigpt4": "MiniGPT4ForConditionalGeneration",
    "qwen_vl": "QWenLMHeadModel",
    "sam": "SamModel",
    "visualglm": "VisualGLMForConditionalGeneration",
    "llava_qwen": "LlavaQwenForCausalLM",
    "internvl2": "InternVLChatModel",
    "janus":"JanusMultiModalityCausalLM",
    "janus_flow":"JanusFlowMultiModality",
}


def resolve_cache_dir(from_hf_hub: bool, from_aistudio: bool, cache_dir: Optional[str] = None) -> str:
    """resolve cache dir for PretrainedModel and PretrainedConfig

    Args:
        from_hf_hub (bool): if load from huggingface hub
        cache_dir (str): cache_dir for models
    """
    if cache_dir is not None:
        return cache_dir
    if from_aistudio:
        return None
    if from_hf_hub:
        return PPNLP_HF_CACHE_HOME

    return PPMIX_MODEL_HOME


def get_model_mapping():

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

        # 2. find the `*model*.py` file as the identifier of PretrainedModel class
        # for llava:
        if model_name == 'llava':
            for file_name in ['language_model.llava_llama','language_model.llava_qwen','multimodal_encoder.clip_model']:
                model_module = import_module(f"paddlemix.models.{model_name}.{file_name}")
                for key in dir(model_module):
                    if key == "PretrainedModel" or key == "MixPretrainedModel":
                        continue
                    value = getattr(model_module, key)
                    if inspect.isclass(value) and issubclass(value, PretrainedModel):
                        mappings[model_name].append((value.__name__, value))

        else:
            for file_name in os.listdir(model_dir):
                if "model" not in file_name:
                    continue

                model_module = import_module(f"paddlemix.models.{model_name}.{file_name[:-3]}")
                for key in dir(model_module):
                    if key == "PretrainedModel" or key == "MixPretrainedModel":
                        continue
                    value = getattr(model_module, key)
                    if inspect.isclass(value) and issubclass(value, PretrainedModel):
                        mappings[model_name].append((value.__name__, value))

    return mappings


class _MIXBaseAutoModelClass:
    # Base class for auto models.
    _pretrained_model_dict = None
    _name_mapping = None
    _task_choice = False
    model_config_file = "config.json"
    legacy_model_config_file = "model_config.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path).`"
        )

    # TODO: Refactor into AutoConfig when available
    @classmethod
    def _get_model_class_from_config(cls, pretrained_model_name_or_path, config_file_path, config=None):
        if config is None:
            with io.open(config_file_path, encoding="utf-8") as f:
                config = json.load(f)

        # Get class name corresponds to this configuration
        if is_standard_config(config):
            architectures = config["architectures"]
            init_class = architectures.pop() if len(architectures) > 0 else None
        else:
            init_class = config.pop("init_class", None)

        if init_class == "ChatGLMModel":
            init_class = "VisualGLMForConditionalGeneration"

        import_class = None
        if init_class:
            for model_flag, names in cls._name_mapping.items():
                for class_name in names:
                    if init_class == class_name[0]:
                        import_class = class_name[1]
                        break
        else:
            logger.info(
                "No model name specified in architectures, use pretrained_model_name_or_path to parse model class"
            )
            # From pretrained_model_name_or_path
            for model_flag, name in ASSIGN_MAPPING.items():
                pretrained_model_name_or_path = pretrained_model_name_or_path.lower().replace("-", "_")
                if model_flag.lower() in pretrained_model_name_or_path:

                    for class_name in cls._name_mapping[model_flag]:
                        if class_name[0] == name:
                            import_class = class_name[1]
                            break
                    break
        if import_class is None:
            raise AttributeError(
                f"Unable to parse 'architectures' or 'init_class' from {config_file_path}. Also unable to infer model class from 'pretrained_model_name_or_path'"
            )

        return import_class

    @classmethod
    def from_config(cls, config, **kwargs):
        model_class = cls._get_model_class_from_config(None, None, config)
        return model_class._from_config(config, **kwargs)

    @classmethod
    def _from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):

        cache_dir = kwargs.get("cache_dir", None)
        from_hf_hub = kwargs.get("from_hf_hub", False)
        from_aistudio = kwargs.get("from_aistudio", False)
        subfolder = kwargs.get("subfolder", "")
        cache_dir = resolve_cache_dir(from_hf_hub, from_aistudio, cache_dir)
        kwargs["cache_dir"] = cache_dir

        if from_hf_hub:
            if hf_file_exists(repo_id=pretrained_model_name_or_path, filename=cls.model_config_file):
                config_file = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename=cls.model_config_file,
                    subfolder=subfolder,
                    cache_dir=cache_dir,
                    library_name="PaddleNLP",
                )
            elif hf_file_exists(repo_id=pretrained_model_name_or_path, filename=cls.legacy_model_config_file):
                logger.info("Standard config do not exist, loading from legacy config")
                config_file = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename=cls.legacy_model_config_file,
                    subfolder=subfolder,
                    cache_dir=cache_dir,
                    library_name="PaddleNLP",
                )
            if os.path.exists(config_file):
                model_class = cls._get_model_class_from_config(pretrained_model_name_or_path, config_file)
                logger.info(f"We are using {model_class} to load '{pretrained_model_name_or_path}'.")
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            else:
                logger.warning(f"{config_file}  is not a valid path to a model config file")
        # From local dir path
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, cls.model_config_file)
            legacy_config_file = os.path.join(pretrained_model_name_or_path, cls.legacy_model_config_file)
            if os.path.exists(config_file):
                model_class = cls._get_model_class_from_config(pretrained_model_name_or_path, config_file)
                logger.info(f"We are using {model_class} to load '{pretrained_model_name_or_path}'.")
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            elif os.path.exists(legacy_config_file):
                logger.info("Standard config do not exist, loading from legacy config")
                model_class = cls._get_model_class_from_config(pretrained_model_name_or_path, legacy_config_file)
                logger.info(f"We are using {model_class} to load '{pretrained_model_name_or_path}'.")
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            else:
                logger.warning(f"{config_file}  is not a valid path to a model config file")
        # Assuming from community-contributed pretrained models
        else:
            standard_community_url = "/".join(
                [COMMUNITY_MODEL_PREFIX, pretrained_model_name_or_path, cls.model_config_file]
            )
            legacy_community_url = "/".join(
                [COMMUNITY_MODEL_PREFIX, pretrained_model_name_or_path, cls.legacy_model_config_file]
            )
            cache_dir = os.path.join(cache_dir, pretrained_model_name_or_path, subfolder)

            try:
                if url_file_exists(standard_community_url):
                    resolved_vocab_file = get_path_from_url_with_filelock(standard_community_url, cache_dir)
                elif url_file_exists(legacy_community_url):
                    logger.info("Standard config do not exist, loading from legacy config")
                    resolved_vocab_file = get_path_from_url_with_filelock(legacy_community_url, cache_dir)
                else:
                    raise RuntimeError("Neither 'config.json' nor 'model_config.json' exists")
            except RuntimeError as err:
                logger.error(err)
                raise RuntimeError(
                    f"Can't load weights for '{pretrained_model_name_or_path}'.\n"
                    f"Please make sure that '{pretrained_model_name_or_path}' is:\n"
                    "- a correct model-identifier of built-in pretrained models,\n"
                    "- or a correct model-identifier of community-contributed pretrained models,\n"
                    "- or the correct path to a directory containing relevant modeling files(model_weights and model_config).\n"
                )

            if os.path.exists(resolved_vocab_file):
                model_class = cls._get_model_class_from_config(pretrained_model_name_or_path, resolved_vocab_file)
                logger.info(f"We are using {model_class} to load '{pretrained_model_name_or_path}'.")
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
            else:
                logger.warning(f"{resolved_vocab_file}  is not a valid path to a model config file")


class AutoModelMIX(_MIXBaseAutoModelClass):
    """
    AutoClass can help you automatically retrieve the relevant model given the provided
    pretrained weights/vocabulary.
    AutoModel is a generic model class that will be instantiated as one of the base model classes
    when created with the from_pretrained() classmethod.
    """

    CONFIGURATION_MODEL_MAPPING = get_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_model_mapping()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Creates an instance of `AutoModel`. Model weights are loaded
        by specifying name of a built-in pretrained model, a pretrained model on HF, a community contributed model,
        or a local file directory path.
        Args:
            pretrained_model_name_or_path (str): Name of pretrained model or dir path
                to load from. The string can be:
                - Name of a built-in pretrained model
                - Name of a community-contributed pretrained model.
                - Local directory path which contains model weights file("model_state.pdparams")
                  and model config file ("model_config.json").
            task (str): Specify a downstream task. Task can be 'Model', 'ForPretraining',
                'ForSequenceClassification', 'ForTokenClassification', 'ForQuestionAnswering',
                'ForMultipleChoice', 'ForMaskedLM', 'ForCausalLM', 'Encoder', 'Decoder',
                'Generator', 'Discriminator', 'ForConditionalGeneration'.
                We only support specify downstream tasks in AutoModel. Defaults to `None`.
            *args (tuple): Position arguments for model `__init__`. If provided,
                use these as position argument values for model initialization.
            **kwargs (dict): Keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for model
                initialization. If the keyword is in `__init__` argument names of
                base model, update argument values of the base model; else update
                argument values of derived model.
        Returns:
            PretrainedModel: An instance of `AutoModelMIX`.
        """
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
