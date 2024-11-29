# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import importlib
import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import PIL.Image
from huggingface_hub import create_repo
from packaging import version
from tqdm.auto import tqdm

from ..configuration_utils import ConfigMixin
from ..utils import (
    DIFFUSERS_CACHE,
    FROM_AISTUDIO,
    FROM_DIFFUSERS,
    FROM_HF_HUB,
    HF_HUB_OFFLINE,
    LOW_CPU_MEM_USAGE_DEFAULT,
    PPDIFFUSERS_CACHE,
    BaseOutput,
    deprecate,
    get_class_from_dynamic_module,
    is_paddle_available,
    is_paddle_version,
    is_paddlenlp_available,
    is_peft_available,
    logging,
    numpy_to_pil,
)

if is_paddle_available():
    import paddle
    import paddle.nn as nn

if is_paddlenlp_available():
    from paddlenlp.transformers import PretrainedModel

    # from paddlenlp.utils.env import (
    #     PADDLE_WEIGHTS_NAME as PPNLP_PADDLE_WEIGHTS_NAME,
    #     PADDLE_WEIGHTS_INDEX_NAME as PPNLP_PADDLE_WEIGHTS_INDEX_NAME,
    #     PYTORCH_WEIGHTS_NAME as PPNLP_PYTORCH_WEIGHTS_NAME,
    #     PYTORCH_WEIGHTS_INDEX_NAME as PPNLP_PYTORCH_WEIGHTS_INDEX_NAME,
    #     SAFE_WEIGHTS_NAME as PPNLP_SAFE_WEIGHTS_NAME,
    #     SAFE_WEIGHTS_INDEX_NAME as PPNLP_SAFE_WEIGHTS_INDEX_NAME,
    # )

from ..models.paddleinfer_runtime import PaddleInferRuntimeModel
from .fastdeploy_utils import FastDeployRuntimeModel

TORCH_INDEX_FILE = "diffusion_pytorch_model.bin"
PADDLE_INDEX_FILE = "model_state.pdparams"

CUSTOM_PIPELINE_FILE_NAME = "pipeline.py"
DUMMY_MODULES_FOLDER = "ppdiffusers.utils"
CONNECTED_PIPES_KEYS = ["prior"]


logger = logging.get_logger(__name__)


LOADABLE_CLASSES = {
    "ppdiffusers": {
        "ModelMixin": ["save_pretrained", "from_pretrained"],
        "SchedulerMixin": ["save_pretrained", "from_pretrained"],
        "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
        "StableDiffusionSafetyChecker": ["save_pretrained", "from_pretrained"],
        "FastDeployRuntimeModel": ["save_pretrained", "from_pretrained"],
        "PaddleInferRuntimeModel": ["save_pretrained", "from_pretrained"],
    },
    "ppdiffusers.transformers": {
        "PretrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PretrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        "ProcessorMixin": ["save_pretrained", "from_pretrained"],
        "ImageProcessingMixin": ["save_pretrained", "from_pretrained"],
    },
    "paddlenlp.transformers": {
        "PretrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PretrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        "ProcessorMixin": ["save_pretrained", "from_pretrained"],
        "ImageProcessingMixin": ["save_pretrained", "from_pretrained"],
        "RobertaTokenizer": ["save_pretrained", "from_pretrained"],
    },
}

ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])


@dataclass
class ImagePipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


@dataclass
class AudioPipelineOutput(BaseOutput):
    """
    Output class for audio pipelines.

    Args:
        audios (`np.ndarray`)
            List of denoised audio samples of a NumPy array of shape `(batch_size, num_channels, sample_rate)`.
    """

    audios: np.ndarray


def is_safetensors_compatible(filenames, variant=None, passed_components=None) -> bool:
    """
    Checking for safetensors compatibility:
    - By default, all models are saved with the default pytorch serialization, so we use the list of default pytorch
      files to know which safetensors files are needed.
    - The model is safetensors compatible only if there is a matching safetensors file for every default pytorch file.

    Converting default pytorch serialized filenames to safetensors serialized filenames:
    - For models from the diffusers library, just replace the ".bin" extension with ".safetensors"
    - For models from the transformers library, the filename changes from "pytorch_model" to "model", and the ".bin"
      extension is replaced with ".safetensors"
    """
    pt_filenames = []

    sf_filenames = set()

    passed_components = passed_components or []

    for filename in filenames:
        _, extension = os.path.splitext(filename)

        if len(filename.split("/")) == 2 and filename.split("/")[0] in passed_components:
            continue

        if extension == ".bin":
            pt_filenames.append(os.path.normpath(filename))
        elif extension == ".safetensors":
            sf_filenames.add(os.path.normpath(filename))

    for filename in pt_filenames:
        #  filename = 'foo/bar/baz.bam' -> path = 'foo/bar', filename = 'baz', extention = '.bam'
        path, filename = os.path.split(filename)
        filename, extension = os.path.splitext(filename)

        if filename.startswith("pytorch_model"):
            filename = filename.replace("pytorch_model", "model")
        else:
            filename = filename

        expected_sf_filename = os.path.normpath(os.path.join(path, filename))
        expected_sf_filename = f"{expected_sf_filename}.safetensors"
        if expected_sf_filename not in sf_filenames:
            logger.warning(f"{expected_sf_filename} not found")
            return False

    return True


def _unwrap_model(model):
    # do nothing
    if is_peft_available():
        from ppdiffusers.peft import PeftModel

        if isinstance(model, PeftModel):
            model = model.base_model.model

    return model


def maybe_raise_or_warn(
    library_name, library, class_name, importable_classes, passed_class_obj, name, is_pipeline_module
):
    """Simple helper method to raise or warn in case incorrect module has been passed"""
    if not is_pipeline_module:
        library = importlib.import_module(library_name)
        class_obj = getattr(library, class_name)
        class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}

        expected_class_obj = None
        for class_name, class_candidate in class_candidates.items():
            if class_candidate is not None and issubclass(class_obj, class_candidate):
                expected_class_obj = class_candidate

        # Dynamo wraps the original model in a private class.
        # I didn't find a public API to get the original class.
        sub_model = passed_class_obj[name]
        unwrapped_sub_model = _unwrap_model(sub_model)
        model_cls = unwrapped_sub_model.__class__

        if not issubclass(model_cls, expected_class_obj):
            raise ValueError(
                f"{passed_class_obj[name]} is of type: {model_cls}, but should be" f" {expected_class_obj}"
            )
    else:
        logger.warning(
            f"You have passed a non-standard module {passed_class_obj[name]}. We cannot verify whether it"
            " has the correct type"
        )


def get_class_obj_and_candidates(
    library_name, class_name, importable_classes, pipelines, is_pipeline_module, component_name=None, cache_dir=None
):
    """Simple helper method to retrieve class object of module as well as potential parent class objects"""
    component_folder = os.path.join(cache_dir, component_name)

    if is_pipeline_module:
        pipeline_module = getattr(pipelines, library_name)

        class_obj = getattr(pipeline_module, class_name)
        class_candidates = {c: class_obj for c in importable_classes.keys()}
    elif os.path.isfile(os.path.join(component_folder, library_name + ".py")):
        # load custom component
        class_obj = get_class_from_dynamic_module(
            component_folder, module_file=library_name + ".py", class_name=class_name
        )
        class_candidates = {c: class_obj for c in importable_classes.keys()}
    else:
        # else we just import it from the library.
        library = importlib.import_module(library_name)

        class_obj = getattr(library, class_name)
        class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}

    # we will use PPNLP PretrainedModel
    if "PretrainedModel" in class_candidates:
        class_candidates["PretrainedModel"] = PretrainedModel
    return class_obj, class_candidates


def _get_pipeline_class(
    class_obj,
    config,
    custom_pipeline=None,
    repo_id=None,
    hub_revision=None,
    class_name=None,
    cache_dir=None,
    revision=None,
):
    if custom_pipeline is not None:
        if custom_pipeline.endswith(".py"):
            path = Path(custom_pipeline)
            # decompose into folder & file
            file_name = path.name
            custom_pipeline = path.parent.absolute()
        elif repo_id is not None:
            file_name = f"{custom_pipeline}.py"
            custom_pipeline = repo_id
        else:
            file_name = CUSTOM_PIPELINE_FILE_NAME

        if repo_id is not None and hub_revision is not None:
            # if we load the pipeline code from the Hub
            # make sure to overwrite the `revison`
            revision = hub_revision

        return get_class_from_dynamic_module(
            custom_pipeline,
            module_file=file_name,
            class_name=class_name,
            repo_id=repo_id,
            cache_dir=cache_dir,
            revision=revision,
        )

    if class_obj != DiffusionPipeline:
        return class_obj

    diffusers_module = importlib.import_module(class_obj.__module__.split(".")[0])
    class_name = config["_class_name"]
    class_name = class_name[4:] if class_name.startswith("Flax") else class_name

    pipeline_cls = getattr(diffusers_module, class_name)

    return pipeline_cls


def load_sub_model(
    library_name: str,
    class_name: str,
    importable_classes: List[Any],
    pipelines: Any,
    is_pipeline_module: bool,
    pipeline_class: Any,
    paddle_dtype: paddle.dtype,
    runtime_options: Any,
    infer_configs: Any,
    use_optim_cache: bool,
    model_variants: Dict[str, str],
    name: str,
    from_diffusers: bool,
    low_cpu_mem_usage: bool,
    cached_folder: Union[str, os.PathLike],
    revision: str = None,
    is_onnx_model: bool = False,
    from_hf_hub: bool = False,
    from_aistudio: bool = False,
    cache_dir: Union[str, os.PathLike] = None,
    variant: str = None,
    use_safetensors: bool = False,
):
    """Helper method to load the module `name` from `library_name` and `class_name`"""
    # retrieve class candidates
    class_obj, class_candidates = get_class_obj_and_candidates(
        library_name,
        class_name,
        importable_classes,
        pipelines,
        is_pipeline_module,
        component_name=name,
        cache_dir=cached_folder,
    )

    load_method_name = None
    # retrive load method name
    for class_name, class_candidate in class_candidates.items():
        if class_candidate is not None and issubclass(class_obj, class_candidate):
            load_method_name = importable_classes[class_name][1]

    # if load method name is None, then we have a dummy module -> raise Error
    if load_method_name is None:
        none_module = class_obj.__module__
        is_dummy_path = none_module.startswith(DUMMY_MODULES_FOLDER)
        if is_dummy_path and "dummy" in none_module:
            # call class_obj for nice error message of missing requirements
            class_obj()

        raise ValueError(
            f"The component {class_obj} of {pipeline_class} cannot be loaded as it does not seem to have"
            f" any of the loading methods defined in {ALL_IMPORTABLE_CLASSES}."
        )
    load_method = getattr(class_obj, load_method_name)

    # add kwargs to loading method
    ppdiffusers_module = importlib.import_module(__name__.split(".")[0])
    loading_kwargs = {}

    # FastDeploy Model
    if issubclass(class_obj, FastDeployRuntimeModel):
        loading_kwargs["runtime_options"] = (
            runtime_options.get(name, None) if isinstance(runtime_options, dict) else runtime_options
        )
        # HACK, this is only for fd onnx model
        if "melgan" in name:
            is_onnx_model = True
        loading_kwargs["is_onnx_model"] = is_onnx_model

    # PaddleInferRuntimeModel
    if issubclass(class_obj, PaddleInferRuntimeModel):
        loading_kwargs["infer_configs"] = (
            infer_configs.get(name, None) if isinstance(infer_configs, dict) else infer_configs
        )
        loading_kwargs["use_optim_cache"] = use_optim_cache

    is_ppdiffusers_model = issubclass(class_obj, ppdiffusers_module.ModelMixin)
    is_paddlenlp_model = issubclass(class_obj, PretrainedModel)

    # PaddleNLP or PPDiffusers Model
    if is_ppdiffusers_model or is_paddlenlp_model:
        loading_kwargs["variant"] = model_variants.pop(name, variant)
        loading_kwargs["use_safetensors"] = use_safetensors
        if is_paddlenlp_model:
            loading_kwargs["convert_from_torch"] = from_diffusers
            loading_kwargs["dtype"] = (
                str(paddle_dtype).replace("paddle.", "") if paddle_dtype is not None else paddle_dtype
            )
            if low_cpu_mem_usage:
                logger.info(
                    f"Auto set low_cpu_mem_usage to `False` for {name}, there may have some bug in paddlenlp's `low_cpu_mem_usage`."
                )
                low_cpu_mem_usage = False
            loading_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage
        else:
            loading_kwargs["from_diffusers"] = from_diffusers
            loading_kwargs["paddle_dtype"] = paddle_dtype
            loading_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

    # check if the module is in a subdirectory
    if os.path.isdir(os.path.join(cached_folder, name)):
        loaded_sub_model = load_method(os.path.join(cached_folder, name), **loading_kwargs)
    else:
        loading_kwargs["from_hf_hub"] = from_hf_hub
        loading_kwargs["from_aistudio"] = from_aistudio
        loading_kwargs["cache_dir"] = cache_dir
        loading_kwargs["subfolder"] = name
        # else load from the root directory
        loaded_sub_model = load_method(cached_folder, **loading_kwargs)

    return loaded_sub_model


class DiffusionPipeline(ConfigMixin):
    r"""
    Base class for all pipelines.

    [`DiffusionPipeline`] stores all components (models, schedulers, and processors) for diffusion pipelines and
    provides methods for loading, downloading and saving models. It also includes methods to:

        - move all PyTorch modules to the device of your choice
        - enable/disable the progress bar for the denoising iteration

    Class attributes:

        - **config_name** (`str`) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.
        - **_optional_components** (`List[str]`) -- List of all optional components that don't have to be passed to the
          pipeline to function (should be overridden by subclasses).
    """

    config_name = "model_index.json"
    model_cpu_offload_seq = None
    _optional_components = []
    _exclude_from_cpu_offload = []
    _load_connected_pipes = False
    _is_fastdeploy = False

    def register_modules(self, **kwargs):
        # import it here to avoid circular import
        ppdiffusers_module = importlib.import_module(__name__.split(".")[0])
        pipelines = getattr(ppdiffusers_module, "pipelines")

        for name, module in kwargs.items():
            # retrieve library
            if module is None or isinstance(module, (tuple, list)) and module[0] is None:
                register_dict = {name: (None, None)}
            else:
                # register the config from the original module, not the dynamo compiled one
                not_compiled_module = _unwrap_model(module)

                # TODO (junnyu) support paddlenlp.transformers check this ?
                if "paddlenlp" in not_compiled_module.__module__.split("."):
                    library = "ppdiffusers.transformers"
                elif "ppdiffusers.transformers" in not_compiled_module.__module__:
                    library = "ppdiffusers.transformers"
                else:
                    library = not_compiled_module.__module__.split(".")[0]

                # check if the module is a pipeline module
                module_path_items = not_compiled_module.__module__.split(".")
                pipeline_dir = module_path_items[-2] if len(module_path_items) > 2 else None

                path = not_compiled_module.__module__.split(".")
                is_pipeline_module = pipeline_dir in path and hasattr(pipelines, pipeline_dir)

                # if library is not in LOADABLE_CLASSES, then it is a custom module.
                # Or if it's a pipeline module, then the module is inside the pipeline
                # folder so we set the library to module name.
                if is_pipeline_module:
                    library = pipeline_dir
                elif library not in LOADABLE_CLASSES:
                    library = not_compiled_module.__module__

                # retrieve class_name
                class_name = not_compiled_module.__class__.__name__

                register_dict = {name: (library, class_name)}

            # save model index config
            self.register_to_config(**register_dict)

            # set models
            setattr(self, name, module)

    def __setattr__(self, name: str, value: Any):
        if name in self.__dict__ and hasattr(self.config, name):
            # We need to overwrite the config if name exists in config
            if isinstance(getattr(self.config, name), (tuple, list)):
                if value is not None and self.config[name][0] is not None:
                    class_library_tuple = (value.__module__.split(".")[0], value.__class__.__name__)
                else:
                    class_library_tuple = (None, None)

                self.register_to_config(**{name: class_library_tuple})
            else:
                self.register_to_config(**{name: value})

        super().__setattr__(name, value)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        push_to_hub: bool = False,
        save_to_aistudio: bool = False,
        max_shard_size: Union[int, str] = "10GB",
        to_diffusers: Optional[bool] = None,
        **kwargs,
    ):
        """
        Save all saveable variables of the pipeline to a directory. A pipeline variable can be saved and loaded if its
        class implements both a save and loading method. The pipeline is easily reloaded using the
        [`~DiffusionPipeline.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save a pipeline to. Will be created if it doesn't exist.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
            variant (`str`, *optional*):
                If specified, weights are saved in the format `pytorch_model.<variant>.bin`.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        model_index_dict = dict(self.config)
        model_index_dict.pop("_class_name", None)
        model_index_dict.pop("_ppdiffusers_version", None)
        model_index_dict.pop("_module", None)
        model_index_dict.pop("_name_or_path", None)

        # create repo
        commit_message = kwargs.pop("commit_message", None)
        private = kwargs.pop("private", False)
        create_pr = kwargs.pop("create_pr", False)
        token = kwargs.pop("token", None)
        token_kwargs = {}
        if token is not None:
            token_kwargs["token"] = token
        repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
        license = kwargs.pop("license", "creativeml-openrail-m")
        exist_ok = kwargs.pop("exist_ok", True)

        if push_to_hub:
            repo_id = create_repo(repo_id, exist_ok=True, private=private, **token_kwargs).repo_id

        if save_to_aistudio:
            from aistudio_sdk.hub import create_repo as aistudio_create_repo

            assert "/" in repo_id, "Please specify the repo id in format of `user_id/repo_name`"
            res = aistudio_create_repo(repo_id=repo_id, private=private, license=license, **token_kwargs)
            if "error_code" in res:
                if res["error_code"] == 10003 and exist_ok:
                    logger.info(
                        f"Repo {repo_id} already exists, it will override files with the same name. To avoid this, please set exist_ok=False"
                    )
                else:
                    logger.error(
                        f"Failed to create repo {repo_id}, error_code: {res['error_code']}, error_msg: {res['error_msg']}"
                    )
            else:
                logger.info(f"Successfully created repo {repo_id}")

        expected_modules, optional_kwargs = self._get_signature_keys(self)

        def is_saveable_module(name, value):
            if name not in expected_modules:
                return False
            if name in self._optional_components and value[0] is None:
                return False
            return True

        model_index_dict = {k: v for k, v in model_index_dict.items() if is_saveable_module(k, v)}
        for pipeline_component_name in model_index_dict.keys():
            sub_model = getattr(self, pipeline_component_name)
            model_cls = sub_model.__class__

            save_method_name = None
            # search for the model's base class in LOADABLE_CLASSES
            for library_name, library_classes in LOADABLE_CLASSES.items():
                if library_name in sys.modules:
                    library = importlib.import_module(library_name)
                else:
                    logger.info(
                        f"{library_name} is not installed. Cannot save {pipeline_component_name} as {library_classes} from {library_name}"
                    )

                for base_class, save_load_methods in library_classes.items():
                    class_candidate = getattr(library, base_class, None)
                    if class_candidate is not None and issubclass(model_cls, class_candidate):
                        # if we found a suitable base class in LOADABLE_CLASSES then grab its save method
                        save_method_name = save_load_methods[0]
                        break
                if save_method_name is not None:
                    break

            if save_method_name is None:
                logger.warn(f"self.{pipeline_component_name}={sub_model} of type {type(sub_model)} cannot be saved.")
                # make sure that unsaveable components are not tried to be loaded afterward
                self.register_to_config(**{pipeline_component_name: (None, None)})
                continue

            save_method = getattr(sub_model, save_method_name)

            # Call the save method with the argument safe_serialization only if it's supported
            save_method_signature = inspect.signature(save_method)
            save_method_accept_safe = "safe_serialization" in save_method_signature.parameters
            save_method_accept_variant = "variant" in save_method_signature.parameters
            save_method_accept_max_shard_size = "max_shard_size" in save_method_signature.parameters
            save_method_accept_to_diffusers = "to_diffusers" in save_method_signature.parameters

            save_kwargs = {}
            if save_method_accept_safe:
                save_kwargs["safe_serialization"] = safe_serialization
            if save_method_accept_variant:
                save_kwargs["variant"] = variant
            if save_method_accept_max_shard_size:
                save_kwargs["max_shard_size"] = max_shard_size
            if save_method_accept_to_diffusers:
                save_kwargs["to_diffusers"] = to_diffusers

            save_method(os.path.join(save_directory, pipeline_component_name), **save_kwargs)

        # finally save the config
        self.save_config(save_directory, to_diffusers=to_diffusers)

        if save_to_aistudio:
            self._upload_folder_aistudio(
                save_directory,
                repo_id,
                commit_message=commit_message,
                **token_kwargs,
            )
        if push_to_hub:
            self._upload_folder(
                save_directory,
                repo_id,
                commit_message=commit_message,
                create_pr=create_pr,
                **token_kwargs,
            )

    def to(self, *args, **kwargs):
        r"""
        Performs Pipeline dtype and/or device conversion. A paddle.dtype and paddle.device are inferred from the
        arguments of `self.to(*args, **kwargs).`

        <Tip>

            If the pipeline already has the correct paddle.dtype and paddle.device, then it is returned as is. Otherwise,
            the returned pipeline is a copy of self with the desired paddle.dtype and paddle.device.

        </Tip>


        Here are the ways to call `to`:

        - `to(dtype, silence_dtype_warnings=False) → DiffusionPipeline` to return a pipeline with the specified
          [`dtype`].
        - `to(device, silence_dtype_warnings=False) → DiffusionPipeline` to return a pipeline with the specified
          [`device`].
        - `to(device=None, dtype=None, silence_dtype_warnings=False) → DiffusionPipeline` to return a pipeline with the
          specified [`device`] and [`dtype`].

        Arguments:
            dtype (`paddle.dtype`, *optional*):
                Returns a pipeline with the specified
            device (`str`, *optional*):
                Returns a pipeline with the specified
            silence_dtype_warnings (`str`, *optional*, defaults to `False`):
                Whether to omit warnings if the target `dtype` is not compatible with the target `device`.

        Returns:
            [`DiffusionPipeline`]: The pipeline converted to specified `dtype` and/or `dtype`.
        """

        paddle_dtype = kwargs.pop("paddle_dtype", None)
        if paddle_dtype is not None:
            deprecate("paddle_dtype", "0.45.0", "")
        paddle_device = kwargs.pop("paddle_device", None)
        if paddle_device is not None:
            deprecate("paddle_device", "0.45.0", "")

        dtype_kwarg = kwargs.pop("dtype", None)
        device_kwarg = kwargs.pop("device", None)

        if paddle_dtype is not None and dtype_kwarg is not None:
            raise ValueError(
                "You have passed both `paddle_dtype` and `dtype` as a keyword argument. Please make sure to only pass `dtype`."
            )

        dtype = paddle_dtype or dtype_kwarg

        if paddle_device is not None and device_kwarg is not None:
            raise ValueError(
                "You have passed both `torch_device` and `device` as a keyword argument. Please make sure to only pass `device`."
            )

        device = paddle_device or device_kwarg

        dtype_arg = None
        device_arg = None
        if len(args) == 1:
            if isinstance(args[0], paddle.dtype):
                dtype_arg = args[0]
            else:
                device_arg = args[0] if args[0] is not None else None
        elif len(args) == 2:
            if isinstance(args[0], paddle.dtype):
                raise ValueError(
                    "When passing two arguments, make sure the first corresponds to `device` and the second to `dtype`."
                )
            device_arg = args[0] if args[0] is not None else None
            dtype_arg = args[1]
        elif len(args) > 2:
            raise ValueError("Please make sure to pass at most two arguments (`device` and `dtype`) `.to(...)`")

        if dtype is not None and dtype_arg is not None:
            raise ValueError(
                "You have passed `dtype` both as an argument and as a keyword argument. Please only pass one of the two."
            )

        dtype = dtype or dtype_arg

        if device is not None and device_arg is not None:
            raise ValueError(
                "You have passed `device` both as an argument and as a keyword argument. Please only pass one of the two."
            )

        device = device or device_arg

        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, nn.Layer)]

        for module in modules:
            module.to(device=device, dtype=dtype)

        return self

    @property
    def device(self) -> str:
        r"""
        Returns:
            `paddle.device`: The paddle device on which the pipeline is located.
        """
        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, nn.Layer)]

        for module in modules:
            return module.place
        return "cpu"

    @property
    def dtype(self) -> paddle.dtype:
        r"""
        Returns:
            `paddle.dtype`: The paddle dtype on which the pipeline is located.
        """
        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, nn.Layer)]

        for module in modules:
            return module.dtype

        return paddle.float32

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        r"""
        Instantiate a Paddle diffusion pipeline from pretrained pipeline weights.

        The pipeline is set in evaluation mode (`model.eval()`) by default.

        If you get the error message below, you need to finetune the weights for your downstream task:

        ```
        Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
        - conv_in.weight: found shape [320, 4, 3, 3] in the checkpoint and [320, 9, 3, 3] in the model instantiated
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
        ```

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *repo id* (for example `CompVis/ldm-text2im-large-256`) of a pretrained pipeline
                      hosted on the Hub.
                    - A path to a *directory* (for example `./my_pipeline_directory/`) containing pipeline weights
                      saved using
                    [`~DiffusionPipeline.save_pretrained`].
            paddle_dtype (`str` or `paddle.dtype`, *optional*):
                Override the default `paddle.dtype` and load the model with another dtype. If "auto" is passed, the
                dtype is automatically derived from the model's weights.
            custom_pipeline (`str`, *optional*):

                <Tip warning={true}>

                🧪 This is an experimental feature and may change in the future.

                </Tip>

                Can be either:

                    - A string, the *repo id* (for example `hf-internal-testing/diffusers-dummy-pipeline`) of a custom
                      pipeline hosted on the Hub. The repository must contain a file called pipeline.py that defines
                      the custom pipeline.
                    - A string, the *file name* of a community pipeline hosted on GitHub under
                      [Community](https://github.com/huggingface/diffusers/tree/main/examples/community). Valid file
                      names must match the file name and not the pipeline script (`clip_guided_stable_diffusion`
                      instead of `clip_guided_stable_diffusion.py`). Community pipelines are always loaded from the
                      current main branch of GitHub.
                    - A path to a directory (`./my_pipeline_directory/`) containing a custom pipeline. The directory
                      must contain a file called `pipeline.py` that defines the custom pipeline.

                For more information on how to load and create custom pipelines, please have a look at [Loading and
                Adding Custom
                Pipelines](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_overview)
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            custom_revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id similar to
                `revision` when loading a custom pipeline from the Hub. It can be a 🤗 Diffusers version when loading a
                custom pipeline from GitHub, otherwise it defaults to `"main"` when loading from the Hub.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.
            device_map (`str` or `Dict[str, Union[int, str, paddle.device]]`, *optional*):
                A map that specifies where each submodule should go. It doesn’t need to be defined for each
                parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                same device.

                Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                each GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                The path to offload weights if device_map contains the value `"disk"`.
            offload_state_dict (`bool`, *optional*):
                If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                when there is some disk offload.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if paddle version >= 2.5.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for Paddle >= 2.5.0. If you are using an older version of Paddle, setting this
                argument to `True` will raise an error.
            use_safetensors (`bool`, *optional*, defaults to `None`):
                If set to `None`, the safetensors weights are downloaded if they're available **and** if the
                safetensors library is installed. If set to `True`, the model is forcibly loaded from safetensors
                weights. If set to `False`, safetensors weights are not loaded.
            use_onnx (`bool`, *optional*, defaults to `None`):
                If set to `True`, ONNX weights will always be downloaded if present. If set to `False`, ONNX weights
                will never be downloaded. By default `use_onnx` defaults to the `_is_onnx` class attribute which is
                `False` for non-ONNX pipelines and `True` for ONNX pipelines. ONNX weights include both files ending
                with `.onnx` and `.pb`.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables (the pipeline components of the specific pipeline
                class). The overwritten components are passed directly to the pipelines `__init__` method. See example
                below for more information.
            variant (`str`, *optional*):
                Load weights from a specified variant filename such as `"fp16"` or `"ema"`. This is ignored when
                loading `from_flax`.

        <Tip>

        To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with
        `huggingface-cli login`.

        </Tip>

        Examples:

        ```py
        >>> from ppdiffusers import DiffusionPipeline

        >>> # Download pipeline from huggingface.co and cache.
        >>> pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")

        >>> # Download pipeline that requires an authorization token
        >>> # For more information on access tokens, please refer to this section
        >>> # of the documentation](https://huggingface.co/docs/hub/security-tokens)
        >>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

        >>> # Use a different scheduler
        >>> from ppdiffusers import LMSDiscreteScheduler

        >>> scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
        >>> pipeline.scheduler = scheduler
        ```
        """
        from_hf_hub = kwargs.pop("from_hf_hub", FROM_HF_HUB)
        from_aistudio = kwargs.pop("from_aistudio", FROM_AISTUDIO)
        cache_dir = kwargs.pop("cache_dir", None)
        if cache_dir is None:
            if from_aistudio:
                cache_dir = None  # TODO, check aistudio cache
            elif from_hf_hub:
                cache_dir = DIFFUSERS_CACHE
            else:
                cache_dir = PPDIFFUSERS_CACHE
        from_diffusers = kwargs.pop("from_diffusers", FROM_DIFFUSERS)
        resume_download = kwargs.pop("resume_download", False)  # noqa: F841
        force_download = kwargs.pop("force_download", False)  # noqa: F841
        proxies = kwargs.pop("proxies", None)  # noqa F841
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)  # noqa: F841
        use_auth_token = kwargs.pop("use_auth_token", None)  # noqa: F841
        revision = kwargs.pop("revision", None)
        paddle_dtype = kwargs.pop("paddle_dtype", None)
        custom_pipeline = kwargs.pop("custom_pipeline", None)  # noqa: F841
        custom_revision = kwargs.pop("custom_revision", None)
        provider = kwargs.pop("provider", None)  # noqa: F841
        sess_options = kwargs.pop("sess_options", None)  # noqa: F841
        device_map = kwargs.pop("device_map", None)  # noqa: F841
        max_memory = kwargs.pop("max_memory", None)  # noqa: F841
        offload_folder = kwargs.pop("offload_folder", None)  # noqa: F841
        offload_state_dict = kwargs.pop("offload_state_dict", False)  # noqa: F841
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", LOW_CPU_MEM_USAGE_DEFAULT)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)  # noqa: F841
        use_onnx = kwargs.pop("use_onnx", None)  # noqa: F841
        use_fastdeploy = kwargs.pop("use_fastdeploy", None)  # noqa: F841
        runtime_options = kwargs.pop("runtime_options", None)
        infer_configs = kwargs.pop("infer_configs", None)
        use_optim_cache = kwargs.pop("use_optim_cache", False)
        load_connected_pipeline = kwargs.pop("load_connected_pipeline", False)  # noqa: F841
        model_variants = kwargs.pop("model_variants", {})

        # 1. Download the checkpoints and configs
        # use snapshot download here to get it working from from_pretrained
        if os.path.isdir(pretrained_model_name_or_path):
            # if pretrained_model_name_or_path.count("/") > 1:
            #     raise ValueError(
            #         f'The provided pretrained_model_name_or_path "{pretrained_model_name_or_path}"'
            #         " is neither a valid local path nor a valid repo id. Please check the parameter."
            #     )
            cached_folder = pretrained_model_name_or_path
            config_dict = cls.load_config(cached_folder)
        else:
            config_dict = cls.load_config(
                pretrained_model_name_or_path,
                from_hf_hub=from_hf_hub,
                from_aistudio=from_aistudio,
                cache_dir=cache_dir,
            )
            cached_folder = pretrained_model_name_or_path

        # pop out "_ignore_files" as it is only needed for download
        ignore_filenames = config_dict.pop("_ignore_files", [])  # noqa: F841

        custom_class_name = None
        if os.path.isfile(os.path.join(cached_folder, f"{custom_pipeline}.py")):
            custom_pipeline = os.path.join(cached_folder, f"{custom_pipeline}.py")
        elif isinstance(config_dict["_class_name"], (list, tuple)) and os.path.isfile(
            os.path.join(cached_folder, f"{config_dict['_class_name'][0]}.py")
        ):
            custom_pipeline = os.path.join(cached_folder, f"{config_dict['_class_name'][0]}.py")
            custom_class_name = config_dict["_class_name"][1]

        pipeline_class = _get_pipeline_class(
            cls,
            config_dict,
            custom_pipeline=custom_pipeline,
            class_name=custom_class_name,
            cache_dir=cache_dir,
            revision=custom_revision,
        )

        # DEPRECATED: To be removed in 1.0.0
        _ppdiffusers_version = (
            config_dict["_diffusers_paddle_version"]
            if "_diffusers_paddle_version" in config_dict
            else config_dict["_ppdiffusers_version"]
        )
        # DEPRECATED: To be removed in 1.0.0
        if pipeline_class.__name__ == "StableDiffusionInpaintPipeline" and version.parse(
            _ppdiffusers_version
        ) <= version.parse("0.5.1"):
            from ppdiffusers import (
                StableDiffusionInpaintPipeline,
                StableDiffusionInpaintPipelineLegacy,
            )

            pipeline_class = StableDiffusionInpaintPipelineLegacy

            deprecation_message = (
                "You are using a legacy checkpoint for inpainting with Stable Diffusion, therefore we are loading the"
                f" {StableDiffusionInpaintPipelineLegacy} class instead of {StableDiffusionInpaintPipeline}. For"
                " better inpainting results, we strongly suggest using Stable Diffusion's official inpainting"
                " checkpoint: https://huggingface.co/runwayml/stable-diffusion-inpainting instead or adapting your"
                f" checkpoint {pretrained_model_name_or_path} to the format of"
                " https://huggingface.co/runwayml/stable-diffusion-inpainting. Note that we do not actively maintain"
                f" the {StableDiffusionInpaintPipelineLegacy} class and will likely remove it in version 1.0.0."
            )
            deprecate("StableDiffusionInpaintPipelineLegacy", "1.0.0", deprecation_message, standard_warn=False)

        # 4. Define expected modules given pipeline signature
        # and define non-None initialized modules (=`init_kwargs`)

        # some modules can be passed directly to the init
        # in this case they are already instantiated in `kwargs`
        # extract them here
        expected_modules, optional_kwargs = cls._get_signature_keys(pipeline_class)
        passed_class_obj = {k: kwargs.pop(k) for k in expected_modules if k in kwargs}
        passed_pipe_kwargs = {k: kwargs.pop(k) for k in optional_kwargs if k in kwargs}

        init_dict, unused_kwargs, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)

        # define init kwargs and make sure that optional component modules are filtered out
        init_kwargs = {
            k: init_dict.pop(k)
            for k in optional_kwargs
            if k in init_dict and k not in pipeline_class._optional_components
        }
        init_kwargs = {**init_kwargs, **passed_pipe_kwargs}

        # remove `null` components
        def load_module(name, value):
            if value[0] is None:
                return False
            if name in passed_class_obj and passed_class_obj[name] is None:
                return False
            return True

        init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

        # 5. Throw nice warnings / errors for fast accelerate loading
        if len(unused_kwargs) > 0:
            logger.warning(
                f"Keyword arguments {unused_kwargs} are not expected by {pipeline_class.__name__} and will be ignored."
            )

        if low_cpu_mem_usage and (not is_paddle_version(">=", "2.5.0") and not is_paddle_version("==", "0.0.0")):
            raise NotImplementedError(
                "Low memory initialization requires paddlepaddle-gpu >= 2.5.0. Please either update your PaddlePaddle version or set"
                " `low_cpu_mem_usage=False`."
            )

        # import it here to avoid circular import
        from ppdiffusers import pipelines

        # 6. Load each module in the pipeline
        for name, (library_name, class_name) in logging.tqdm(
            sorted(init_dict.items(), key=lambda x: x[0]), desc="Loading pipeline components..."
        ):
            # 6.1 - now that JAX/Flax is an official framework of the library, we might load from Flax names
            if library_name in ["diffusers_paddle", "diffusers"]:
                library_name = "ppdiffusers"
            if library_name in ["transformers", "paddlenlp.transformers"]:
                library_name = "ppdiffusers.transformers"

            class_name = class_name[4:] if class_name.startswith("Flax") else class_name
            # support HF fast tokenizer
            class_name = (
                class_name[:-4] if class_name.endswith("Fast") and "tokenizer" in class_name.lower() else class_name
            )

            # 6.2 Define all importable classes
            is_pipeline_module = hasattr(pipelines, library_name)
            importable_classes = ALL_IMPORTABLE_CLASSES
            loaded_sub_model = None

            # 6.3 Use passed sub model or load class_name from library_name
            if name in passed_class_obj:
                # if the model is in a pipeline module, then we load it from the pipeline
                # check that passed_class_obj has correct parent class
                maybe_raise_or_warn(
                    library_name, library, class_name, importable_classes, passed_class_obj, name, is_pipeline_module
                )

                loaded_sub_model = passed_class_obj[name]
            else:
                # load sub model
                loaded_sub_model = load_sub_model(
                    library_name=library_name,
                    class_name=class_name,
                    importable_classes=importable_classes,
                    pipelines=pipelines,
                    is_pipeline_module=is_pipeline_module,
                    pipeline_class=pipeline_class,
                    paddle_dtype=paddle_dtype,
                    runtime_options=runtime_options,
                    infer_configs=infer_configs,
                    use_optim_cache=use_optim_cache,
                    model_variants=model_variants,
                    name=name,
                    from_diffusers=from_diffusers,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                    cached_folder=cached_folder,
                    revision=revision,
                    is_onnx_model=False,
                    from_hf_hub=from_hf_hub,
                    from_aistudio=from_aistudio,
                    cache_dir=cache_dir,
                    variant=variant,
                    use_safetensors=use_safetensors,
                )
                logger.info(
                    f"Loaded {name} as {class_name} from `{name}` subfolder of {pretrained_model_name_or_path}."
                )
            init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

        # 7. Potentially add passed objects if expected
        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        passed_modules = list(passed_class_obj.keys())
        optional_modules = pipeline_class._optional_components
        if len(missing_modules) > 0 and missing_modules <= set(passed_modules + optional_modules):
            for module in missing_modules:
                init_kwargs[module] = passed_class_obj.get(module, None)
        elif len(missing_modules) > 0:
            passed_modules = set(list(init_kwargs.keys()) + list(passed_class_obj.keys())) - optional_kwargs
            raise ValueError(
                f"Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed."
            )

        # 8. (TODO, junnyu) make sure all modules are in eval mode and cast dtype
        for name, _module in init_kwargs.items():
            if isinstance(_module, nn.Layer):
                _module.eval()
                if paddle_dtype is not None:
                    if str(paddle_dtype) in ["paddle.float32", "float32"]:
                        _module.to(dtype="float32")
                    else:
                        paddle.amp.decorate(
                            _module,
                            level="O2",
                            dtype=str(paddle_dtype).replace("paddle.", ""),
                        )
            elif isinstance(_module, (tuple, list)):
                for _submodule in _module:
                    if isinstance(_submodule, nn.Layer):
                        _submodule.eval()
                        if paddle_dtype is not None:
                            if str(paddle_dtype) in ["paddle.float32", "float32"]:
                                _module.to(dtype="float32")
                            else:
                                paddle.amp.decorate(
                                    _submodule,
                                    level="O2",
                                    dtype=str(paddle_dtype).replace("paddle.", ""),
                                )
        # 8. Instantiate the pipeline
        model = pipeline_class(**init_kwargs)

        # 9. Save where the model was instantiated from
        model.register_to_config(_name_or_path=pretrained_model_name_or_path)
        return model

    @property
    def name_or_path(self) -> str:
        return getattr(self.config, "_name_or_path", None)

    @classmethod
    def _get_signature_keys(cls, obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - {"self"}

        optional_names = list(optional_parameters)
        for name in optional_names:
            if name in cls._optional_components:
                expected_modules.add(name)
                optional_parameters.remove(name)

        return expected_modules, optional_parameters

    @property
    def components(self) -> Dict[str, Any]:
        r"""
        The `self.components` property can be useful to run different pipelines with the same weights and
        configurations without reallocating additional memory.

        Returns (`dict`):
            A dictionary containing all the modules needed to initialize the pipeline.

        Examples:

        ```py
        >>> from ppdiffusers import (
        ...     StableDiffusionPipeline,
        ...     StableDiffusionImg2ImgPipeline,
        ...     StableDiffusionInpaintPipeline,
        ... )

        >>> text2img = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
        >>> inpaint = StableDiffusionInpaintPipeline(**text2img.components)
        ```
        """
        expected_modules, optional_parameters = self._get_signature_keys(self)
        components = {
            k: getattr(self, k) for k in self.config.keys() if not k.startswith("_") and k not in optional_parameters
        }

        if set(components.keys()) != expected_modules:
            raise ValueError(
                f"{self} has been incorrectly initialized or {self.__class__} is incorrectly implemented. Expected"
                f" {expected_modules} to be defined, but {components.keys()} are defined."
            )

        return components

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a NumPy image or a batch of images to a PIL image.
        """
        return numpy_to_pil(images)

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[str] = None):
        r"""
        Enable memory efficient attention as implemented in xformers.

        When this option is enabled, you should observe lower GPU memory usage and a potential speed up at inference
        time. Speed up at training time is not guaranteed.

        Warning: When Memory Efficient Attention and Sliced attention are both enabled, the Memory Efficient Attention
        is used.

        Parameters:
            attention_op (`Callable`, *optional*):
                Override the default `None` operator for use as `op` argument to the
                [`memory_efficient_attention()`](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.memory_efficient_attention)
                function of xFormers.

        Examples:

        ```py
        >>> import paddle
        >>> from ppdiffusers import DiffusionPipeline

        >>> pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", paddle_dtype=paddle.float16)
        >>> pipe.enable_xformers_memory_efficient_attention("cutlass")
        ```
        """
        self.set_use_memory_efficient_attention_xformers(True, attention_op)

    def disable_xformers_memory_efficient_attention(self):
        r"""
        Disable memory efficient attention as implemented in xformers.
        """
        self.set_use_memory_efficient_attention_xformers(False)

    def set_use_memory_efficient_attention_xformers(self, valid: bool, attention_op: Optional[str] = None) -> None:
        # Recursively walk through all the children.
        # Any children which exposes the set_use_memory_efficient_attention_xformers method
        # gets the message
        def fn_recursive_set_mem_eff(module: nn.Layer):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid, attention_op)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, nn.Layer)]

        for module in modules:
            fn_recursive_set_mem_eff(module)

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation. When this option is enabled, the attention module splits the input tensor
        in slices to compute attention in several steps. For more than one attention head, the computation is performed
        sequentially over each head. This is useful to save some memory in exchange for a small speed decrease.

        <Tip warning={true}>

        ⚠️ Don't enable attention slicing if you're already using `scaled_dot_product_attention` (SDPA) from Paddle
        2.5 or PPXFormers. These attention computations are already very memory efficient so you won't need to enable
        this function. If you enable attention slicing with SDPA or PPXFormers, it can lead to serious slow downs!

        </Tip>

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maximum amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.

        Examples:

        ```py
        >>> import paddle
        >>> from ppdiffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5",
        ...     paddle_dtype=paddle.float16,
        ...     use_safetensors=True,
        ... )

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> pipe.enable_attention_slicing()
        >>> image = pipe(prompt).images[0]
        ```
        """
        self.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously called, attention is
        computed in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    def set_attention_slice(self, slice_size: Optional[int]):
        module_names, _ = self._get_signature_keys(self)
        modules = [getattr(self, n, None) for n in module_names]
        modules = [m for m in modules if isinstance(m, nn.Layer) and hasattr(m, "set_attention_slice")]

        for module in modules:
            module.set_attention_slice(slice_size)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        if hasattr(self, "vae"):
            self.vae.enable_slicing()
        if hasattr(self, "vqvae"):
            self.vqvae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        if hasattr(self, "vae"):
            self.vae.disable_slicing()
        if hasattr(self, "vqvae"):
            self.vqvae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.
        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        if hasattr(self, "vae"):
            self.vae.enable_tiling()
        if hasattr(self, "vqvae"):
            self.vqvae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        if hasattr(self, "vae"):
            self.vae.disable_tiling()
        if hasattr(self, "vqvae"):
            self.vqvae.disable_tiling()

    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet.disable_freeu()

    def maybe_free_model_hooks(self):
        r"""
        Function that offloads all components, removes all model hooks that were added when using
        `enable_model_cpu_offload` and then applies them again. In case the model has not been offloaded this function
        is a no-op. Make sure to add this function to the end of the `__call__` function of your pipeline so that it
        functions correctly when applying enable_model_cpu_offload.
        """
        # do nothing
        return
