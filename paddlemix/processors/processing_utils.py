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

import copy
import json
import os
import tempfile
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
from huggingface_hub import (
    create_repo,
    get_hf_file_metadata,
    hf_hub_download,
    hf_hub_url,
    repo_type_and_id_from_hf_id,
    upload_folder,
)
from huggingface_hub.utils import EntryNotFoundError
from paddlenlp import __version__
from paddlenlp.transformers.tokenizer_utils_base import BatchEncoding

from paddlemix.utils.downloader import (
    COMMUNITY_MODEL_PREFIX,
    get_path_from_url_with_filelock,
    resolve_cache_dir,
)
from paddlemix.utils.log import logger

try:
    from paddlenlp.transformers.aistudio_utils import aistudio_download
except:
    logger.warning("aistudio_download not import, if you want to use , require paddlenlp develop")
    aistudio_download = None
    pass
import aistudio_sdk

PROCESSOR_CONFIG_MAPPING = {
    "image": "image_preprocessor_config.json",
    "text": "text_preprocessor_config.json",
    "audio": "audio_preprocessor_config.json",
    "image_train": "image_preprocessor_config.json",
    "text_train": "text_preprocessor_config.json",
    "image_eval": "image_preprocessor_config.json",
    "text_eval": "text_preprocessor_config.json",
}


class BaseProcessingMixin(object):
    """
    This is an base processor mixin used to provide saving/loading functionality for sequential and feature
    extractors.
    """

    _auto_class = None
    input_type = None

    def __init__(self, **kwargs):
        """Set elements of `kwargs` as attributes."""
        # Pop "processor_class" as it should be saved as private attribute
        self._processor_class = kwargs.pop("processor_class", None)
        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def _set_processor_class(self, processor_class: str):
        """Sets processor class as an attribute."""
        self._processor_class = processor_class

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        r"""
        Instantiate a type of [`~processing_utils.BaseProcessingMixin`] from an processor.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained processor hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a processor file saved using the
                  [`~processing_utils.BaseProcessingMixin.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - a path or url to a saved processor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model processor should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the processor files and override the cached versions if
                they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            use_auth_token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.


                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".

                </Tip>

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final processor object. If `True`, then this
                functions returns a `Tuple(processor, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not processor attributes: i.e., the part of
                `kwargs` which has not been used to update `processor` and is otherwise ignored.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are processor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* processor attributes is
                controlled by the `return_unused_kwargs` keyword parameter.

        Returns:
            A processor of type [`~processing_utils.BaseProcessingMixin`].
        ```"""
        processor_dict, kwargs = cls.get_processor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(processor_dict, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Save an processor object to the directory `save_directory`, so that it can be re-loaded using the
        [`~processing_utils.BaseProcessingMixin.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the processor JSON file will be saved (will be created if it does not exist).
            kwargs:
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_processor_file = os.path.join(save_directory, PROCESSOR_CONFIG_MAPPING[self.input_type])

        self.to_json_file(output_processor_file)
        logger.info(f"processor saved in {output_processor_file}")

        return [output_processor_file]

    def save_to_hf_hub(
        self,
        repo_id: str,
        private: Optional[bool] = None,
        subfolder: Optional[str] = None,
        commit_message: Optional[str] = None,
        revision: Optional[str] = None,
        create_pr: bool = False,
    ):
        """
        Uploads all elements of this processor to a new HuggingFace Hub repository.
        Args:
            repo_id (str): Repository name for your processor in the Hub.
            private (bool, optional): Whether the processor is set to private
            subfolder (str, optional): Push to a subfolder of the repo instead of the root
            commit_message (str, optional) — The summary / title / first line of the generated commit. Defaults to: f"Upload {path_in_repo} with huggingface_hub"
            revision (str, optional) — The git revision to commit from. Defaults to the head of the "main" branch.
            create_pr (boolean, optional) — Whether or not to create a Pull Request with that commit. Defaults to False.
                If revision is not set, PR is opened against the "main" branch. If revision is set and is a branch, PR is opened against this branch.
                If revision is set and is not a branch name (example: a commit oid), an RevisionNotFoundError is returned by the server.

        Returns: The url of the commit of your model in the given repository.
        """
        repo_url = create_repo(repo_id, private=private, exist_ok=True)

        # Infer complete repo_id from repo_url
        # Can be different from the input `repo_id` if repo_owner was implicit
        _, repo_owner, repo_name = repo_type_and_id_from_hf_id(repo_url)

        repo_id = f"{repo_owner}/{repo_name}"

        # Check if README file already exist in repo
        try:
            get_hf_file_metadata(hf_hub_url(repo_id=repo_id, filename="README.md", revision=revision))
            has_readme = True
        except EntryNotFoundError:
            has_readme = False

        with tempfile.TemporaryDirectory() as root_dir:
            if subfolder is not None:
                save_dir = os.path.join(root_dir, subfolder)
            else:
                save_dir = root_dir
            # save model
            self.save_pretrained(save_dir)
            # Add readme if does not exist
            logger.info("README.md not found, adding the default README.md")
            if not has_readme:
                with open(os.path.join(root_dir, "README.md"), "w") as f:
                    f.write(f"---\nlibrary_name: paddlenlp\n---\n# {repo_id}")

            # Upload model and return
            logger.info(f"Pushing to the {repo_id}. This might take a while")
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=root_dir,
                commit_message=commit_message,
                revision=revision,
                create_pr=create_pr,
            )

    def save_to_aistudio(
        self,
        repo_id,
        private=True,
        license="Apache License 2.0",
        exist_ok=True,
        safe_serialization=True,
        subfolder=None,
        merge_tensor_parallel=False,
        **kwargs
    ):
        """
        Uploads all elements of this model to a new AiStudio Hub repository.
        Args:
            repo_id (str): Repository name for your model/tokenizer in the Hub.
            token (str): Your token for the Hub.
            private (bool, optional): Whether the model/tokenizer is set to private. Defaults to True.
            license (str): The license of your model/tokenizer. Defaults to: "Apache License 2.0".
            exist_ok (bool, optional): Whether to override existing repository. Defaults to: True.
            safe_serialization (bool, optional): Whether to save the model in safe serialization way. Defaults to: True.
            subfolder (str, optional): Push to a subfolder of the repo instead of the root
            merge_tensor_parallel (bool): Whether to merge the tensor parallel weights. Defaults to False.
        """

        res = aistudio_sdk.hub.create_repo(repo_id=repo_id, private=private, license=license, **kwargs)
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

        with tempfile.TemporaryDirectory() as root_dir:
            if subfolder is not None:
                save_dir = os.path.join(root_dir, subfolder)
            else:
                save_dir = root_dir

            # save model
            self.save_pretrained(save_dir)

            # Upload model and return
            logger.info(f"Pushing to the {repo_id}. This might take a while")
            for filename in os.listdir(save_dir):
                path_in_repo = os.path.join(subfolder, filename) if subfolder is not None else filename
                res = aistudio_sdk.hub.upload(
                    repo_id=repo_id,
                    path_or_fileobj=os.path.join(save_dir, filename),
                    path_in_repo=path_in_repo,
                    **kwargs,
                )
                if "error_code" in res:
                    logger.error(
                        f"Failed to upload {filename}, error_code: {res['error_code']}, error_msg: {res['error_msg']}"
                    )
                else:
                    logger.info(f"{filename}: {res['message']}")

    @classmethod
    def get_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        processor of type [`~processor_utils.BaseProcessingMixin`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            from_hf_hub (bool, optional): whether to load from Huggingface Hub
            subfolder (str, optional) An optional value corresponding to a folder inside the repo.


        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the processor object.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        from_hf_hub = kwargs.pop("from_hf_hub", False)
        from_aistudio = kwargs.get("from_aistudio", False)
        subfolder = kwargs.pop("subfolder", None)
        cache_dir = resolve_cache_dir(pretrained_model_name_or_path, from_hf_hub, cache_dir)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            resolved_processor_file = os.path.join(
                pretrained_model_name_or_path, PROCESSOR_CONFIG_MAPPING[cls.input_type]
            )
        elif os.path.isfile(pretrained_model_name_or_path):
            resolved_processor_file = pretrained_model_name_or_path
            is_local = True
        elif from_hf_hub:
            processor_file = PROCESSOR_CONFIG_MAPPING[cls.input_type]
            resolved_processor_file = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename=processor_file,
                cache_dir=cache_dir,
                subfolder=subfolder,
                library_name="PaddleNLP",
                library_version=__version__,
            )
        elif from_aistudio and aistudio_download is not None:
            processor_file = PROCESSOR_CONFIG_MAPPING[cls.input_type]
            if subfolder is not None:
                processor_file = os.path.join(subfolder, processor_file)

            pretrained_model_name_or_path_list = pretrained_model_name_or_path.split("/")
            if len(pretrained_model_name_or_path_list) > 2:
                pretrained_model_name_or_path = os.path.join(
                    pretrained_model_name_or_path_list[0], pretrained_model_name_or_path_list[1]
                )

            resolved_processor_file = aistudio_download(repo_id=pretrained_model_name_or_path, filename=processor_file)
        else:
            # Assuming from community-contributed pretrained models
            processor_file = "/".join(
                [
                    COMMUNITY_MODEL_PREFIX,
                    pretrained_model_name_or_path,
                    PROCESSOR_CONFIG_MAPPING[cls.input_type],
                ]
            )
            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_processor_file = get_path_from_url_with_filelock(processor_file, cache_dir)
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load processor for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'BOS', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {PROCESSOR_CONFIG_MAPPING[cls.input_type]} file"
                )

        try:
            # Load processor dict
            with open(resolved_processor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            processor_dict = json.loads(text)

        except json.JSONDecodeError:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_processor_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_processor_file}")
        else:
            logger.info(f"loading configuration file {processor_file} from cache at {resolved_processor_file}")

        return processor_dict, kwargs

    @classmethod
    def from_dict(cls, processor_dict: Dict[str, Any], **kwargs):
        """
        Instantiates a type of [`~processing_utils.BaseProcessingMixin`] from a Python dictionary of parameters.

        Args:
            processor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~processing_utils.BaseProcessingMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the processor object.

        Returns:
            [`~processing_utils.BaseProcessingMixin`]: The processor object instantiated from those
            parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        processor = cls(**processor_dict)

        # Update processor with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(processor, key):
                setattr(processor, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Processor {processor}")
        if return_unused_kwargs:
            return processor, kwargs
        else:
            return processor

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this processor instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["processor_type"] = self.__class__.__name__

        return output

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]):
        """
        Instantiates a processor of type [`~processing_utils.BaseProcessingMixin`] from the path to a JSON
        file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            A processor of type [`~processing_utils.BaseProcessingMixin`]: The processor object
            instantiated from that JSON file.
        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        processor_dict = json.loads(text)
        return cls(**processor_dict)

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        dictionary = self.to_dict()

        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        # make sure private name "_processor_class" is correctly
        # saved as "processor_class"
        _processor_class = dictionary.pop("_processor_class", None)
        if _processor_class is not None:
            dictionary["processor_class"] = _processor_class

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this processor instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"


class BaseImageProcessor(BaseProcessingMixin):
    input_type = "image"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, images, **kwargs) -> BatchEncoding:
        """Preprocess an image or a batch of images."""
        return self.preprocess(images, **kwargs)

    def preprocess(self, images, **kwargs) -> BatchEncoding:
        raise NotImplementedError("Each image processor must implement its own preprocess method")


class BaseTextProcessor(BaseProcessingMixin):
    input_type = "text"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, text, **kwargs) -> BatchEncoding:
        """Preprocess an image or a batch of images."""
        return self.preprocess(text, **kwargs)

    def preprocess(self, text, **kwargs) -> BatchEncoding:
        raise NotImplementedError("Each image processor must implement its own preprocess method")


class BaseAudioProcessor(BaseProcessingMixin):
    input_type = "audio"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, audios, **kwargs) -> BatchEncoding:
        """Preprocess an audio or a batch of audios."""
        return self.preprocess(audios, **kwargs)

    def preprocess(self, audios, **kwargs) -> BatchEncoding:
        raise NotImplementedError("Each audios processor must implement its own preprocess method")


VALID_SIZE_DICT_KEYS = (
    {"height", "width"},
    {"shortest_edge"},
    {"shortest_edge", "longest_edge"},
)


def is_valid_size_dict(size_dict):
    if not isinstance(size_dict, dict):
        return False

    size_dict_keys = set(size_dict.keys())
    for allowed_keys in VALID_SIZE_DICT_KEYS:
        if size_dict_keys == allowed_keys:
            return True
    return False


def convert_to_size_dict(
    size,
    max_size: Optional[int] = None,
    default_to_square: bool = True,
    height_width_order: bool = True,
):
    # By default, if size is an int we assume it represents a tuple of (size, size).
    if isinstance(size, int) and default_to_square:
        if max_size is not None:
            raise ValueError("Cannot specify both size as an int, with default_to_square=True and max_size")
        return {"height": size, "width": size}
    # In other configs, if size is an int and default_to_square is False, size represents the length of
    # the shortest edge after resizing.
    elif isinstance(size, int) and not default_to_square:
        size_dict = {"shortest_edge": size}
        if max_size is not None:
            size_dict["longest_edge"] = max_size
        return size_dict
    # Otherwise, if size is a tuple it's either (height, width) or (width, height)
    elif isinstance(size, (tuple, list)) and height_width_order:
        return {"height": size[0], "width": size[1]}
    elif isinstance(size, (tuple, list)) and not height_width_order:
        return {"height": size[1], "width": size[0]}

    raise ValueError(f"Could not convert size input to size dict: {size}")


def get_size_dict(
    size: Union[int, Iterable[int], Dict[str, int]] = None,
    max_size: Optional[int] = None,
    height_width_order: bool = True,
    default_to_square: bool = True,
    param_name="size",
) -> dict:
    """
    Converts the old size parameter in the config into the new dict expected in the config. This is to ensure backwards
    compatibility with the old image processor configs and removes ambiguity over whether the tuple is in (height,
    width) or (width, height) format.

    - If `size` is tuple, it is converted to `{"height": size[0], "width": size[1]}` or `{"height": size[1], "width":
    size[0]}` if `height_width_order` is `False`.
    - If `size` is an int, and `default_to_square` is `True`, it is converted to `{"height": size, "width": size}`.
    - If `size` is an int and `default_to_square` is False, it is converted to `{"shortest_edge": size}`. If `max_size`
      is set, it is added to the dict as `{"longest_edge": max_size}`.

    Args:
        size (`Union[int, Iterable[int], Dict[str, int]]`, *optional*):
            The `size` parameter to be cast into a size dictionary.
        max_size (`Optional[int]`, *optional*):
            The `max_size` parameter to be cast into a size dictionary.
        height_width_order (`bool`, *optional*, defaults to `True`):
            If `size` is a tuple, whether it's in (height, width) or (width, height) order.
        default_to_square (`bool`, *optional*, defaults to `True`):
            If `size` is an int, whether to default to a square image or not.
    """
    if not isinstance(size, dict):
        size_dict = convert_to_size_dict(size, max_size, default_to_square, height_width_order)
        logger.info(
            f"{param_name} should be a dictionary on of the following set of keys: {VALID_SIZE_DICT_KEYS}, got {size}."
            f" Converted to {size_dict}.",
        )
    else:
        size_dict = size

    if not is_valid_size_dict(size_dict):
        raise ValueError(
            f"{param_name} must have one of the following set of keys: {VALID_SIZE_DICT_KEYS}, got {size_dict.keys()}"
        )
    return size_dict
