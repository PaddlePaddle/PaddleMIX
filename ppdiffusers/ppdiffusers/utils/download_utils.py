# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import json
import os
import os.path
import re
import tempfile
import warnings
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Optional, Union
from urllib.parse import quote

import requests
from aistudio_sdk.hub import create_repo as aistudio_create_repo
from aistudio_sdk.hub import download as aistudio_base_download
from aistudio_sdk.hub import upload as aistudio_upload
from filelock import FileLock
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from huggingface_hub.file_download import _chmod_and_move, http_get
from huggingface_hub.utils import (
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from packaging import version
from requests import HTTPError
from tqdm import tqdm

from ..version import VERSION as __version__
from .constants import (  # DIFFUSERS; PPDIFFUSERS; TRANSFORMERS; PADDLENLP; PADDLE_SAFETENSORS_WEIGHTS_NAME,; PADDLE_SAFETENSORS_WEIGHTS_NAME_INDEX_NAME,; PADDLE_WEIGHTS_NAME,; PADDLE_WEIGHTS_NAME_INDEX_NAME,; PPNLP_PADDLE_WEIGHTS_INDEX_NAME,; PPNLP_PADDLE_WEIGHTS_NAME,; PPNLP_SAFE_WEIGHTS_INDEX_NAME,; PPNLP_SAFE_WEIGHTS_NAME,; TORCH_SAFETENSORS_WEIGHTS_NAME_INDEX_NAME,; TORCH_WEIGHTS_NAME_INDEX_NAME,; TRANSFORMERS_SAFE_WEIGHTS_INDEX_NAME,; TRANSFORMERS_SAFE_WEIGHTS_NAME,; TRANSFORMERS_TORCH_WEIGHTS_INDEX_NAME,; TRANSFORMERS_TORCH_WEIGHTS_NAME,
    DEPRECATED_REVISION_ARGS,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    PPDIFFUSERS_CACHE,
    PPNLP_BOS_RESOLVE_ENDPOINT,
    TORCH_SAFETENSORS_WEIGHTS_NAME,
    TORCH_WEIGHTS_NAME,
)
from .logging import get_logger

logger = get_logger(__name__)


def _add_subfolder(weights_name: str, subfolder: Optional[str] = None) -> str:
    if subfolder is not None and subfolder != "":
        weights_name = "/".join([subfolder, weights_name])
    return weights_name


def _add_variant(weights_name: str, variant: Optional[str] = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name


# https://github.com/huggingface/diffusers/blob/da2ce1a6b92f48cabe9e9d3944c4ee8b007b2871/src/diffusers/utils/hub_utils.py#L246
def _get_model_file(
    pretrained_model_name_or_path,
    weights_name,
    subfolder="",
    cache_dir=PPDIFFUSERS_CACHE,
    force_download=False,
    revision=None,
    proxies=None,
    resume_download=False,
    local_files_only=None,
    use_auth_token=None,
    user_agent=None,
    commit_hash=None,
    file_lock_timeout=-1,
    from_hf_hub=False,
    from_aistudio=False,
    token=None,
):
    # deprecate
    use_auth_token = use_auth_token or token
    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if os.path.isfile(pretrained_model_name_or_path):
        return pretrained_model_name_or_path
    elif os.path.isdir(pretrained_model_name_or_path):
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, weights_name)):
            # Load from a Paddle checkpoint
            model_file = os.path.join(pretrained_model_name_or_path, weights_name)
            return model_file
        elif subfolder is not None and os.path.isfile(
            os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
        ):
            model_file = os.path.join(pretrained_model_name_or_path, subfolder, weights_name)
            return model_file
        else:
            raise EnvironmentError(
                f"Error no file named {weights_name} found in directory {pretrained_model_name_or_path}."
            )
    else:
        return bos_aistudio_hf_download(
            pretrained_model_name_or_path,
            weights_name,
            subfolder=subfolder,
            cache_dir=cache_dir,
            force_download=force_download,
            revision=revision,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            user_agent=user_agent,
            file_lock_timeout=file_lock_timeout,
            commit_hash=commit_hash,
            from_hf_hub=from_hf_hub,
            from_aistudio=from_aistudio,
        )


# 1. bos download
REPO_TYPES = ["model"]
DEFAULT_REVISION = "main"
# REPO_ID_SEPARATOR = "--"
REGEX_COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")
PPDIFFUSERS_BOS_URL_TEMPLATE = PPNLP_BOS_RESOLVE_ENDPOINT + "/{repo_type}/community/{repo_id}/{revision}/{filename}"


def ppdiffusers_bos_url(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"

    if repo_type is None:
        repo_type = REPO_TYPES[0]
    if repo_type not in REPO_TYPES:
        raise ValueError("Invalid repo type")
    if repo_type == "model":
        repo_type = "models"
    if revision is None:
        revision = DEFAULT_REVISION
    return PPDIFFUSERS_BOS_URL_TEMPLATE.format(
        repo_type=repo_type,
        repo_id=repo_id,
        revision=quote(revision, safe=""),
        filename=quote(filename),
    ).replace(f"/{DEFAULT_REVISION}/", "/")


def repo_folder_name(*, repo_id: str, repo_type: str) -> str:
    # """Return a serialized version of a hf.co repo name and type, safe for disk storage
    # as a single non-nested folder.
    # Example: models--julien-c--EsperBERTo-small
    # """
    # remove all `/` occurrences to correctly convert repo to directory name
    # parts = ["ppdiffusers", f"{repo_type}s", *repo_id.split("/")]
    # return REPO_ID_SEPARATOR.join(parts)
    return repo_id


def ppdiffusers_url_download(
    url_to_download: str,
    cache_dir: Union[str, Path, None] = None,
    filename: Optional[str] = None,
    force_download: bool = False,
    resume_download: bool = False,
    file_lock_timeout: int = -1,
):
    if not url_file_exists(url_to_download):
        raise EntryNotFoundError(f"The file {url_to_download} does not exist.")

    if cache_dir is None:
        cache_dir = PPDIFFUSERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if filename is None:
        filename = url_to_download.split("/")[-1]
    file_path = os.path.join(cache_dir, filename)
    # Prevent parallel downloads of the same file with a lock.
    lock_path = file_path + ".lock"
    # Some Windows versions do not allow for paths longer than 255 characters.
    # In this case, we must specify it is an extended path by using the "\\?\" prefix.
    if os.name == "nt" and len(os.path.abspath(lock_path)) > 255:
        lock_path = "\\\\?\\" + os.path.abspath(lock_path)

    if os.name == "nt" and len(os.path.abspath(file_path)) > 255:
        file_path = "\\\\?\\" + os.path.abspath(file_path)

    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with FileLock(lock_path, timeout=file_lock_timeout):
        # If the download just completed while the lock was activated.
        if os.path.exists(file_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return file_path

        if resume_download:
            incomplete_path = file_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager():
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(  # type: ignore
                tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False
            )
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info("downloading %s to %s", url_to_download, temp_file.name)

            http_get(
                url_to_download,
                temp_file,
                proxies=None,
                resume_size=resume_size,
                headers=None,
            )

        logger.info("storing %s in cache at %s", url_to_download, file_path)
        _chmod_and_move(temp_file.name, Path(file_path))
    try:
        os.remove(lock_path)
    except OSError:
        pass
    return file_path


def url_file_exists(url: str) -> bool:
    """check whether the url file exists

        refer to: https://stackoverflow.com/questions/2486145/python-check-if-url-to-jpg-exists

    Args:
        url (str): the url of target file

    Returns:
        bool: whether the url file exists
    """

    def is_url(path):
        """
        Whether path is URL.
        Args:
            path (string): URL string or not.
        """
        return path.startswith("http://") or path.startswith("https://")

    if not is_url(url):
        return False

    result = requests.head(url)
    return result.status_code == requests.codes.ok


def bos_download(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    force_download: bool = False,
    resume_download: bool = False,
    file_lock_timeout: int = -1,
):
    if cache_dir is None:
        cache_dir = PPDIFFUSERS_CACHE
    if revision is None:
        revision = DEFAULT_REVISION
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        # This is used to create a URL, and not a local path, hence the forward slash.
        filename = f"{subfolder}/{filename}"

    if repo_type is None:
        repo_type = REPO_TYPES[0]

    if repo_type not in REPO_TYPES:
        raise ValueError(f"Invalid repo type: {repo_type}. Accepted repo types are:" f" {str(REPO_TYPES)}")
    storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type))
    os.makedirs(storage_folder, exist_ok=True)

    # cross platform transcription of filename, to be used as a local file path.
    relative_filename = os.path.join(*filename.split("/"))

    if REGEX_COMMIT_HASH.match(revision):
        pointer_path = os.path.join(storage_folder, revision, relative_filename)
    else:
        pointer_path = os.path.join(storage_folder, relative_filename)

    if os.path.exists(pointer_path) and not force_download:
        return pointer_path

    url_to_download = ppdiffusers_bos_url(repo_id, filename, repo_type=repo_type, revision=revision)
    if not url_file_exists(url_to_download):
        raise EntryNotFoundError(f"The file {url_to_download} does not exist.")

    blob_path = os.path.join(storage_folder, filename)
    # Prevent parallel downloads of the same file with a lock.
    lock_path = blob_path + ".lock"

    # Some Windows versions do not allow for paths longer than 255 characters.
    # In this case, we must specify it is an extended path by using the "\\?\" prefix.
    if os.name == "nt" and len(os.path.abspath(lock_path)) > 255:
        lock_path = "\\\\?\\" + os.path.abspath(lock_path)

    if os.name == "nt" and len(os.path.abspath(blob_path)) > 255:
        blob_path = "\\\\?\\" + os.path.abspath(blob_path)

    os.makedirs(os.path.dirname(lock_path), exist_ok=True)

    with FileLock(lock_path, timeout=file_lock_timeout):
        # If the download just completed while the lock was activated.
        if os.path.exists(pointer_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return pointer_path

        if resume_download:
            incomplete_path = blob_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager():
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(  # type: ignore
                tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False
            )
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info("downloading %s to %s", url_to_download, temp_file.name)

            http_get(
                url_to_download,
                temp_file,
                proxies=None,
                resume_size=resume_size,
                headers=None,
            )

        logger.info("storing %s in cache at %s", url_to_download, blob_path)
        _chmod_and_move(temp_file.name, Path(blob_path))
    try:
        os.remove(lock_path)
    except OSError:
        pass

    return pointer_path


# 2. aistudio download
class UnauthorizedError(Exception):
    pass


def aistudio_download(
    repo_id: str,
    filename: str = None,
    cache_dir: Optional[str] = None,
    subfolder: Optional[str] = "",
    revision: Optional[str] = None,
    **kwargs,
):

    if revision is None:
        revision = "master"
    if subfolder == "":
        subfolder = None
    filename = _add_subfolder(filename, subfolder)
    download_kwargs = {}
    if revision is not None:
        download_kwargs["revision"] = revision
    # currently do not support cache_dir
    # if cache_dir is not None:
    #     download_kwargs["cache_dir"] = cache_dir
    res = aistudio_base_download(
        repo_id=repo_id,
        filename=filename,
        **download_kwargs,
    )
    if "path" in res:
        return res["path"]
    else:
        if res["error_code"] == 10001:
            raise ValueError("Illegal argument error")
        elif res["error_code"] == 10002:
            raise UnauthorizedError(
                "Unauthorized Access. Please ensure that you have provided the AIStudio Access Token and you have access to the requested asset"
            )
        elif res["error_code"] == 12001:
            raise EntryNotFoundError(f"Cannot find the requested file '{filename}' in repo '{repo_id}'")
        else:
            raise Exception(f"Unknown error: {res}")


# MERGED Download utils
def bos_aistudio_hf_download(
    pretrained_model_name_or_path,
    weights_name,
    *,
    subfolder=None,
    cache_dir=None,
    force_download=False,
    revision=None,
    proxies=None,
    resume_download=False,
    local_files_only=None,
    use_auth_token=None,
    user_agent=None,
    file_lock_timeout=-1,
    commit_hash=None,
    from_hf_hub=False,
    from_aistudio=False,
    token=None,
):
    token = use_auth_token or token
    if subfolder is None:
        subfolder = ""
    if from_aistudio:
        model_file = aistudio_download(
            pretrained_model_name_or_path,
            weights_name,
            cache_dir=cache_dir,
            subfolder=subfolder,
            revision=revision or commit_hash,
        )
        return model_file
    elif from_hf_hub:
        # 1. First check if deprecated way of loading from branches is used
        if (
            revision in DEPRECATED_REVISION_ARGS
            and (weights_name == TORCH_WEIGHTS_NAME or weights_name == TORCH_SAFETENSORS_WEIGHTS_NAME)
            and version.parse(version.parse(__version__).base_version) >= version.parse("0.22.0")
        ):
            try:
                model_file = hf_hub_download(
                    pretrained_model_name_or_path,
                    filename=_add_variant(weights_name, revision),
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    subfolder=subfolder,
                    revision=revision or commit_hash,
                )
                warnings.warn(
                    f"Loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'` is deprecated. Loading instead from `revision='main'` with `variant={revision}`. Loading model variants via `revision='{revision}'` will be removed in diffusers v1. Please use `variant='{revision}'` instead.",
                    FutureWarning,
                )
                return model_file
            except:  # noqa: E722
                warnings.warn(
                    f"You are loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'`. This behavior is deprecated and will be removed in diffusers v1. One should use `variant='{revision}'` instead. However, it appears that {pretrained_model_name_or_path} currently does not have a {_add_variant(weights_name, revision)} file in the 'main' branch of {pretrained_model_name_or_path}. \n The Diffusers team and community would be very grateful if you could open an issue: https://github.com/huggingface/diffusers/issues/new with the title '{pretrained_model_name_or_path} is missing {_add_variant(weights_name, revision)}' so that the correct variant file can be added.",
                    FutureWarning,
                )
        # 2. Load model file as usual
        try:
            model_file = hf_hub_download(
                pretrained_model_name_or_path,
                filename=weights_name,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token,
                user_agent=user_agent,
                subfolder=subfolder,
                revision=revision,
            )
            return model_file

        except RepositoryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                "token having permission to this repo with `use_auth_token` or log in with `huggingface-cli "
                "login`."
            )
        except RevisionNotFoundError:
            raise EnvironmentError(
                f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                "this model name. Check the model page at "
                f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
            )
        except EntryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {weights_name}."
            )
        except HTTPError as err:
            raise EnvironmentError(
                f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n{err}"
            )
        except ValueError:
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                f" directory containing a file named {weights_name} or"
                " \nCheckout your internet connection or see how to run the library in"
                " offline mode at 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
            )
        except EnvironmentError:
            raise EnvironmentError(
                f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                f"containing a file named {weights_name}"
            )
    else:
        try:
            model_file = bos_download(
                pretrained_model_name_or_path,
                filename=weights_name,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                subfolder=subfolder,
                revision=revision,
                file_lock_timeout=file_lock_timeout,
            )
            return model_file
        except HTTPError as err:
            raise EnvironmentError(
                f"{err}!\n"
                f"There was a specific connection error when trying to load '{pretrained_model_name_or_path}'! "
                f"We couldn't connect to '{PPNLP_BOS_RESOLVE_ENDPOINT}' to load this model, couldn't find it "
                f"in the cached files and it looks like '{pretrained_model_name_or_path}' is not the path to a "
                f"directory containing a file named '{weights_name}'."
            )
        except EnvironmentError:
            raise EnvironmentError(
                f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                f"'{PPNLP_BOS_RESOLVE_ENDPOINT}', make sure you don't have a local directory with the same name. "
                f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                f"containing a file named '{weights_name}'"
            )
        except KeyboardInterrupt:
            raise EnvironmentError(
                "You have interrupted the download, if you want to continue the download, you can set `resume_download=True`!"
            )


class SaveToAistudioMixin:
    """
    A Mixin to push a model, scheduler, or pipeline to the Aistudio Hub.
    """

    def _upload_folder_aistudio(
        self,
        working_dir: Union[str, os.PathLike],
        repo_id: str,
        token: Optional[str] = None,
        commit_message: Optional[str] = None,
        root_dir=None,
        create_pr: bool = False,
    ):
        """
        Uploads all files in `working_dir` to `repo_id`.
        """
        assert "/" in repo_id, "Please specify the repo id in format of `user_id/repo_name`"
        token_kwargs = {}
        if token is not None:
            token_kwargs["token"] = token
        if root_dir is None:
            root_dir = working_dir
        if commit_message is None:
            if "Model" in self.__class__.__name__:
                commit_message = "Upload model"
            elif "Scheduler" in self.__class__.__name__:
                commit_message = "Upload scheduler"
            else:
                commit_message = f"Upload {self.__class__.__name__}"

        # Upload model and return
        logger.info(f"Pushing to the {repo_id}. This might take a while")
        for filename in Path(working_dir).glob("**/*"):
            if filename.is_file():
                path_in_repo = str(filename.relative_to(root_dir))
                path_or_fileobj = str(filename.absolute())
                res = aistudio_upload(
                    repo_id=repo_id,
                    path_or_fileobj=path_or_fileobj,
                    path_in_repo=path_in_repo,
                    commit_message=commit_message,
                    **token_kwargs,
                )

                if "error_code" in res:
                    logger.error(
                        f"Failed to upload {filename}, error_code: {res['error_code']}, error_msg: {res['error_msg']}"
                    )
                else:
                    logger.info(f"{filename}: {res['message']}")

    def save_to_aistudio(
        self,
        repo_id,
        commit_message: Optional[str] = None,
        private: Optional[bool] = False,
        token: Optional[str] = None,
        license="creativeml-openrail-m",
        exist_ok=True,
        subfolder=None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        to_diffusers: bool = None,
        max_shard_size: Union[int, str] = "5GB",
    ):
        """
        Uploads all elements of this model to a new AiStudio Hub repository.
        Args:
            repo_id (str): Repository name for your model/tokenizer in the Hub.
            private (bool, optional): Whether the model/tokenizer is set to private. Defaults to True.
            license (str): The license of your model/tokenizer. Defaults to: "creativeml-openrail-m".
            exist_ok (bool, optional): Whether to override existing repository. Defaults to: True.
            subfolder (str, optional): Push to a subfolder of the repo instead of the root
        """
        assert "/" in repo_id, "Please specify the repo id in format of `user_id/repo_name`"
        token_kwargs = {}
        if token is not None:
            token_kwargs["token"] = token

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

        with tempfile.TemporaryDirectory() as root_dir:
            if subfolder is not None:
                save_dir = os.path.join(root_dir, subfolder)
            else:
                save_dir = root_dir

            # Save all files.
            save_kwargs = {}
            if "Scheduler" not in self.__class__.__name__:
                save_kwargs.update({"safe_serialization": safe_serialization})
                save_kwargs.update({"variant": variant})
                save_kwargs.update({"to_diffusers": to_diffusers})
                save_kwargs.update({"max_shard_size": max_shard_size})

            # save model
            self.save_pretrained(
                save_dir,
                **save_kwargs,
            )

            return self._upload_folder_aistudio(
                save_dir,
                repo_id,
                commit_message=commit_message,
                root_dir=root_dir,
                **token_kwargs,
            )


def get_checkpoint_shard_files(
    pretrained_model_name_or_path,
    index_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    local_files_only=False,
    use_auth_token=None,
    user_agent=None,
    revision=None,
    subfolder="",
    commit_hash=None,
    from_hf_hub=False,
    from_aistudio=False,
    token=None,
):
    use_auth_token = use_auth_token or token
    if subfolder is None:
        subfolder = ""
    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")

    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()

    # First, let's deal with local folder.
    if os.path.isdir(pretrained_model_name_or_path):
        shard_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in shard_filenames]
        return shard_filenames, sharded_metadata

    # At this stage pretrained_model_name_or_path is a model identifier on the Hub
    cached_filenames = []
    # Check if the model is already cached or not. We only try the last checkpoint, this should cover most cases of
    # downloaded (if interrupted).
    try:
        last_shard = try_to_load_from_cache(
            pretrained_model_name_or_path, shard_filenames[-1], cache_dir=cache_dir, revision=commit_hash
        )
    except:
        last_shard = None
    show_progress_bar = last_shard is None or force_download
    for shard_filename in tqdm(shard_filenames, desc="Downloading shards", disable=not show_progress_bar):
        # Load from URL
        cached_filename = _get_model_file(
            pretrained_model_name_or_path,
            shard_filename,
            subfolder=subfolder,
            cache_dir=cache_dir,
            force_download=force_download,
            revision=revision,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            user_agent=user_agent,
            commit_hash=commit_hash,
            from_hf_hub=from_hf_hub,
            from_aistudio=from_aistudio,
        )
        cached_filenames.append(cached_filename)

    return cached_filenames, sharded_metadata
