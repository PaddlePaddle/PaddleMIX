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

import os
from pathlib import Path
from typing import Dict, Literal, Optional, Union

from huggingface_hub.utils import (
    EntryNotFoundError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from requests import HTTPError

from .aistudio_hub.download import aistudio_hub_download, aistudio_hub_file_exists
from .bos.download import bos_download, bos_file_exists
from .hf_hub.download import hf_hub_download, hf_hub_file_exists


def bos_aistudio_hf_download(
    repo_id: str = None,
    filename: str = None,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    local_dir: Union[str, Path, None] = None,
    local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
    user_agent: Union[Dict, str, None] = None,
    force_download: bool = False,
    proxies: Optional[Dict] = None,
    etag_timeout: float = 10,
    resume_download: bool = False,
    token: Union[bool, str, None] = None,
    local_files_only: bool = False,
    endpoint: Optional[str] = None,
    url: Optional[str] = None,
    from_bos: bool = True,
    from_aistudio: bool = False,
    from_hf_hub: bool = False,
) -> str:
    download_kwargs = dict(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder if subfolder is not None else "",
        repo_type=repo_type,
        revision=revision,
        library_name=library_name,
        library_version=library_version,
        cache_dir=cache_dir,
        local_dir=local_dir,
        local_dir_use_symlinks=local_dir_use_symlinks,
        user_agent=user_agent,
        force_download=force_download,
        proxies=proxies,
        etag_timeout=etag_timeout,
        resume_download=resume_download,
        token=token,
        local_files_only=local_files_only,
        endpoint=endpoint,
    )
    cached_file = None
    log_endpoint = "N/A"
    log_filename = os.path.join(download_kwargs["subfolder"], filename)
    try:
        if from_bos:
            log_endpoint = "BOS"
            download_kwargs["url"] = url
            cached_file = bos_download(
                **download_kwargs,
            )
        elif from_aistudio:
            log_endpoint = "Aistudio Hub"
            cached_file = aistudio_hub_download(
                **download_kwargs,
            )
        elif from_hf_hub:
            log_endpoint = "Huggingface Hub"
            cached_file = hf_hub_download(
                **download_kwargs,
            )
    except LocalEntryNotFoundError:
        raise EnvironmentError(
            "Cannot find the requested files in the cached path and"
            " outgoing traffic has been disabled. To enable model look-ups"
            " and downloads online, set 'local_files_only' to False."
        )
    except RepositoryNotFoundError:
        raise EnvironmentError(
            f"{repo_id} is not a local folder and is not a valid model identifier "
            f"listed on '{log_endpoint}'\nIf this is a private repository, make sure to pass a "
            "token having permission to this repo."
        )
    except RevisionNotFoundError:
        raise EnvironmentError(
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
            "this model name. Check the model page at "
            f"'{log_endpoint}' for available revisions."
        )
    except EntryNotFoundError:
        raise EnvironmentError(f"{repo_id} does not appear to have a file named {log_filename}.")
    except HTTPError as err:
        raise EnvironmentError(f"There was a specific connection error when trying to load {repo_id}:\n{err}")
    except ValueError:
        raise EnvironmentError(
            f"We couldn't connect to '{log_endpoint}' to load this model, couldn't find it"
            f" in the cached files and it looks like {repo_id} is not the path to a"
            f" directory containing a file named {log_filename} or"
            " \nCheckout your internet connection or see how to run the library in offline mode."
        )
    except EnvironmentError:
        raise EnvironmentError(
            f"Can't load the model for '{repo_id}'. If you were trying to load it from "
            f"'{log_endpoint}', make sure you don't have a local directory with the same name. "
            f"Otherwise, make sure '{repo_id}' is the correct path to a directory "
            f"containing a file named {log_filename}"
        )
    return cached_file


def bos_aistudio_hf_file_exist(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    endpoint: Optional[str] = None,
    from_bos: bool = True,
    from_aistudio: bool = False,
    from_hf_hub: bool = False,
):
    if subfolder is None:
        subfolder = ""
    filename = os.path.join(subfolder, filename)
    if from_bos:
        out = bos_file_exists(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            token=token,  # donot need token
            endpoint=endpoint,
        )
    elif from_aistudio:
        out = aistudio_hub_file_exists(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            token=token,
            endpoint=endpoint,
        )
    elif from_hf_hub:
        out = hf_hub_file_exists(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            token=token,
        )
    return out
