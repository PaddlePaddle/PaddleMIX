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

import copy
import io
import logging
import os
import re
import shutil
import stat
import tempfile
import threading
import time
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import BinaryIO, Callable, Dict, Generator, Literal, Optional, Union
from urllib.parse import quote, urlparse

import requests
from filelock import FileLock
from huggingface_hub.utils import (
    EntryNotFoundError,
    FileMetadataError,
    GatedRepoError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    hf_raise_for_status,
    tqdm,
)
from requests import Response
from requests.adapters import HTTPAdapter
from requests.models import PreparedRequest

logger = logging.getLogger(__name__)

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}


def _is_true(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.upper() in ENV_VARS_TRUE_VALUES


def _as_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    return int(value)


# Regex to get filename from a "Content-Disposition" header for CDN-served files
HEADER_FILENAME_PATTERN = re.compile(r'filename="(?P<filename>.*?)";')


X_AMZN_TRACE_ID = "X-Amzn-Trace-Id"
X_REQUEST_ID = "x-request-id"
DEFAULT_ETAG_TIMEOUT = 10


VERSION = "0.1.5"
DEFAULT_REQUEST_TIMEOUT = 10

default_home = os.path.join(os.path.expanduser("~"), ".cache")
AISTUDIO_HOME = os.path.expanduser(
    os.getenv(
        "AISTUDIO_HOME",
        os.path.join(os.getenv("XDG_CACHE_HOME", default_home), "paddle"),
    )
)
AISTUDIO_TOKEN_PATH = os.path.join(AISTUDIO_HOME, "token")
AISTUDIO_HUB_DISABLE_IMPLICIT_TOKEN: bool = _is_true(os.environ.get("AISTUDIO_HUB_DISABLE_IMPLICIT_TOKEN"))
AISTUDIO_HUB_OFFLINE = _is_true(os.environ.get("AISTUDIO_HUB_OFFLINE"))

AISTUDIO_HUB_ETAG_TIMEOUT: int = _as_int(os.environ.get("AISTUDIO_HUB_ETAG_TIMEOUT")) or DEFAULT_ETAG_TIMEOUT
default_cache_path = os.path.join(AISTUDIO_HOME, "aistudio")
AISTUDIO_HUB_CACHE = os.getenv("AISTUDIO_HUB_CACHE", default_cache_path)

AISTUDIO_HUB_DISABLE_SYMLINKS_WARNING: bool = _is_true(os.environ.get("AISTUDIO_HUB_DISABLE_SYMLINKS_WARNING"))
AISTUDIO_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD: int = (
    _as_int(os.environ.get("AISTUDIO_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD")) or 5 * 1024 * 1024
)
DOWNLOAD_CHUNK_SIZE = 10 * 1024 * 1024

DEFAULT_REVISION = "master"
REPO_TYPE_DATASET = "dataset"
REPO_TYPE_SPACE = "space"
REPO_TYPE_MODEL = "model"
REPO_TYPES = [None, REPO_TYPE_MODEL, REPO_TYPE_DATASET, REPO_TYPE_SPACE]
REPO_TYPES_URL_PREFIXES = {
    REPO_TYPE_DATASET: "datasets/",
    REPO_TYPE_SPACE: "spaces/",
}

ENDPOINT = os.getenv("AISTUDIO_ENDPOINT", "http://git.aistudio.baidu.com")
AISTUDIO_CO_URL_TEMPLATE = ENDPOINT + "/api/v1/repos/{user_name}/{repo_name}/contents/{filename}"
AISTUDIO_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD: int = (
    _as_int(os.environ.get("AISTUDIO_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD")) or 5 * 1024 * 1024
)
REPO_ID_SEPARATOR = "--"
# Return value when trying to load a file from cache but the file does not exist in the distant repo.
# _CACHED_NO_EXIST = object()
# _CACHED_NO_EXIST_T = Any
REGEX_COMMIT_HASH = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_DOWNLOAD_TIMEOUT = 10
AISTUDIO_HUB_DOWNLOAD_TIMEOUT: int = (
    _as_int(os.environ.get("AISTUDIO_HUB_DOWNLOAD_TIMEOUT")) or DEFAULT_DOWNLOAD_TIMEOUT
)


class LocalTokenNotFoundError(EnvironmentError):
    """Raised if local token is required but not found."""


def repo_folder_name(*, repo_id: str, repo_type: str) -> str:
    """Return a serialized version of a aistudio repo name and type, safe for disk storage
    as a single non-nested folder.

    Example: models--julien-c--EsperBERTo-small
    """
    # remove all `/` occurrences to correctly convert repo to directory name
    parts = [f"{repo_type}s", *repo_id.split("/")]
    return REPO_ID_SEPARATOR.join(parts)


def _get_pointer_path(storage_folder: str, revision: str, relative_filename: str) -> str:
    # Using `os.path.abspath` instead of `Path.resolve()` to avoid resolving symlinks
    snapshot_path = os.path.join(storage_folder, "snapshots")
    pointer_path = os.path.join(snapshot_path, revision, relative_filename)
    if Path(os.path.abspath(snapshot_path)) not in Path(os.path.abspath(pointer_path)).parents:
        raise ValueError(
            "Invalid pointer path: cannot create pointer path in snapshot folder if"
            f" `storage_folder='{storage_folder}'`, `revision='{revision}'` and"
            f" `relative_filename='{relative_filename}'`."
        )
    return pointer_path


def _create_symlink(src: str, dst: str, new_blob: bool = False) -> None:
    """Create a symbolic link named dst pointing to src.

    By default, it will try to create a symlink using a relative path. Relative paths have 2 advantages:
    - If the cache_folder is moved (example: back-up on a shared drive), relative paths within the cache folder will
      not brake.
    - Relative paths seems to be better handled on Windows. Issue was reported 3 times in less than a week when
      changing from relative to absolute paths. See https://github.com/huggingface/huggingface_hub/issues/1398,
      https://github.com/huggingface/diffusers/issues/2729 and https://github.com/huggingface/transformers/pull/22228.
      NOTE: The issue with absolute paths doesn't happen on admin mode.
    When creating a symlink from the cache to a local folder, it is possible that a relative path cannot be created.
    This happens when paths are not on the same volume. In that case, we use absolute paths.


    The result layout looks something like
        └── [ 128]  snapshots
            ├── [ 128]  2439f60ef33a0d46d85da5001d52aeda5b00ce9f
            │   ├── [  52]  README.md -> ../../../blobs/d7edf6bd2a681fb0175f7735299831ee1b22b812
            │   └── [  76]  pytorch_model.bin -> ../../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd

    If symlinks cannot be created on this platform (most likely to be Windows), the workaround is to avoid symlinks by
    having the actual file in `dst`. If it is a new file (`new_blob=True`), we move it to `dst`. If it is not a new file
    (`new_blob=False`), we don't know if the blob file is already referenced elsewhere. To avoid breaking existing
    cache, the file is duplicated on the disk.

    In case symlinks are not supported, a warning message is displayed to the user once when loading `huggingface_hub`.
    The warning message can be disable with the `DISABLE_SYMLINKS_WARNING` environment variable.
    """
    try:
        os.remove(dst)
    except OSError:
        pass

    abs_src = os.path.abspath(os.path.expanduser(src))
    abs_dst = os.path.abspath(os.path.expanduser(dst))
    abs_dst_folder = os.path.dirname(abs_dst)

    # Use relative_dst in priority
    try:
        relative_src = os.path.relpath(abs_src, abs_dst_folder)
    except ValueError:
        # Raised on Windows if src and dst are not on the same volume. This is the case when creating a symlink to a
        # local_dir instead of within the cache directory.
        # See https://docs.python.org/3/library/os.path.html#os.path.relpath
        relative_src = None

    try:
        commonpath = os.path.commonpath([abs_src, abs_dst])
        _support_symlinks = are_symlinks_supported(commonpath)
    except ValueError:
        # Raised if src and dst are not on the same volume. Symlinks will still work on Linux/Macos.
        # See https://docs.python.org/3/library/os.path.html#os.path.commonpath
        _support_symlinks = os.name != "nt"
    except PermissionError:
        # Permission error means src and dst are not in the same volume (e.g. destination path has been provided
        # by the user via `local_dir`. Let's test symlink support there)
        _support_symlinks = are_symlinks_supported(abs_dst_folder)

    # Symlinks are supported => let's create a symlink.
    if _support_symlinks:
        src_rel_or_abs = relative_src or abs_src
        logger.debug(f"Creating pointer from {src_rel_or_abs} to {abs_dst}")
        try:
            os.symlink(src_rel_or_abs, abs_dst)
            return
        except FileExistsError:
            if os.path.islink(abs_dst) and os.path.realpath(abs_dst) == os.path.realpath(abs_src):
                # `abs_dst` already exists and is a symlink to the `abs_src` blob. It is most likely that the file has
                # been cached twice concurrently (exactly between `os.remove` and `os.symlink`). Do nothing.
                return
            else:
                # Very unlikely to happen. Means a file `dst` has been created exactly between `os.remove` and
                # `os.symlink` and is not a symlink to the `abs_src` blob file. Raise exception.
                raise
        except PermissionError:
            # Permission error means src and dst are not in the same volume (e.g. download to local dir) and symlink
            # is supported on both volumes but not between them. Let's just make a hard copy in that case.
            pass

    # Symlinks are not supported => let's move or copy the file.
    if new_blob:
        logger.info(f"Symlink not supported. Moving file from {abs_src} to {abs_dst}")
        shutil.move(abs_src, abs_dst)
    else:
        logger.info(f"Symlink not supported. Copying file from {abs_src} to {abs_dst}")
        shutil.copyfile(abs_src, abs_dst)


_are_symlinks_supported_in_dir: Dict[str, bool] = {}


def _set_write_permission_and_retry(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)


@contextmanager
def SoftTemporaryDirectory(
    suffix: Optional[str] = None,
    prefix: Optional[str] = None,
    dir: Optional[Union[Path, str]] = None,
    **kwargs,
) -> Generator[str, None, None]:
    """
    Context manager to create a temporary directory and safely delete it.

    If tmp directory cannot be deleted normally, we set the WRITE permission and retry.
    If cleanup still fails, we give up but don't raise an exception. This is equivalent
    to  `tempfile.TemporaryDirectory(..., ignore_cleanup_errors=True)` introduced in
    Python 3.10.

    See https://www.scivision.dev/python-tempfile-permission-error-windows/.
    """
    tmpdir = tempfile.TemporaryDirectory(prefix=prefix, suffix=suffix, dir=dir, **kwargs)
    yield tmpdir.name

    try:
        # First once with normal cleanup
        shutil.rmtree(tmpdir.name)
    except Exception:
        # If failed, try to set write permission and retry
        try:
            shutil.rmtree(tmpdir.name, onerror=_set_write_permission_and_retry)
        except Exception:
            pass

    # And finally, cleanup the tmpdir.
    # If it fails again, give up but do not throw error
    try:
        tmpdir.cleanup()
    except Exception:
        pass


def are_symlinks_supported(cache_dir: Union[str, Path, None] = None) -> bool:
    """Return whether the symlinks are supported on the machine.

    Since symlinks support can change depending on the mounted disk, we need to check
    on the precise cache folder. By default, the default Aistudio cache directory is checked.

    Args:
        cache_dir (`str`, `Path`, *optional*):
            Path to the folder where cached files are stored.

    Returns: [bool] Whether symlinks are supported in the directory.
    """
    # Defaults to Aistudio cache
    if cache_dir is None:
        cache_dir = AISTUDIO_HUB_CACHE
    cache_dir = str(Path(cache_dir).expanduser().resolve())  # make it unique

    # Check symlink compatibility only once (per cache directory) at first time use
    if cache_dir not in _are_symlinks_supported_in_dir:
        _are_symlinks_supported_in_dir[cache_dir] = True

        os.makedirs(cache_dir, exist_ok=True)
        with SoftTemporaryDirectory(dir=cache_dir) as tmpdir:
            src_path = Path(tmpdir) / "dummy_file_src"
            src_path.touch()
            dst_path = Path(tmpdir) / "dummy_file_dst"

            # Relative source path as in `_create_symlink``
            relative_src = os.path.relpath(src_path, start=os.path.dirname(dst_path))
            try:
                os.symlink(relative_src, dst_path)
            except OSError:
                # Likely running on Windows
                _are_symlinks_supported_in_dir[cache_dir] = False

                if not AISTUDIO_HUB_DISABLE_SYMLINKS_WARNING:
                    message = (
                        "`huggingface_hub` cache-system uses symlinks by default to"
                        " efficiently store duplicated files but your machine does not"
                        f" support them in {cache_dir}. Caching files will still work"
                        " but in a degraded version that might require more space on"
                        " your disk. This warning can be disabled by setting the"
                        " `AISTUDIO_HUB_DISABLE_SYMLINKS_WARNING` environment variable."
                    )
                    if os.name == "nt":
                        message += (
                            "\nTo support symlinks on Windows, you either need to"
                            " activate Developer Mode or to run Python as an"
                            " administrator. In order to see activate developer mode,"
                            " see this article:"
                            " https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development"
                        )
                    warnings.warn(message)

    return _are_symlinks_supported_in_dir[cache_dir]


def _to_local_dir(
    path: str, local_dir: str, relative_filename: str, use_symlinks: Union[bool, Literal["auto"]]
) -> str:
    """Place a file in a local dir (different than cache_dir).

    Either symlink to blob file in cache or duplicate file depending on `use_symlinks` and file size.
    """
    # Using `os.path.abspath` instead of `Path.resolve()` to avoid resolving symlinks
    local_dir_filepath = os.path.join(local_dir, relative_filename)
    if Path(os.path.abspath(local_dir)) not in Path(os.path.abspath(local_dir_filepath)).parents:
        raise ValueError(
            f"Cannot copy file '{relative_filename}' to local dir '{local_dir}': file would not be in the local"
            " directory."
        )

    os.makedirs(os.path.dirname(local_dir_filepath), exist_ok=True)
    real_blob_path = os.path.realpath(path)

    # If "auto" (default) copy-paste small files to ease manual editing but symlink big files to save disk
    if use_symlinks == "auto":
        use_symlinks = os.stat(real_blob_path).st_size > AISTUDIO_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD

    if use_symlinks:
        _create_symlink(real_blob_path, local_dir_filepath, new_blob=False)
    else:
        shutil.copyfile(real_blob_path, local_dir_filepath)
    return local_dir_filepath


def aistudio_hub_url(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> str:
    """Construct the URL of a file from the given information.

    The resolved address can either be a huggingface.co-hosted url, or a link to
    Cloudfront (a Content Delivery Network, or CDN) for large files which are
    more than a few MBs.

    Args:
        repo_id (`str`):
            A namespace (user or an organization) name and a repo name separated
            by a `/`.
        filename (`str`):
            The name of the file in the repo.
        subfolder (`str`, *optional*):
            An optional value corresponding to a folder inside the repo.
        repo_type (`str`, *optional*):
            Set to `"dataset"` or `"space"` if downloading from a dataset or space,
            `None` or `"model"` if downloading from a model. Default is `None`.
        revision (`str`, *optional*):
            An optional Git revision id which can be a branch name, a tag, or a
            commit hash.

    Example:

    ```python
    >>> from . import aistudio_hub_url

    >>> aistudio_hub_url(
    ...     repo_id="julien-c/EsperBERTo-small", filename="pytorch_model.bin"
    ... )
    ```

    <Tip>

    Notes:

        Cloudfront is replicated over the globe so downloads are way faster for
        the end user (and it also lowers our bandwidth costs).

        Cloudfront aggressively caches files by default (default TTL is 24
        hours), however this is not an issue here because we implement a
        git-based versioning system on huggingface.co, which means that we store
        the files on S3/Cloudfront in a content-addressable way (i.e., the file
        name is its hash). Using content-addressable filenames means cache can't
        ever be stale.

        In terms of client-side caching from this library, we base our caching
        on the objects' entity tag (`ETag`), which is an identifier of a
        specific version of a resource [1]_. An object's ETag is: its git-sha1
        if stored in git, or its sha256 if stored in git-lfs.

    </Tip>

    References:

    -  [1] https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag
    """
    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"

    if repo_type not in REPO_TYPES:
        raise ValueError("Invalid repo type")

    if repo_type in REPO_TYPES_URL_PREFIXES:
        repo_id = REPO_TYPES_URL_PREFIXES[repo_type] + repo_id

    if revision is None:
        revision = DEFAULT_REVISION

    # NEW ADD
    if "/" not in repo_id:
        raise ValueError("repo_id must be in the format of 'namespace/name'")
    user_name, repo_name = repo_id.split("/")
    user_name = user_name.strip()
    repo_name = repo_name.strip()

    url = AISTUDIO_CO_URL_TEMPLATE.format(
        user_name=quote(user_name, safe=""), repo_name=quote(repo_name, safe=""), filename=quote(filename, safe="")
    )
    # Update endpoint if provided
    if endpoint is not None and url.startswith(ENDPOINT):
        url = endpoint + url[len(ENDPOINT) :]

    if revision != "master":
        url += f"?ref={quote(revision, safe='')}"
    return url


def _clean_token(token: Optional[str]) -> Optional[str]:
    """Clean token by removing trailing and leading spaces and newlines.

    If token is an empty string, return None.
    """
    if token is None:
        return None
    return token.replace("\r", "").replace("\n", "").strip() or None


def _get_token_from_environment() -> Optional[str]:
    return _clean_token(os.environ.get("AISTUDIO_ACCESS_TOKEN") or os.environ.get("AISTUDIO_TOKEN"))


def _get_token_from_file() -> Optional[str]:
    try:
        return _clean_token(Path(AISTUDIO_TOKEN_PATH).read_text())
    except FileNotFoundError:
        return None


def get_token() -> Optional[str]:
    """
    Get token if user is logged in.

    Note: in most cases, you should use [`build_aistudio_headers`] instead. This method is only useful
          if you want to retrieve the token for other purposes than sending an HTTP request.

    Token is retrieved in priority from the `AISTUDIO_ACCESS_TOKEN` environment variable. Otherwise, we read the token file located
    in the Aistudio home folder. Returns None if user is not logged in.

    Returns:
        `str` or `None`: The token, `None` if it doesn't exist.
    """
    return _get_token_from_environment() or _get_token_from_file()


def get_token_to_send(token: Optional[Union[bool, str]]) -> Optional[str]:
    """Select the token to send from either `token` or the cache."""
    # Case token is explicitly provided
    if isinstance(token, str):
        return token

    # Case token is explicitly forbidden
    if token is False:
        return None

    # Token is not provided: we get it from local cache
    cached_token = get_token()

    # Case token is explicitly required
    if token is True:
        if cached_token is None:
            raise LocalTokenNotFoundError(
                "Token is required (`token=True`), but no token found. You"
                " to provide a token or be logged in to Aistudio Hub . See"
                "https://ai.baidu.com/ai-doc/AISTUDIO/slmkadt9z#2-%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8%E8%AE%BF%E9%97%AE%E4%BB%A4%E7%89%8C."
            )
        return cached_token

    # Case implicit use of the token is forbidden by env variable
    if AISTUDIO_HUB_DISABLE_IMPLICIT_TOKEN:
        return None

    # Otherwise: we use the cached token as the user has not explicitly forbidden it
    return cached_token


def _validate_token_to_send(token: Optional[str], is_write_action: bool) -> None:
    if is_write_action:
        if token is None:
            raise ValueError(
                "Token is required (write-access action) but no token found. You need"
                " to provide a token or be logged in to Aistudio Hub . See"
                "https://ai.baidu.com/ai-doc/AISTUDIO/slmkadt9z#2-%E5%A6%82%E4%BD%95%E4%BD%BF%E7%94%A8%E8%AE%BF%E9%97%AE%E4%BB%A4%E7%89%8C."
            )


def build_aistudio_headers(
    *,
    token: Optional[Union[bool, str]] = None,
    is_write_action: bool = False,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    user_agent: Union[Dict, str, None] = None,
) -> Dict[str, str]:
    # Get auth token to send
    token_to_send = get_token_to_send(token)
    _validate_token_to_send(token_to_send, is_write_action=is_write_action)

    # Combine headers
    headers = {"Content-Type": "application/json", "SDK-Version": str(VERSION)}
    if token_to_send is not None:
        headers["Authorization"] = f"token {token_to_send}"
    return headers


class OfflineModeIsEnabled(ConnectionError):
    """Raised when a request is made but `AISTUDIO_HUB_OFFLINE=1` is set as environment variable."""


class UniqueRequestIdAdapter(HTTPAdapter):
    X_AMZN_TRACE_ID = "X-Amzn-Trace-Id"

    def add_headers(self, request, **kwargs):
        super().add_headers(request, **kwargs)

        # Add random request ID => easier for server-side debug
        if X_AMZN_TRACE_ID not in request.headers:
            request.headers[X_AMZN_TRACE_ID] = request.headers.get(X_REQUEST_ID) or str(uuid.uuid4())

        # Add debug log
        has_token = str(request.headers.get("Authorization", "")).startswith("token")
        logger.debug(
            f"Request {request.headers[X_AMZN_TRACE_ID]}: {request.method} {request.url} (authenticated: {has_token})"
        )

    def send(self, request: PreparedRequest, *args, **kwargs) -> Response:
        """Catch any RequestException to append request id to the error message for debugging."""
        try:
            return super().send(request, *args, **kwargs)
        except requests.RequestException as e:
            request_id = request.headers.get(X_AMZN_TRACE_ID)
            if request_id is not None:
                # Taken from https://stackoverflow.com/a/58270258
                e.args = (*e.args, f"(Request ID: {request_id})")
            raise


class OfflineAdapter(HTTPAdapter):
    def send(self, request: PreparedRequest, *args, **kwargs) -> Response:
        raise OfflineModeIsEnabled(
            f"Cannot reach {request.url}: offline mode is enabled. To disable it, please unset the `AISTUDIO_HUB_OFFLINE` environment variable."
        )


def _default_backend_factory() -> requests.Session:
    session = requests.Session()
    if AISTUDIO_HUB_OFFLINE:
        session.mount("http://", OfflineAdapter())
        session.mount("https://", OfflineAdapter())
    else:
        session.mount("http://", UniqueRequestIdAdapter())
        session.mount("https://", UniqueRequestIdAdapter())
    return session


BACKEND_FACTORY_T = Callable[[], requests.Session]

_GLOBAL_BACKEND_FACTORY: BACKEND_FACTORY_T = _default_backend_factory
HTTP_METHOD_T = Literal["GET", "OPTIONS", "HEAD", "POST", "PUT", "PATCH", "DELETE"]


@lru_cache
def _get_session_from_cache(process_id: int, thread_id: int) -> requests.Session:
    """
    Create a new session per thread using global factory. Using LRU cache (maxsize 128) to avoid memory leaks when
    using thousands of threads. Cache is cleared when `configure_http_backend` is called.
    """
    return _GLOBAL_BACKEND_FACTORY()


def reset_sessions() -> None:
    """Reset the cache of sessions.

    Mostly used internally when sessions are reconfigured or an SSLError is raised.
    See [`configure_http_backend`] for more details.
    """
    _get_session_from_cache.cache_clear()


def get_session() -> requests.Session:
    """
    Get a `requests.Session` object, using the session factory from the user.

    Use [`get_session`] to get a configured Session. Since `requests.Session` is not guaranteed to be thread-safe,
    `huggingface_hub` creates 1 Session instance per thread. They are all instantiated using the same `backend_factory`
    set in [`configure_http_backend`]. A LRU cache is used to cache the created sessions (and connections) between
    calls. Max size is 128 to avoid memory leaks if thousands of threads are spawned.

    See [this issue](https://github.com/psf/requests/issues/2766) to know more about thread-safety in `requests`.

    Example:
    ```py
    import requests
    from huggingface_hub import configure_http_backend, get_session

    # Create a factory function that returns a Session with configured proxies
    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.proxies = {"http": "http://10.10.1.10:3128", "https": "https://10.10.1.11:1080"}
        return session

    # Set it as the default session factory
    configure_http_backend(backend_factory=backend_factory)

    # In practice, this is mostly done internally in `huggingface_hub`
    session = get_session()
    ```
    """
    return _get_session_from_cache(process_id=os.getpid(), thread_id=threading.get_ident())


def _request_wrapper(
    method: HTTP_METHOD_T, url: str, *, follow_relative_redirects: bool = False, **params
) -> requests.Response:
    """Wrapper around requests methods to follow relative redirects if `follow_relative_redirects=True` even when
    `allow_redirection=False`.

    Args:
        method (`str`):
            HTTP method, such as 'GET' or 'HEAD'.
        url (`str`):
            The URL of the resource to fetch.
        follow_relative_redirects (`bool`, *optional*, defaults to `False`)
            If True, relative redirection (redirection to the same site) will be resolved even when `allow_redirection`
            kwarg is set to False. Useful when we want to follow a redirection to a renamed repository without
            following redirection to a CDN.
        **params (`dict`, *optional*):
            Params to pass to `requests.request`.
    """
    # Recursively follow relative redirects
    if follow_relative_redirects:
        response = _request_wrapper(
            method=method,
            url=url,
            follow_relative_redirects=False,
            **params,
        )

        # If redirection, we redirect only relative paths.
        # This is useful in case of a renamed repository.
        if 300 <= response.status_code <= 399:
            parsed_target = urlparse(response.headers["Location"])
            if parsed_target.netloc == "":
                # This means it is a relative 'location' headers, as allowed by RFC 7231.
                # (e.g. '/path/to/resource' instead of 'http://domain.tld/path/to/resource')
                # We want to follow this relative redirect !
                #
                # Highly inspired by `resolve_redirects` from requests library.
                # See https://github.com/psf/requests/blob/main/requests/sessions.py#L159
                next_url = urlparse(url)._replace(path=parsed_target.path).geturl()
                return _request_wrapper(method=method, url=next_url, follow_relative_redirects=True, **params)
        return response

    # Perform request and return if status_code is not in the retry list.
    response = get_session().request(method=method, url=url, **params)
    hf_raise_for_status(response)
    return response


@dataclass(frozen=True)
class AistudioFileMetadata:
    """Data structure containing information about a file versioned on the Aistudio Hub.

    Returned by [`get_aistudio_file_metadata`] based on a URL.

    Args:
        commit_hash (`str`, *optional*):
            The commit_hash related to the file.
        etag (`str`, *optional*):
            Etag of the file on the server.
        location (`str`):
            Location where to download the file. Can be a Hub url or not (CDN).
        size (`size`):
            Size of the file. In case of an LFS file, contains the size of the actual
            LFS file, not the pointer.
    """

    commit_hash: Optional[str]
    etag: Optional[str]
    location: str
    size: Optional[int]


def _normalize_etag(etag: Optional[str]) -> Optional[str]:
    """Normalize ETag HTTP header, so it can be used to create nice filepaths.

    The HTTP spec allows two forms of ETag:
      ETag: W/"<etag_value>"
      ETag: "<etag_value>"

    For now, we only expect the second form from the server, but we want to be future-proof so we support both. For
    more context, see `TestNormalizeEtag` tests and https://github.com/huggingface/huggingface_hub/pull/1428.

    Args:
        etag (`str`, *optional*): HTTP header

    Returns:
        `str` or `None`: string that can be used as a nice directory name.
        Returns `None` if input is None.
    """
    if etag is None:
        return None
    return etag.lstrip("W/").strip('"')


def get_aistudio_file_metadata(
    url: str,
    token: Union[bool, str, None] = None,
    proxies: Optional[Dict] = None,
    timeout: Optional[float] = DEFAULT_REQUEST_TIMEOUT,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    user_agent: Union[Dict, str, None] = None,
) -> AistudioFileMetadata:
    """Fetch metadata of a file versioned on the Hub for a given url.

    Args:
        url (`str`):
            File url, for example returned by [`aistudio_hub_url`].
        token (`str` or `bool`, *optional*):
            A token to be used for the download.
                - If `True`, the token is read from the Aistudio config
                  folder.
                - If `False` or `None`, no token is provided.
                - If a string, it's used as the authentication token.
        proxies (`dict`, *optional*):
            Dictionary mapping protocol to the URL of the proxy passed to
            `requests.request`.
        timeout (`float`, *optional*, defaults to 10):
            How many seconds to wait for the server to send metadata before giving up.
        library_name (`str`, *optional*):
            The name of the library to which the object corresponds.
        library_version (`str`, *optional*):
            The version of the library.
        user_agent (`dict`, `str`, *optional*):
            The user-agent info in the form of a dictionary or a string.

    Returns:
        A [`AistudioFileMetadata`] object containing metadata such as location, etag, size and
        commit_hash.
    """
    headers = build_aistudio_headers(
        token=token, library_name=library_name, library_version=library_version, user_agent=user_agent
    )
    headers["Accept-Encoding"] = "identity"  # prevent any compression => we want to know the real size of the file

    # Retrieve metadata
    r = _request_wrapper(
        method="GET",
        url=url,
        headers=headers,
        allow_redirects=False,
        follow_relative_redirects=True,
        proxies=proxies,
        timeout=timeout,
    )
    hf_raise_for_status(r)
    res = r.json()

    # Return
    return AistudioFileMetadata(
        commit_hash=res["sha"],
        etag=_normalize_etag(res["last_commit_sha"]),
        location=res["git_url"],
        size=res["size"],
    )


def _cache_commit_hash_for_specific_revision(storage_folder: str, revision: str, commit_hash: str) -> None:
    """Cache reference between a revision (tag, branch or truncated commit hash) and the corresponding commit hash.

    Does nothing if `revision` is already a proper `commit_hash` or reference is already cached.
    """
    # if revision != commit_hash:
    ref_path = Path(storage_folder) / "refs" / revision
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    if not ref_path.exists() or commit_hash != ref_path.read_text():
        # Update ref only if has been updated. Could cause useless error in case
        # repo is already cached and user doesn't have write access to cache folder.
        # See https://github.com/huggingface/huggingface_hub/issues/1216.
        ref_path.write_text(commit_hash)


def _check_disk_space(expected_size: int, target_dir: Union[str, Path]) -> None:
    """Check disk usage and log a warning if there is not enough disk space to download the file.

    Args:
        expected_size (`int`):
            The expected size of the file in bytes.
        target_dir (`str`):
            The directory where the file will be stored after downloading.
    """

    target_dir = Path(target_dir)  # format as `Path`
    for path in [target_dir] + list(target_dir.parents):  # first check target_dir, then each parents one by one
        try:
            target_dir_free = shutil.disk_usage(path).free
            if target_dir_free < expected_size:
                warnings.warn(
                    "Not enough free disk space to download the file. "
                    f"The expected file size is: {expected_size / 1e6:.2f} MB. "
                    f"The target location {target_dir} only has {target_dir_free / 1e6:.2f} MB free disk space."
                )
            return
        except OSError:  # raise on anything: file does not exist or space disk cannot be checked
            pass


def http_get(
    url: str,
    temp_file: BinaryIO,
    *,
    proxies=None,
    resume_size: float = 0,
    headers: Optional[Dict[str, str]] = None,
    expected_size: Optional[int] = None,
    _nb_retries: int = 5,
):
    """
    Download a remote file. Do not gobble up errors, and will return errors tailored to the Hugging Face Hub.

    If ConnectionError (SSLError) or ReadTimeout happen while streaming data from the server, it is most likely a
    transient error (network outage?). We log a warning message and try to resume the download a few times before
    giving up. The method gives up after 5 attempts if no new data has being received from the server.
    """
    initial_headers = headers
    headers = copy.deepcopy(headers) or {}
    if resume_size > 0:
        headers["Range"] = "bytes=%d-" % (resume_size,)

    r = _request_wrapper(
        method="GET", url=url, stream=True, proxies=proxies, headers=headers, timeout=AISTUDIO_HUB_DOWNLOAD_TIMEOUT
    )
    hf_raise_for_status(r)
    content_length = r.headers.get("Content-Length")

    # NOTE: 'total' is the total number of bytes to download, not the number of bytes in the file.
    #       If the file is compressed, the number of bytes in the saved file will be higher than 'total'.
    total = resume_size + int(content_length) if content_length is not None else None

    displayed_name = url
    content_disposition = r.headers.get("Content-Disposition")
    if content_disposition is not None:
        match = HEADER_FILENAME_PATTERN.search(content_disposition)
        if match is not None:
            # Means file is on CDN
            displayed_name = match.groupdict()["filename"]

    # Truncate filename if too long to display
    if len(displayed_name) > 40:
        displayed_name = f"(…){displayed_name[-40:]}"

    consistency_error_message = (
        f"Consistency check failed: file should be of size {expected_size} but has size"
        f" {{actual_size}} ({displayed_name}).\nWe are sorry for the inconvenience. Please retry download and"
        " pass `force_download=True, resume_download=False` as argument.\nIf the issue persists, please let us"
        " know by opening an issue on https://github.com/huggingface/huggingface_hub."
    )

    # Stream file to buffer
    with tqdm(
        unit="B",
        unit_scale=True,
        total=total,
        initial=resume_size,
        desc=displayed_name,
        disable=bool(logger.getEffectiveLevel() == logging.NOTSET),
    ) as progress:
        new_resume_size = resume_size
        try:
            for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    progress.update(len(chunk))
                    temp_file.write(chunk)
                    new_resume_size += len(chunk)
                    # Some data has been downloaded from the server so we reset the number of retries.
                    _nb_retries = 5
        except (requests.ConnectionError, requests.ReadTimeout) as e:
            # If ConnectionError (SSLError) or ReadTimeout happen while streaming data from the server, it is most likely
            # a transient error (network outage?). We log a warning message and try to resume the download a few times
            # before giving up. Tre retry mechanism is basic but should be enough in most cases.
            if _nb_retries <= 0:
                logger.warning("Error while downloading from %s: %s\nMax retries exceeded.", url, str(e))
                raise
            logger.warning("Error while downloading from %s: %s\nTrying to resume download...", url, str(e))
            time.sleep(1)
            reset_sessions()  # In case of SSLError it's best to reset the shared requests.Session objects
            return http_get(
                url=url,
                temp_file=temp_file,
                proxies=proxies,
                resume_size=new_resume_size,
                headers=initial_headers,
                expected_size=expected_size,
                _nb_retries=_nb_retries - 1,
            )

        if expected_size is not None and expected_size != temp_file.tell():
            raise EnvironmentError(
                consistency_error_message.format(
                    actual_size=temp_file.tell(),
                )
            )


def _chmod_and_replace(src: str, dst: str) -> None:
    """Set correct permission before moving a blob from tmp directory to cache dir.

    Do not take into account the `umask` from the process as there is no convenient way
    to get it that is thread-safe.

    See:
    - About umask: https://docs.python.org/3/library/os.html#os.umask
    - Thread-safety: https://stackoverflow.com/a/70343066
    - About solution: https://github.com/huggingface/huggingface_hub/pull/1220#issuecomment-1326211591
    - Fix issue: https://github.com/huggingface/huggingface_hub/issues/1141
    - Fix issue: https://github.com/huggingface/huggingface_hub/issues/1215
    """
    # Get umask by creating a temporary file in the cached repo folder.
    tmp_file = Path(dst).parent.parent / f"tmp_{uuid.uuid4()}"
    try:
        tmp_file.touch()
        cache_dir_mode = Path(tmp_file).stat().st_mode
        os.chmod(src, stat.S_IMODE(cache_dir_mode))
    finally:
        tmp_file.unlink()

    shutil.move(src, dst)


def aistudio_download(
    repo_id: str,
    filename: str,
    *,
    subfolder: Optional[str] = None,
    repo_type: Optional[str] = None,
    revision: Optional[str] = None,
    library_name: Optional[str] = None,
    library_version: Optional[str] = None,
    cache_dir: Union[str, Path, None] = None,
    local_dir: Union[str, Path, None] = None,
    local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto",
    # TODO
    user_agent: Union[Dict, str, None] = None,
    force_download: bool = False,
    proxies: Optional[Dict] = None,
    etag_timeout: float = DEFAULT_ETAG_TIMEOUT,
    resume_download: bool = False,
    token: Optional[str] = None,
    local_files_only: bool = False,
    endpoint: Optional[str] = None,
    **kwargs,
):
    if AISTUDIO_HUB_ETAG_TIMEOUT != DEFAULT_ETAG_TIMEOUT:
        # Respect environment variable above user value
        etag_timeout = AISTUDIO_HUB_ETAG_TIMEOUT

    if cache_dir is None:
        cache_dir = AISTUDIO_HUB_CACHE
    if revision is None:
        revision = DEFAULT_REVISION
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if isinstance(local_dir, Path):
        local_dir = str(local_dir)
    locks_dir = os.path.join(cache_dir, ".locks")

    if subfolder == "":
        subfolder = None
    if subfolder is not None:
        # This is used to create a URL, and not a local path, hence the forward slash.
        filename = f"{subfolder}/{filename}"

    if repo_type is None:
        repo_type = "model"
    if repo_type not in REPO_TYPES:
        raise ValueError(f"Invalid repo type: {repo_type}. Accepted repo types are: {str(REPO_TYPES)}")

    storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type))
    os.makedirs(storage_folder, exist_ok=True)

    # cross platform transcription of filename, to be used as a local file path.
    relative_filename = os.path.join(*filename.split("/"))
    if os.name == "nt":
        if relative_filename.startswith("..\\") or "\\..\\" in relative_filename:
            raise ValueError(
                f"Invalid filename: cannot handle filename '{relative_filename}' on Windows. Please ask the repository"
                " owner to rename this file."
            )

    # if user provides a commit_hash and they already have the file on disk,
    # shortcut everything.
    # TODO, 当前不支持commit id下载，因此这个肯定跑的。
    if not force_download:  # REGEX_COMMIT_HASH.match(revision)
        pointer_path = _get_pointer_path(storage_folder, revision, relative_filename)
        if os.path.exists(pointer_path):
            if local_dir is not None:
                return _to_local_dir(pointer_path, local_dir, relative_filename, use_symlinks=local_dir_use_symlinks)
            return pointer_path

    url = aistudio_hub_url(repo_id, filename, repo_type=repo_type, revision=revision, endpoint=endpoint)

    headers = build_aistudio_headers(
        token=token,
        library_name=library_name,
        library_version=library_version,
        user_agent=user_agent,
    )
    url_to_download = url.replace("/contents/", "/media/")
    etag = None
    commit_hash = None
    expected_size = None
    head_call_error: Optional[Exception] = None
    if not local_files_only:
        try:
            try:
                metadata = get_aistudio_file_metadata(
                    url=url,
                    token=token,
                    proxies=proxies,
                    timeout=etag_timeout,
                    library_name=library_name,
                    library_version=library_version,
                    user_agent=user_agent,
                )
            except EntryNotFoundError as http_error:  # noqa: F841
                raise
            # Commit hash must exist
            # TODO，这里修改了commit hash，强迫为revision了。
            commit_hash = revision  # metadata.commit_hash
            if commit_hash is None:
                raise FileMetadataError(
                    "Distant resource does not seem to be on aistudio hub. It is possible that a configuration issue"
                    " prevents you from downloading resources from aistudio hub. Please check your firewall"
                    " and proxy settings and make sure your SSL certificates are updated."
                )

            # Etag must exist
            etag = metadata.etag
            # We favor a custom header indicating the etag of the linked resource, and
            # we fallback to the regular etag header.
            # If we don't have any of those, raise an error.
            if etag is None:
                raise FileMetadataError(
                    "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
                )

            # Expected (uncompressed) size
            expected_size = metadata.size

        except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
            # Actually raise for those subclasses of ConnectionError
            raise
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            OfflineModeIsEnabled,
        ) as error:
            # Otherwise, our Internet connection is down.
            # etag is None
            head_call_error = error
            pass
        except (RevisionNotFoundError, EntryNotFoundError):
            # The repo was found but the revision or entry doesn't exist on the Hub (never existed or got deleted)
            raise
        except requests.HTTPError as error:
            # Multiple reasons for an http error:
            # - Repository is private and invalid/missing token sent
            # - Repository is gated and invalid/missing token sent
            # - Hub is down (error 500 or 504)
            # => let's switch to 'local_files_only=True' to check if the files are already cached.
            #    (if it's not the case, the error will be re-raised)
            head_call_error = error
            pass
        except FileMetadataError as error:
            # Multiple reasons for a FileMetadataError:
            # - Wrong network configuration (proxy, firewall, SSL certificates)
            # - Inconsistency on the Hub
            # => let's switch to 'local_files_only=True' to check if the files are already cached.
            #    (if it's not the case, the error will be re-raised)
            head_call_error = error
            pass

    # etag can be None for several reasons:
    # 1. we passed local_files_only.
    # 2. we don't have a connection
    # 3. Hub is down (HTTP 500 or 504)
    # 4. repo is not found -for example private or gated- and invalid/missing token sent
    # 5. Hub is blocked by a firewall or proxy is not set correctly.
    # => Try to get the last downloaded one from the specified revision.
    #
    # If the specified revision is a commit hash, look inside "snapshots".
    # If the specified revision is a branch or tag, look inside "refs".
    if etag is None:
        # In those cases, we cannot force download.
        if force_download:
            raise ValueError(
                "We have no connection or you passed local_files_only, so force_download is not an accepted option."
            )

        # Try to get "commit_hash" from "revision"
        commit_hash = None
        if REGEX_COMMIT_HASH.match(revision):
            commit_hash = revision
        else:
            ref_path = os.path.join(storage_folder, "refs", revision)
            if os.path.isfile(ref_path):
                with open(ref_path) as f:
                    commit_hash = f.read()

        # Return pointer file if exists
        if commit_hash is not None:
            pointer_path = _get_pointer_path(storage_folder, commit_hash, relative_filename)
            if os.path.exists(pointer_path):
                if local_dir is not None:
                    return _to_local_dir(
                        pointer_path, local_dir, relative_filename, use_symlinks=local_dir_use_symlinks
                    )
                return pointer_path

        # If we couldn't find an appropriate file on disk, raise an error.
        # If files cannot be found and local_files_only=True,
        # the models might've been found if local_files_only=False
        # Notify the user about that
        if local_files_only:
            raise LocalEntryNotFoundError(
                "Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable"
                " aistudio hub look-ups and downloads online, set 'local_files_only' to False."
            )
        elif isinstance(head_call_error, RepositoryNotFoundError) or isinstance(head_call_error, GatedRepoError):
            # Repo not found => let's raise the actual error
            raise head_call_error
        else:
            # Otherwise: most likely a connection issue or Hub downtime => let's warn the user
            raise LocalEntryNotFoundError(
                "An error happened while trying to locate the file on the Hub and we cannot find the requested files"
                " in the local cache. Please check your connection and try again or make sure your Internet connection"
                " is on."
            ) from head_call_error

    # From now on, etag and commit_hash are not None.
    assert etag is not None, "etag must have been retrieved from server"
    assert commit_hash is not None, "commit_hash must have been retrieved from server"
    blob_path = os.path.join(storage_folder, "blobs", etag)
    pointer_path = _get_pointer_path(storage_folder, commit_hash, relative_filename)

    os.makedirs(os.path.dirname(blob_path), exist_ok=True)
    os.makedirs(os.path.dirname(pointer_path), exist_ok=True)
    # if passed revision is not identical to commit_hash
    # then revision has to be a branch name or tag name.
    # In that case store a ref.
    _cache_commit_hash_for_specific_revision(storage_folder, revision, commit_hash)

    if os.path.exists(pointer_path) and not force_download:
        if local_dir is not None:
            return _to_local_dir(pointer_path, local_dir, relative_filename, use_symlinks=local_dir_use_symlinks)
        return pointer_path

    if os.path.exists(blob_path) and not force_download:
        # we have the blob already, but not the pointer
        if local_dir is not None:  # to local dir
            return _to_local_dir(blob_path, local_dir, relative_filename, use_symlinks=local_dir_use_symlinks)
        else:  # or in snapshot cache
            _create_symlink(blob_path, pointer_path, new_blob=False)
            return pointer_path

    # Prevent parallel downloads of the same file with a lock.
    # etag could be duplicated across repos,
    lock_path = os.path.join(locks_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type), f"{etag}.lock")

    # Some Windows versions do not allow for paths longer than 255 characters.
    # In this case, we must specify it is an extended path by using the "\\?\" prefix.
    if os.name == "nt" and len(os.path.abspath(lock_path)) > 255:
        lock_path = "\\\\?\\" + os.path.abspath(lock_path)

    if os.name == "nt" and len(os.path.abspath(blob_path)) > 255:
        blob_path = "\\\\?\\" + os.path.abspath(blob_path)

    Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path):
        # If the download just completed while the lock was activated.
        if os.path.exists(pointer_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            if local_dir is not None:
                return _to_local_dir(pointer_path, local_dir, relative_filename, use_symlinks=local_dir_use_symlinks)
            return pointer_path

        if resume_download:
            incomplete_path = blob_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager() -> Generator[io.BufferedWriter, None, None]:
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
            logger.info("downloading %s to %s", url, temp_file.name)

            if expected_size is not None:  # might be None if HTTP header not set correctly
                # Check tmp path
                _check_disk_space(expected_size, os.path.dirname(temp_file.name))

                # Check destination
                _check_disk_space(expected_size, os.path.dirname(blob_path))
                if local_dir is not None:
                    _check_disk_space(expected_size, local_dir)

            http_get(
                url_to_download,
                temp_file,
                proxies=proxies,
                resume_size=resume_size,
                headers=headers,
                expected_size=expected_size,
            )
        if local_dir is None:
            logger.debug(f"Storing {url} in cache at {blob_path}")
            _chmod_and_replace(temp_file.name, blob_path)
            _create_symlink(blob_path, pointer_path, new_blob=True)
        else:
            local_dir_filepath = os.path.join(local_dir, relative_filename)
            os.makedirs(os.path.dirname(local_dir_filepath), exist_ok=True)

            # If "auto" (default) copy-paste small files to ease manual editing but symlink big files to save disk
            # In both cases, blob file is cached.
            is_big_file = os.stat(temp_file.name).st_size > AISTUDIO_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD
            if local_dir_use_symlinks is True or (local_dir_use_symlinks == "auto" and is_big_file):
                logger.debug(f"Storing {url} in cache at {blob_path}")
                _chmod_and_replace(temp_file.name, blob_path)
                logger.debug("Create symlink to local dir")
                _create_symlink(blob_path, local_dir_filepath, new_blob=False)
            elif local_dir_use_symlinks == "auto" and not is_big_file:
                logger.debug(f"Storing {url} in cache at {blob_path}")
                _chmod_and_replace(temp_file.name, blob_path)
                logger.debug("Duplicate in local dir (small file and use_symlink set to 'auto')")
                shutil.copyfile(blob_path, local_dir_filepath)
            else:
                logger.debug(f"Storing {url} in local_dir at {local_dir_filepath} (not cached).")
                _chmod_and_replace(temp_file.name, local_dir_filepath)
            pointer_path = local_dir_filepath  # for return value

    return pointer_path
