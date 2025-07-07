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


import errno
import hashlib
import io
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from PIL import Image
from tqdm import tqdm

from . import retry

__all__ = ["download_image", "open_image_from_url", "download_url_to_file"]


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
}


@contextmanager
def cache_url(url, headers=HEADERS, timeout=5):
    response = urlopen(Request(url=url, headers=headers), timeout=timeout)
    with io.BytesIO() as f:
        block_sz = 1024 * 10
        while True:
            buffer = response.read(block_sz)
            if not buffer:
                break
            f.write(buffer)
        yield f


def download_image(url: str, path: str, timeout: int = 10, max_retries: int = 3) -> Dict[str, str]:
    result = {
        "status": "",
        "url": url,
        "path": path,
    }
    cnt = 0
    while True:
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with cache_url(url, timeout=timeout) as f:
                with open(path, "wb") as fp:
                    fp.write(f.getvalue())
            result["status"] = "success"
        except Exception as e:
            if not isinstance(e, HTTPError):
                cnt += 1
                if cnt <= max_retries:
                    continue
            if isinstance(e, HTTPError):
                result["status"] = "expired"
            else:
                result["status"] = str(e)
        break
    return result


@retry(max_trials=3, delay=0.1, suppress_exceptions=False)
def open_image_from_url(url: str, timeout: int = 10) -> Image.Image:
    """open image from url or local path"""
    if not urlparse(url).scheme:
        return Image.open(url)
    else:
        # return Image.open(BytesIO(requests.get(url).content))
        with cache_url(url, timeout=timeout) as f:
            return Image.open(f)


def download_url_to_file(url: str, dst: str, hash_prefix: Optional[str] = None, progress: bool = True) -> None:
    r"""Download object at the given URL to a local path.
    Args:
        url (str): URL of the object to download
        dst (str): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_prefix (str, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.
            Default: None
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True

    Reference:
        https://github.com/pytorch/pytorch/blob/main/torch/hub.py
    """
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    # We deliberately do not use NamedTemporaryFile to avoid restrictive
    # file permissions being applied to the downloaded file.
    dst = os.path.expanduser(dst)
    for seq in range(tempfile.TMP_MAX):
        tmp_dst = dst + "." + uuid.uuid4().hex + ".partial"
        try:
            f = open(tmp_dst, "w+b")
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(errno.EEXIST, "No usable temporary file name found")

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(128 * 1024)
                if len(buffer) == 0:
                    break
                f.write(buffer)  # type: ignore[possibly-undefined]
                if hash_prefix is not None:
                    sha256.update(buffer)  # type: ignore[possibly-undefined]
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()  # type: ignore[possibly-undefined]
            if digest[: len(hash_prefix)] != hash_prefix:
                raise RuntimeError(f'invalid hash value (expected "{hash_prefix}", got "{digest}")')
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)
