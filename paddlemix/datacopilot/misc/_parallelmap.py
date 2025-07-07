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


import concurrent.futures as futures
import math
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from tqdm import tqdm


class ParallelMode(Enum):
    THREAD = "multithread"
    PROCESS = "multiprocess"


def parallel_map(
    func: Callable,
    items: Sequence,
    *,
    max_workers: int = 8,
    chunk_size: int = 50000,
    mode: ParallelMode = ParallelMode.THREAD,
    progress: bool = True,
    order: bool = True
) -> List[Any]:
    """parallel exec"""
    if max_workers == 1:
        records = []
        for item in tqdm(items, disable=not progress):
            records.append(func(item))
        return records

    def _exec(executor: futures.Executor):
        records = []
        k = min(chunk_size, max_workers * 5000)
        with tqdm(total=len(items), disable=not progress) as p:
            for i in range(0, len(items), k):
                if order:
                    for result in executor.map(func, items[i : i + k]):
                        records.append(result)
                        p.update(1)
                else:
                    outputs = [executor.submit(func, x) for x in items[i : i + k]]
                    for task in futures.as_completed(outputs):
                        result = task.result()
                        records.append(result)
                        p.update(1)
        return records

    if mode == ParallelMode.THREAD:
        with futures.ThreadPoolExecutor(max_workers) as executor:
            return _exec(executor)

    elif mode == ParallelMode.PROCESS:
        with futures.ProcessPoolExecutor(max_workers) as executor:
            return _exec(executor)

    else:
        raise RuntimeError("")


def enumerate_chunk(
    items: Sequence,
    *,
    chunk_size: int = 1,
    num_chunks: Optional[int] = None,
    start: int = 0,
):
    if num_chunks is None:
        num_chunks = math.ceil(len(items) / chunk_size)
        indices = [i * chunk_size for i in range(num_chunks)] + [
            len(items),
        ]
    else:
        chunk_size = math.floor(len(items) / num_chunks)
        indices = [i * chunk_size for i in range(num_chunks)] + [
            len(items),
        ]

    for i in range(start, num_chunks):
        yield i, items[indices[i] : indices[i + 1]]


def list_dir(dir: str, *, pattern: str = "*", recursive: bool = True):
    if not recursive:
        return Path(dir).glob(pattern)
    else:
        return Path(dir).rglob(pattern)
