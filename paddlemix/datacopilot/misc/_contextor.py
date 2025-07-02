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


import random
import shutil
import tempfile
from contextlib import contextmanager
from typing import Iterator, Optional

import numpy as np


@contextmanager
def freeze_rng_state(seed: Optional[int] = None):
    state = random.getstate()
    np_state = np.random.get_state()
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    yield
    np.random.set_state(np_state)
    random.setstate(state)


@contextmanager
def open_tmp_dir() -> Iterator[str]:
    dir = tempfile.mkdtemp()
    try:
        yield dir
    finally:
        shutil.rmtree(dir)


@contextmanager
def open_tmp_file():
    with tempfile.NamedTemporaryFile(delete=True) as f:
        yield f
